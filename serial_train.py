import gc
gc.disable()
import os
import sys
import datetime
import time
import pickle
import base64
import numpy as np
import json
from ctypes import cdll
import ctypes

from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
import io
import modelnet
import datasource
from statistics import mean

NUM_GLOBAL_ROUNDS = 2
NUM_LOCAL_EPOCHS = 10 # at each local node
NUM_TEST_EPOCH = 4

#training variables  -------------
lr = 0.001
# ------------------------

if not torch.cuda.is_available():
    from torchsummary import summary

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')



class GlobalModel(object):
    def __init__(self, num_peers, identifier,type):
        self.global_weights = []
        self.aggregated_weight = 0
        self.current_round = 0
        self.global_model = LocalModel(0,datasource.DataSetFactory(type),'global_model')
        # iid or non-iid
        self.data_split_type = type 
        self.num_nodes = num_peers
        self.local_models = []
        self.total_train_time = 0
        self.id = identifier
        self.global_loss = []
        self.global_acc = []
        self.best_acc = 0
        

    def save_model(self, model, name):
        torch.save(model.state_dict(), 'saved_model/'+name)
    
    def load_weights(self, model, weights):
        model.load_state_dict(weights)

    def start_global_train(self):
        ## initialize peer nodes
        for i in range(self.num_nodes):
            node_id = i+1
            local_model = LocalModel(node_id, datasource.DataSetFactory(self.data_split_type), self.id)
            self.local_models.append(local_model)
        
        ## start global epochs
        for round_num in range(NUM_GLOBAL_ROUNDS):
            print('Starting GLOBAL round: '+str(round_num+1))
            self.current_round = round_num+1
            if round_num > 0:
                for lm in self.local_models:
                    lm.model.load_state_dict(self.aggregated_weight)
                    lm.current_weights = self.aggregated_weight
            
            ## start local epochs
            local_model_train_times = []
            for curr_model in range(self.num_nodes):
                model_state, curr_loss, curr_acc = self.local_models[curr_model].train_one_round()
                self.local_models[curr_model].current_weights = model_state

            for curr_model in range(self.num_nodes):
                local_model_train_times.append(self.local_models[curr_model].train_time)
            self.total_train_time += max(local_model_train_times)
            print('**** DONE training for ROUND: '+str(self.current_round))


            # aggregation
            print('#nodes:',len(self.local_models))
            agg_start = datetime.datetime.now()
            global_dict = self.local_models[0].current_weights
            total_size = 0
            n = self.num_nodes
            for lm in self.local_models:
                total_size += lm.x_train
            for k in global_dict.keys():
                global_dict[k] = torch.stack([self.local_models[i].current_weights[k].float()*(n*self.local_models[i].x_train/total_size) for i in range(n)], 0).mean(0)
            # for lm in self.local_models:
            #     lm.model.load_state_dict(global_dict)
            #     lm.current_weights = global_dict
            self.aggregated_weight = global_dict
            agg_end = datetime.datetime.now()
            self.total_train_time += (agg_end-agg_start).seconds

            self.global_model.model.load_state_dict(self.aggregated_weight)
            _, val_loss, val_acc = self.global_model.test('test')
            self.global_loss.append(val_loss)
            self.global_acc.append(val_acc)
            print('current_G_LOSS:',val_loss)
            print('current_G_ACC:',val_acc)
            if val_acc > self.best_acc:
                self.best_acc = val_acc
                torch.save(self.global_model.model.state_dict(), "saved_models/global_model-"+str(int(val_acc)))

        print('G_LOSS', self.global_loss)
        print('G_ACC', self.global_acc)
        print('TOTAL_TRAIN_TIME',self.total_train_time)
        print(" ====== DONE global training for all epochs")



class LocalModel(object):
    def __init__(self, id, data, marker):
        self.train_losses = []
        self.train_accs = []
        self.valid_aucs = []
        self.valid_accs = []

        self.counter_flag = 1

        self.model_id = id
        self.marker=marker

        self.datasource = data
        self.model = modelnet.siamese_model().to(device)
       
        # define loss function and optimizer
        self.optimizer = optim.Adam(self.model.parameters(), lr=lr)
        self.criterion = nn.BCELoss()
	
        self.x_train = self.datasource.num_train
        self.x_valid = self.datasource.num_valid
        self.x_test = self.datasource.num_test

        self.current_weights = None
        self.train_time = 0

        self.training_start_time = int(round(time.time()))


    def load_weights(self, state):
        self.model.load_state_dict(state)

    def train_one_round(self):
        print("starting local training round for MODEL: "+str(self.model_id))

        loss_history = [[], []]
        accuracy_history = [[], []]
        val_accuracy = []
        val_loss = []
        train_n_minibatches = self.datasource.train_loader.__len__()
        train_start_time = datetime.datetime.now()

        best_acc = 0
        best_loss = 0

        for epoch in range(NUM_LOCAL_EPOCHS):
            self.model.train()
            for batch_idx, x in enumerate(self.datasource.train_loader,0):
                if(batch_idx%2 == 0):
                    print(f"epoch: {epoch+1} batch: {batch_idx+1}")
                self.optimizer.zero_grad()
                x, y, z = x[0], x[1], x[2]
                y_pred = self.model(x, y)

                # Calculating Loss
                loss = self.criterion(y_pred, z)
                loss.backward()
                self.optimizer.step()
                loss_history[0].append(float(loss.detach()))

                # Calaculating Accuracy
                correct = 0
                y_pred = y_pred.to(device).detach().numpy().tolist()
                y = z.cpu().detach().numpy().tolist()
                for i, j in zip(y, y_pred):
                    if round(j[0]) == int(i[0]):
                        correct = correct + 1
                accuracy_history[0].append((correct / x.shape[0]) * 100)

            loss_history[1].append(
                sum(loss_history[0][-1 : -train_n_minibatches - 1 : -1]) / train_n_minibatches
            )
            accuracy_history[1].append(
                sum(accuracy_history[0][-1 : -train_n_minibatches - 1 : -1])
                / train_n_minibatches
            )

            _,v_loss,v_acc = self.test('test')
            val_accuracy.append(v_acc);
            val_loss.append(v_loss);

            if best_acc < accuracy_history[1][-1]:
                best_acc = accuracy_history[1][-1]
                best_loss = loss_history[1][-1]
                torch.save(self.model.state_dict(), "saved_models/local_model"+str(self.marker)+':'+str(self.model_id))

            if (epoch + 1) % 1 == 0:
                print(
                    f"---------------------------------------EPOCH {epoch+1}-------------------------------------------"
                )
                print(f"Loss for EPOCH {epoch+1}  TRAIN LOSS : {loss_history[1][-1]}", end=" ")
                print(f"TRAIN ACCURACY : {accuracy_history[1][-1]}")
                print(f"VAL_ACC: {v_acc} VAL_LOSS: {v_loss}")
                print(
                    f"---------------------------------------------------------------------------------------------"
                )
            
        print(f"\n After {epoch} epochs BEST ACCURACY : {best_acc}%  BEST LOSS : {best_loss}")
        total_validation_loss = np.mean(loss_history[1])
        total_validation_accuracy = np.mean(accuracy_history[1])
        # return self.model.state_dict(), total_validation_loss,  total_validation_accuracy
        train_end_time = datetime.datetime.now()
        self.train_time += (train_end_time - train_start_time).seconds
        return self.model.state_dict(), loss_history[1][-1], accuracy_history[1][-1]
        
    def test(self, split):
        print('starting evaluting the model on split: '+split)
        self.model.eval()

        data_loader = self.datasource.valid_loader if split == 'valid' else self.datasource.test_loader
        loss_history = [[], []]
        accuracy_history = [[], []]
        test_n_minibatches = data_loader.__len__()

        for epoch in range(1):
            with torch.no_grad():
                for batch_idx, x in enumerate(data_loader,0):
                    x, y, z = x[0], x[1], x[2]
                    y_pred = self.model(x, y)

                    # Calculating Loss
                    loss = self.criterion(y_pred, z)
                    self.optimizer.step()
                    loss_history[0].append(float(loss.detach()))

                    # Calaculating Accuracy
                    correct = 0
                    y_pred = y_pred.to(device).detach().numpy().tolist()
                    y = z.cpu().detach().numpy().tolist()
                    for i, j in zip(y, y_pred):
                        if round(j[0]) == int(i[0]):
                            correct = correct + 1
                    accuracy_history[0].append((correct / x.shape[0]) * 100)

                loss_history[1].append(
                    sum(loss_history[0][-1 : -test_n_minibatches - 1 : -1]) / test_n_minibatches
                )
                accuracy_history[1].append(
                    sum(accuracy_history[0][-1 : -test_n_minibatches - 1 : -1])
                    / test_n_minibatches
                )

                print('Local Epoch [%d/%d] %s Loss: %.3f, Accuracy: %.3f' % (
                        epoch + 1, 1, split, loss_history[1][-1], accuracy_history[1][-1]))
        total_validation_loss = np.mean(loss_history[1])
        total_validation_accuracy = np.mean(accuracy_history[1])
        return self.model.state_dict(), total_validation_loss, total_validation_accuracy

    def validate(self):
        print('start validation')
        split, loss, acc = self.test('valid')
        print('%s  loss: %.3f  acc:%.3f' % (split, loss, acc))

        return loss, acc

    def evaluate(self):
        print('start evaluation')
        split, loss, acc = self.test('test')
        print('%s  loss: %.3f  acc:%.3f' % (split, loss, acc))













global_model = GlobalModel(6,'6_peers_iid','iid')
global_model.start_global_train()