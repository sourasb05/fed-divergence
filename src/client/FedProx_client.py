import copy
import math
import torch
import torch.nn as nn
import random
from torch.utils.data import DataLoader
from src.client.FedBase_client import FedBase

class FedProx_Client(FedBase):

    def __init__(self, args, i, model, loss, train_set, test_set, data_ratio, device):
        super().__init__(args, i, model, loss, train_set, test_set, data_ratio, device)
        
    

    def local_train(self, global_model_param):
        
        self.local_model.train()
        for iters in range(0, self.local_iters):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.local_model(inputs)
                loss = self.loss(output, labels)
                
                proximal_term = 0.0
                for param, g_param in zip(self.local_model.parameters(), global_model_param):
                    proximal_term += (self.lambda_prox / 2) * torch.norm(param - g_param) ** 2
                loss += proximal_term
                loss.backward()
                self.optimizer.step()

    def update_parameters(self, global_model):
           for l_param, g_param in zip(self.local_model.parameters(), global_model):
                l_param.data = g_param.data.clone()


    def global_eval_train_data(self, global_model):
        self.local_model.eval()
        train_correct = 0
        loss = 0
        self.update_parameters(global_model)
        for x, y in self.trainloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            train_correct += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)
        return train_correct, y.shape[0], loss

    def global_eval_test_data(self, global_model):
        self.local_model.eval()
        test_correct = 0
        loss = 0
        self.update_parameters(global_model)
        for x, y in self.testloaderfull:
            x, y = x.to(self.device), y.to(self.device)
            output = self.local_model(x)
            test_correct += (torch.sum(torch.argmax(output, dim=1) == y)).item()
            loss += self.loss(output, y)

        return test_correct, y.shape[0], loss