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
        for iter in range(0, self.local_iters):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.local_model(inputs)
                loss = self.criterion(output, labels)
                
                proximal_term = 0.0
                for param, g_param in zip(self.local_model.parameters(), global_model_param):
                    proximal_term += (self.lambda_prox / 2) * torch.norm(param - g_param) ** 2
                loss += proximal_term
                loss.backward()
                self.optimizer.step()
