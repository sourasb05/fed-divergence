
import torch
import torch.nn as nn
from src.client.FedBase_client import FedBase

import os
class Fed_Avg_Client(FedBase):

    def __init__(self, args, i, model, loss, train_set, test_set, data_ratio, device):
        super().__init__(args,i, model, loss, train_set, test_set, data_ratio, device)
        
    

    def local_train(self):
        
        self.local_model.train()
        for iters in range(0, self.local_iters):
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.local_model(inputs)
                loss = self.loss(output, labels)
                loss.backward()
                self.optimizer.step()
        

