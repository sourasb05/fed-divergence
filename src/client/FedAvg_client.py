
import torch
import torch.nn as nn
from src.client.FedBase_client import FedBase

import os
class Fed_Avg_Client(FedBase):

    def __init__(self, args, i, model, loss, train_set, test_set, data_ratio, device, current_directory):
        super().__init__(args,i, model, loss, train_set, test_set, data_ratio, device, current_directory)
        
    

    def local_train(self, t):
        
        self.local_model.train()

        for iter in range(0, self.local_iters):
            total_loss=0.0
            for inputs, labels in self.trainloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                self.optimizer.zero_grad()
                output = self.local_model(inputs)
                loss = self.criterion(output, labels)
                loss.backward()
                total_loss += loss
                self.optimizer.step()
            self.test_local(t)
