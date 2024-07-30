
import torch
import torch.nn as nn
from src.client.FedBase_client import FedBase

import os
class Fed_Avg_Client(FedBase):

    def __init__(self, args, i, model, loss, train_set, test_set, data_ratio, device):
        super().__init__(args,i, model, loss, train_set, test_set, data_ratio, device)
        
    

    def local_train(self):
        
        self.local_model.train()
        if self.id == 2:
            for param in self.local_model.parameters():
                print(f"local model before local iter : {param}")

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
            #print(f"loss at user {self.id} at epoch : {iter} : {total_loss/len(self.trainloader)} ")

        """if self.id == 2:
            for param in self.local_model.parameters():
                print(f"local model after local iter : {param}")
        """