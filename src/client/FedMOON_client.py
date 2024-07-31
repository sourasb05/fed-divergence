import copy
import torch
import torch.nn as nn
from src.client.FedBase_client import FedBase

class FedMOON_Client(FedBase):

    def __init__(self, args, i, model, loss, train_set, test_set, data_ratio, device, current_directory):
        super().__init__(args, i, model, loss, train_set, test_set, data_ratio, device, current_directory)
        self.global_model = copy.deepcopy(model)
        self.prev_model = copy.deepcopy(model)
        self.temperature = args.temperature
        self.mu = args.mu
    
    def initialize_previous_model(self):
        for prev_param, curr_param in zip(self.prev_model.parameters(), self.local_model.parameters()):
            prev_param.data = curr_param.data.clone()
            # prev_param.grad.data = curr_param.grad.data.clone()    

    def initialize_global_model(self, global_model_params):
        for prev_global_param, curr_global_param in zip(self.global_model.parameters(),  global_model_params):
            prev_global_param.data = curr_global_param.data.clone()
            #prev_global_param.grad.data = curr_global_param.grad.data.clone()          
    
    def local_train(self, global_model_param, t):
        self.initialize_previous_model()
        self.initialize_global_model(global_model_param)
        self.local_model.train()
        
        cnt=0
        cos=torch.nn.CosineSimilarity(dim=-1)

        for iter in range(0, self.local_iters):
            """epoch_loss_collector = []
            epoch_loss1_collector = []
            epoch_loss2_collector = []"""
            for inputs, target in self.trainloader:
                inputs, target = inputs.to(self.device), target.to(self.device)
                self.optimizer.zero_grad()
                inputs.requires_grad = False
                target.requires_grad = False
                target = target.long()

                _, pro1, out = self.local_model(inputs)
                _, pro2, _ = self.global_model(inputs)
    
                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                posi = cos(pro1, pro2)
                logits = posi.reshape(-1,1)

                _, pro3, _ = self.prev_model(inputs)
                nega = cos(pro1, pro3)
                logits = torch.cat((logits, nega.reshape(-1,1)), dim=1)

                logits /= self.temperature
                labels = torch.zeros(inputs.size(0)).cuda().long()

                loss2 = self.mu * self.criterion(logits, labels)

                loss1 = self.criterion(out, target)
                loss = loss1 + loss2

                loss.backward()
                self.optimizer.step()
                

                cnt += 1
            self.test_local(t)
                
"""

            epoch_loss_collector.append(loss.item())
            epoch_loss1_collector.append(loss1.item())
            epoch_loss2_collector.append(loss2.item())

        epoch_loss = sum(epoch_loss_collector) / len(epoch_loss_collector)
        epoch_loss1 = sum(epoch_loss1_collector) / len(epoch_loss1_collector)
        epoch_loss2 = sum(epoch_loss2_collector) / len(epoch_loss2_collector)
        logger.info('Epoch: %d Loss: %f Loss1: %f Loss2: %f' % (epoch, epoch_loss, epoch_loss1, epoch_loss2))

"""
    
