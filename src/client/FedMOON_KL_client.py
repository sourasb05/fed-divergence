import copy
import torch
import math
import torch.nn as nn
import torch.nn.functional as F

from src.client.FedBase_client import FedBase

class FedMOON_KL_Client(FedBase):

    def __init__(self, args, i, model, loss, train_set, test_set, data_ratio, device):
        super().__init__(args, i, model, loss, train_set, test_set, data_ratio, device)
        self.global_model = copy.deepcopy(model)
        self.prev_model = copy.deepcopy(model)
        self.temperature = args.temperature
        self.mu = args.mu
        self.lamda = args.lamda
    
    def initialize_previous_model(self):
        for prev_param, curr_param in zip(self.prev_model.parameters(), self.local_model.parameters()):
            prev_param.data = curr_param.data.clone()
            # prev_param.grad.data = curr_param.grad.data.clone()    

    def initialize_global_model(self, global_model_params):
        for prev_global_param, curr_global_param in zip(self.global_model.parameters(),  global_model_params):
            prev_global_param.data = curr_global_param.data.clone()
            #prev_global_param.grad.data = curr_global_param.grad.data.clone()  

    
    def kl_divergence(self, p, q):
        """Kullback-Leibler divergence for two distributions"""
        return torch.sum(p * torch.log(p / (q + 1e-10) + 1e-10), dim=-1)

    def js_divergence(self, p, q):
        """Jensen-Shannon divergence for two distributions"""
        m = 0.5 * (p + q)
        return 0.5 * (self.kl_divergence(p, m) + self.kl_divergence(q, m))

    def calculate_JSD(self):
        total_jsd_old_global = 0.0
        total_jsd_new_global = 0.0
        total_jsd_new_old = 0.0
    
        for old_param, new_param, global_param in zip(self.prev_model.parameters(), self.local_model.parameters(), self.global_model.parameters()):
            # Flatten the parameters
            old_param_flat = old_param.view(-1)
            new_param_flat = new_param.view(-1)
            global_param_flat = global_param.view(-1)
            
            # Normalize the parameters to form probability distributions
            old_prob = F.softmax(old_param_flat, dim=0)   # z_prev
            new_prob = F.softmax(new_param_flat, dim=0)   # z_new
            global_prob = F.softmax(global_param_flat, dim=0)  # z_global
            
            # Calculate Jensen-Shannon Divergence
            jsd_old_global = self.js_divergence(old_prob, global_prob)  # >
            jsd_new_global = self.js_divergence(new_prob, global_prob)  # <
            jsd_new_old = self.js_divergence(new_prob, old_prob)  # Changed from KL divergence to JSD

            # Add it to the total
            total_jsd_old_global += jsd_old_global
            total_jsd_new_global += jsd_new_global
            total_jsd_new_old += jsd_new_old
        
        # Calculate the total entropy as given in the original function
        total_entropy = -math.log(math.exp(total_jsd_new_global) / (math.exp(total_jsd_old_global) + math.exp(total_jsd_new_old)))
        
        return total_entropy

    
    def calculate_entropies(self):
        total_entropy_old_global = 0.0
        total_entropy_new_global = 0.0
        total_entropy_new_old = 0.0
        total_entropy = 0.0
    
        for old_param, new_param, global_param in zip(self.prev_model.parameters(), self.local_model.parameters(), self.global_model.parameters()):
            # Flatten the parameters
            old_param_flat = old_param.view(-1)
            new_param_flat = new_param.view(-1)
            global_param_flat = global_param.view(-1)
            
            # Normalize the parameters to form probability distributions
            old_prob = F.softmax(old_param_flat, dim=0)   #z_prev
            new_prob = F.softmax(new_param_flat, dim=0)   #z_newm
            global_prob = F.softmax(global_param_flat, dim=0)  #z_global
            
            # Calculate relative entropy using KL divergence
            entropy_old_global = F.kl_div(old_prob.log(), global_prob, reduction='batchmean')  # >
            entropy_new_global = F.kl_div(new_prob.log(), global_prob, reduction='batchmean')  # <
            # entropy_old_new = F.kl_div(old_prob.log(), new_prob, reduction='batchmean')
            entropy_new_old = F.kl_div(new_prob.log(), old_prob, reduction='batchmean')

            
            # Add it to the total
            total_entropy_old_global += entropy_old_global
            total_entropy_new_global += entropy_new_global
            total_entropy_new_old += entropy_new_old
        
       
        total_entropy = -math.log(math.exp(total_entropy_new_global)/(math.exp(total_entropy_old_global) + math.exp(total_entropy_new_old)))
        
        return total_entropy        
    
    def local_train(self, global_model_param):
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
                

                #loss3 = self.calculate_entropies()
                loss3 =self.calculate_JSD()

                loss = loss1 + (1-self.lamda)*loss2 + self.lamda*loss3
                
                loss.backward()
                self.optimizer.step()

                cnt += 1


    
