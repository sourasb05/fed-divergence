
from tqdm import trange
import datetime
import wandb
import matplotlib.pyplot as plt
from src.client.FedAvg_client import *
from src.utils.utils import *
import numpy as np
import torch.nn.init as init
from src.server.FedBase_server import Base_server
import sys
# implementation of FedAvg server


class Fed_Avg_Server(Base_server):
    def __init__(self, args, model, loss, device, curr_dir):
        super().__init__(args, model, loss, device, curr_dir)
        # for param in self.global_model.parameters():
        #    print(f"from fedavg : {param}")
        #print(self.global_model.parameters())
        # sys.exit()
        data = read_data(args)
        print(data[0])
        total_users = len(data[0])

        for i in range(0,total_users):
            train, test = read_user_data(i,data,args.dataset)
            data_ratio = len(data[1])/len(train)
            # print("data_ratio",data_ratio) ## The ratio is not fixed yet
            user = Fed_Avg_Client(args,i,model,loss,train,test,data_ratio,device)   # Creating the instance of the users. 
            self.users.append(user)
        
    def global_update(self):
        for param in self.global_model.parameters():
            if param.requires_grad:
                init.zeros_(param) 
        
        N = len(self.selected_users)

        for user in self.selected_users:
            print(user.data_ratio)
            for g_param, l_param in  zip(self.global_model.parameters(), user.local_model.parameters()):
                g_param.data = g_param.data + 0.1 * l_param.data.clone()
                

    def train(self):
        
        for t in trange(self.global_iters):
            print(len(self.users))
            print(self.num_users_perGR)
            self.selected_users = self.select_users( t, self.num_users_perGR)
            
            self.send_parameters()   # server send parameters to every users
            
            print("number of selected users",len(self.selected_users))
            for user in self.selected_users:
                user.local_train()
            
            # self.initialize_parameters_to_zero()  # Because we are averaging parameters
            for param in self.global_model.parameters():
                print(f"from fedavg before aggtegation at round {t} : {param}")
            self.global_update()
            for param in self.global_model.parameters():
                print(f"from fedavg after aggtegation  at round {t} : {param}")

            self.evaluate(t, "global")  # evaluate global model
                
                
