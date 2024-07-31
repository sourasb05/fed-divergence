import torch 
import os
import copy
import h5py
import numpy as np
import pandas as pd
from tqdm import trange
from datetime import date
import matplotlib.pyplot as plt
from src.client.FedProx_client import FedProx_Client
from src.server.FedBase_server import Base_server
from src.utils.utils import select_users, read_data, read_user_data 
import numpy as np
import torch.nn.init as init

# implementation of FedAvg server


class FedProx_Server(Base_server):

    def __init__(self, args, model, loss, device, curr_dir):
        super().__init__(args, model, loss, device, curr_dir)

        data = read_data(args)
        total_users = len(data[0])


                
        for i in range(0,total_users):
            train, test = read_user_data(i,data,args.dataset)
            data_ratio = 1/total_users
            # print("data_ratio",data_ratio) ## The ratio is not fixed yet
            user = FedProx_Client(args,i, model, loss, train, test, data_ratio, device, curr_dir)   # Creating the instance of the users. 
        
            self.users.append(user)
        
    def global_update(self): 
        
        
        N = len(self.selected_users)

        for user in self.selected_users:
            for g_param, l_param in  zip(self.global_model.parameters(), user.local_model.parameters()):
                g_param.data = g_param.data + user.data_ratio * l_param.data
    


    def train(self):
        
        for t in trange(self.global_iters):
            print(len(self.users))
            print(self.num_users_perGR)
            self.selected_users = self.select_users( t, self.num_users_perGR)
            self.send_parameters()   # server send parameters to every users
            
            print("number of selected users",len(self.selected_users))
            for user in self.selected_users:
                user.local_train(self.global_model.parameters(), t)
            
            self.initialize_parameters_to_zero()  # Because we are averaging parameters
            self.global_update()
            self.evaluate(t, "global")  # evaluate global model
                
                

                
