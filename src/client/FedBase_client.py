import copy
import os
import numpy as np
import math
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader

from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score


class FedBase:
    """
    Base class for users in FL
    """

    def __init__(self, args, i, model,loss,train_set,test_set,data_ratio,device):

        self.local_model = copy.deepcopy(model)
        
        self.train_samples = len(train_set)
        self.test_samples = len(test_set)
        self.local_iters = args.local_iters
        self.learning_rate = args.lr
        self.lambda_prox = args.lambda_prox
        self.batch_size = args.batch_size
        self.device = device
        self.loss = loss
        self.data_ratio = data_ratio
        self.participation_prob = 1.0
        self.minimum_test_loss = 10000.0
        self.optimizer_name = args.optimizer
        self.id = i
        #self.p = args.p
        
        self.trainloader = DataLoader(train_set, self.batch_size)
        self.trainloaderfull = DataLoader(train_set, self.train_samples)
        
        self.testloader =  DataLoader(test_set, self.test_samples)
        self.optimizer = optim.Adam(self.local_model.parameters(), lr=self.learning_rate)
        
       

    def selection(self):
        outcomes = [0,1]
        weights = [1-self.participation_prob, self.participation_prob]
        participation_choice = random.choices(outcomes, weights=weights)[0]

        return participation_choice
    
    

    def set_parameters(self, glob_model):
        glob_model_param=glob_model.parameters()
        # Store the initial local model parameters
        if isinstance(glob_model_param, nn.Parameter):
            print("It is nn.Param")
            for l_param, g_param in zip(self.local_model.parameters(), glob_model_param):
                l_param.data = g_param.data.clone()
        elif isinstance(glob_model_param, list):
            print("It is a list")
            for idx, l_param in enumerate(self.local_model.parameters()):
                l_param.data = glob_model_param[idx].clone()

            
    def get_parameters(self):
        for param in self.local_model.parameters():
            param.detach()
        return self.local_model.parameters()

    def clone_model_paramenter(self, param, clone_param):
        for param, clone_param in zip(param, clone_param):
            clone_param.data = param.data.clone()
        return clone_param

    def get_updated_parameters(self):
        return self.local_weight_updated

    def update_parameters(self, new_params):
        for param, new_param in zip(self.local_model.parameters(), new_params):
            param.data = new_param.data.clone()
    
    def save_model(self, glob_iter, current_loss):
        if glob_iter == self.num_glob_iters-1:
            model_path = self.current_directory + "/models/" + self.algorithm + "/local_model/" + str(self.id) + "/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.local_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "local_checkpoint_GR" + str(glob_iter) + ".pt"))
            
        if current_loss < self.minimum_test_loss:
            self.minimum_test_loss = current_loss
            model_path = self.current_directory + "/models/" + self.algorithm + "/local_model/" + str(self.id) + "/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.local_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "best_local_checkpoint" + ".pt"))

    
    def load_model(self):
        models_dir = self.current_directory + "/models/" + self.algorithm + "/local_model/" + str(self.id) + "/"
        model_state_dict = torch.load(os.path.join(models_dir, str(self.id), "best_local_checkpoint.pt"))["model_state_dict"]
        self.local_model.load_state_dict(model_state_dict)
        self.local_model.eval()

    
    
    def test_el(self, global_model=None):
        self.local_model.eval()
        if global_model!=None:
            self.update_parameters(global_model)
        
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.testloader:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        validation_loss = total_loss / len(self.testloader)
        precision = precision_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        recall = recall_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
        f1 = f1_score(y_true, y_pred, average='weighted', zero_division=0)  # Use 'macro' for unweighted
                
        return accuracy, validation_loss, precision, recall, f1

    def train_el(self, global_model=None):
        self.local_model.eval()
        y_true = []
        y_pred = []
        
        total_loss = 0.0
        if global_model !=None:
            self.update_parameters(global_model)
        
        with torch.no_grad():  # Inference mode, gradients not needed
            for inputs, labels in self.trainloaderfull:
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                outputs = self.local_model(inputs)
                loss = self.loss(outputs, labels)
                total_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                y_true.extend(labels.cpu().numpy())  # Collect true labels
                y_pred.extend(predicted.cpu().numpy())  # Collect predicted labels

        # Convert collected labels to numpy arrays for metric calculation
        y_true = np.array(y_true)
        y_pred = np.array(y_pred)
    
        # Calculate metrics
        accuracy = accuracy_score(y_true, y_pred)
        train_loss = total_loss / len(self.testloader)
                
        return accuracy, train_loss
    

    