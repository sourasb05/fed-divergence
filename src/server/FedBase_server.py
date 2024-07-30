import torch.nn.init as init
import torch
import copy
import h5py
import os
from datetime import date
import numpy as np
import datetime
import wandb
import statistics


class Base_server:
    def __init__(self, args, model, loss, device, curr_dir):
        self.global_model = copy.deepcopy(model)
        self.loss = loss
        self.device = device
        self.exp_no = args.exp_no
        self.current_directory = curr_dir
        self.global_iters = args.global_iters
        self.dataset_name = args.dataset
        self.fl_algorithm = args.fl_algorithm
        self.model_name = args.model_name
        self.lr = args.lr
        self.lambda_prox = args.lambda_prox
        self.batch_size = args.batch_size
        self.num_users_perGR = args.num_users_perGR
        self.optimizer_name = args.optimizer
        self.num_labels=args.num_labels
       
        self.global_train_acc = []
        self.global_test_acc = []            
        self.global_train_loss = []
        self.global_test_loss  = []
        self.global_precision  = []
        self.global_recall  = []
        self.global_f1score  = []

        self.local_train_acc = []
        self.local_test_acc = []            
        self.local_train_loss = []
        self.local_test_loss  = []
        self.local_precision  = []
        self.local_recall  = []
        self.local_f1score  = []

        #self.global_model_parameters

        self.users = []
        
        date_and_time = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        
        self.wandb = wandb.init(
            project="DIPA2-loss-function",
            name="{}_{}".format(args.fl_algorithm, date_and_time),
            mode=None if args.wandb else "disabled")
        
        """for param in self.global_model.parameters():
            print(f"from fedbase : {param}")
      """

    def send_parameters(self):   
        if len(self.selected_users) == 0:
            assert(f"AssertionError : The client can't be zero")
        else:
            for user in self.selected_users:
                #print(user.id)
                """
                for param in self.global_model.parameters():
                    print(f"from fedbase send_parameters : {param}")
                """
                user.set_parameters(self.global_model.parameters())

    def initialize_parameters_to_zero(self):
        for param in self.global_model.parameters():
            if param.requires_grad:
                init.zeros_(param)
    
    def save_model(self, glob_iter):
        if glob_iter == self.num_glob_iters-1:
            model_path = self.current_directory + "/models/" + self.algorithm + "/global_model/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.global_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "server_checkpoint_GR" + str(glob_iter) + ".pt"))
            
        if self.global_test_loss[glob_iter] < self.minimum_test_loss:
            self.minimum_test_loss = self.global_test_loss[glob_iter]
            model_path = self.current_directory + "/models/" + self.algorithm + "/global_model/"
            if not os.path.exists(model_path):
                os.makedirs(model_path)
            checkpoint = {'GR': glob_iter,
                        'model_state_dict': self.global_model.state_dict(),
                        'loss': self.minimum_test_loss
                        }
            torch.save(checkpoint, os.path.join(model_path, "best_server_checkpoint" + ".pt"))
            
    def select_users(self, round, k):
        np.random.seed(round)
        return np.random.choice(self.users, k, replace=False) 
            
    
    def eval_test(self, which_model, t):
        accs = []
        losses = []
        precisions = []
        recalls = []
        f1s = []
        cms = []
        if which_model == 'global':
            for c in self.selected_users:
                accuracy, loss, precision, recall, f1 = c.test_el(self.global_model.parameters())
               
                accs.append(accuracy)
                losses.append(loss)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                
            
        elif which_model == 'local':
            for c in self.selected_users:
                accuracy, loss, precision, recall, f1 = c.test_el(None)
                accs.append(accuracy)
                losses.append(loss)
                precisions.append(precision)
                recalls.append(recall)
                f1s.append(f1)
                

            
        return accs, losses, precisions, recalls, f1s

    def eval_train(self, which_model, t):
        accs = []
        losses = []
        
        if which_model == 'global':
            for c in self.selected_users:
                accuracy, loss = c.train_el(self.global_model.parameters())
                accs.append(accuracy)
                losses.append(loss)
        elif which_model == 'local':
            for c in self.selected_users:
                accuracy, loss = c.train_el(None)
                accs.append(accuracy)
                losses.append(loss)
        

        return accs, losses


    def evaluate(self, t, which_model):

        if which_model == "global":
            test_accs, test_losses, precisions, recalls, f1s = self.eval_test(which_model, t)
            train_accs, train_losses  = self.eval_train(which_model, t)

            self.global_train_acc.append(statistics.mean(train_accs))
            self.global_test_acc.append(statistics.mean(test_accs))
            self.global_train_loss.append(statistics.mean(train_losses))
            self.global_test_loss.append(statistics.mean(test_losses))
            self.global_precision.append(statistics.mean(precisions))
            self.global_recall.append(statistics.mean(recalls))
            self.global_f1score.append(statistics.mean(f1s))

            print(f"Global Trainning Accurancy: {self.global_train_acc[t]}" )
            print(f"Global Trainning Loss: {self.global_train_loss[t]}")
            print(f"Global test accurancy: {self.global_test_acc[t]}")
            print(f"Global test_loss: {self.global_test_loss[t]}")
            print(f"Global Precision: {self.global_precision[t]}")
            print(f"Global Recall: {self.global_recall[t]}")
            print(f"Global f1score: {self.global_f1score[t]}")

            self.wandb.log(data={ "global_train_accs" : statistics.mean(train_accs)})
            self.wandb.log(data={ "global_test_accs" : statistics.mean(test_accs)})
            self.wandb.log(data={ "global_train_loss" : statistics.mean(train_losses)})
            self.wandb.log(data={ "global_test_loss" : statistics.mean(test_losses)})
            self.wandb.log(data={ "global_precision" : statistics.mean(precisions)})
            self.wandb.log(data={ "global_recall" : statistics.mean(recalls)})
            self.wandb.log(data={ "global_F1" : statistics.mean(f1s)})



        elif which_model == "local":

            test_accs, test_losses, precisions, recalls, f1s = self.eval_test(which_model, t)
            train_accs, train_losses  = self.eval_train(which_model, t)

            self.local_train_acc.append(statistics.mean(train_accs))
            self.local_test_acc.append(statistics.mean(test_accs))
            self.local_train_loss.append(statistics.mean(train_losses))
            self.local_test_loss.append(statistics.mean(test_losses))
            self.local_precision.append(statistics.mean(precisions))
            self.local_recall.append(statistics.mean(recalls))
            self.local_f1score.append(statistics.mean(f1s))

            print(f"Local Trainning Accurancy: {self.local_train_acc[t]}" )
            print(f"Local Trainning Loss: {self.local_train_loss[t]}")
            print(f"Local test accurancy: {self.local_test_acc[t]}")
            print(f"Local test_loss: {self.local_test_loss[t]}")
            print(f"Local Precision: {self.local_precision[t]}")
            print(f"Local Recall: {self.local_recall[t]}")
            print(f"Local f1score: {self.local_f1score[t]}")

            self.wandb.log(data={ "local_train_accs" : statistics.mean(train_accs)})
            self.wandb.log(data={ "local_test_accs" : statistics.mean(test_accs)})
            self.wandb.log(data={ "local_train_loss" : statistics.mean(train_losses)})
            self.wandb.log(data={ "local_test_loss" : statistics.mean(test_losses)})
            self.wandb.log(data={ "local_precision" : statistics.mean(precisions)})
            self.wandb.log(data={ "local_recall" : statistics.mean(recalls)})
            self.wandb.log(data={ "local_F1" : statistics.mean(f1s)})

    
    def save_file(self):
        today = date.today()
        d1 = today.strftime("%d_%m_%Y")
       
        print("exp_no ", self.exp_no)
        alg = str(self.exp_no) + "_dataset_" + str(self.dataset_name) + "algorithm_" + str(self.fl_algorithm) + "_model_" + str(self.model_name) + "_" + d1
        
   
        print(alg)
       
        directory_name = self.fl_algorithm + "/" + self.dataset_name + "/" + str(self.model_name) + "/num_clients/" +  str(self.num_labels)
        
        if not os.path.exists("./results/"+directory_name):
        # If the directory does not exist, create it
            os.makedirs('./results/' + directory_name)



        with h5py.File("./results/"+ directory_name + "/" + '{}.h5'.format(alg), 'w') as hf:
            hf.create_dataset('exp_no', data=self.exp_no)
            hf.create_dataset('lr', data=self.lr)
            hf.create_dataset('batch_size', data=self.batch_size) 
            
            hf.create_dataset('global_rounds', data=self.global_iters)
            hf.create_dataset('global_train_accuracy', data=self.global_train_acc)
            hf.create_dataset('global_train_loss', data=self.global_train_loss)
            hf.create_dataset('global_test_accuracy', data=self.global_test_acc)
            hf.create_dataset('global_test_loss', data=self.global_test_loss)

            hf.create_dataset('global_precision', data=self.global_precision)
            hf.create_dataset('global_recall', data=self.global_recall)
            hf.create_dataset('global_f1', data=self.global_f1score)

            hf.create_dataset('local_train_accuracy', data=self.local_train_acc)
            hf.create_dataset('local_train_loss', data=self.local_train_loss)
            hf.create_dataset('local_test_accuracy', data=self.local_test_acc)
            hf.create_dataset('local_test_loss', data=self.local_test_loss)

            hf.create_dataset('local_precision', data=self.local_precision)
            hf.create_dataset('local_recall', data=self.local_recall)
            hf.create_dataset('local_f1', data=self.local_f1score)


            hf.close()


    def __del__(self):
        self.wandb.finish()


        