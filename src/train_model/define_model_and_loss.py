import torch.nn as nn 
from src.train_model.model import *
from src.utils.utils import read_data


class ModelAndLossNotSelectedException(Exception):
    def __init__(self, message="Model and loss not selected"):
        self.message = message
        super().__init__(self.message)


def define_model_and_loss(args, device):
    try:
        if args.model_name == "CNN":
            if args.dataset == "MNIST":
                model = cnn_Mnist().to(device)
                loss = nn.CrossEntropyLoss()

            elif args.dataset == "FMNIST":
                model = cnn_Fmnist(10).to(device)
                loss = nn.CrossEntropyLoss()

            elif args.dataset == "CIFAR10":
                if args.fl_algorithm in ["MOON", "MOON_KL", "MOON_L2"]:
                    model = cnn_Cifar10_MOON().to(device)
                    loss = nn.CrossEntropyLoss()
                else:
                    model = cnn_Cifar10().to(device)
                    loss = nn.CrossEntropyLoss()


            elif args.dataset == "EMNIST":
                model = cnn_Emnist().to(device)
                loss = nn.CrossEntropyLoss()

            elif args.dataset == "CELEBA":
                model = cnn_Celeba().to(device)
                loss = nn.NLLLoss()

            elif args.dataset == "CIFAR100":
                model = cnn_Cifar100().to(device)
                loss = nn.CrossEntropyLoss()
            else:
                raise ModelAndLossNotSelectedException()
        
        elif args.model_name == "MCLR":
            if args.dataset == "EMNIST":
                if args.split_method == 'byclass':
                    model = Mclr_Logistic(784,62).to(device)
                elif args.split_method == 'digits':
                    model = Mclr_Logistic(784,10).to(device)
                loss = nn.NLLLoss()
            
            elif args.dataset == "FMNIST":
                model = Mclr_Logistic(784,10).to(device)
                loss = nn.NLLLoss() 

            elif args.dataset == "CIFAR10":
                model = Mclr_Logistic(3072,10).to(device)
                loss = nn.NLLLoss() 
                
            elif args.dataset == "CIFAR100":
                model = Mclr_Logistic(3072,100).to(device)
                loss = nn.NLLLoss() 

            elif args.dataset == "MNIST":
                model = Mclr_Logistic().to(device)
                loss = nn.CrossEntropyLoss()
            else:
                raise ModelAndLossNotSelectedException()

        else:
            raise ModelAndLossNotSelectedException()

        return model, loss
    
    except ModelAndLossNotSelectedException as e:
        print(e)
        return None, None
