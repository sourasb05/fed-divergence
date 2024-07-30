import warnings
warnings.filterwarnings("ignore")
from src.train_model.define_model_and_loss import define_model_and_loss
from src.server.FedAvg_server import Fed_Avg_Server
from src.server.FedProx_server import FedProx_Server
from src.server.FedMOON_server import FedMOON_Server
from src.server.FedMOON_KL_server import FedMOON_KL_Server
from src.server.FedMOON_l2_server import FedMOON_l2_Server
from src.options import args_parser
import torch
import os

torch.manual_seed(0)


def main(args):
    

    device = torch.device("cuda:{}".format(args.gpu) if torch.cuda.is_available() and args.gpu != -1 else "cpu")
    current_directory = os.getcwd()
    print(current_directory)
    
    
    while args.exp_no < args.times:

        model, loss = define_model_and_loss(args, device)
        print(model)
        print(loss)
            
        print(args.dataset)
              
        
        if args.fl_algorithm == "FedAvg":
            server = Fed_Avg_Server(args, model, loss, device, current_directory)

        elif args.fl_algorithm == "FedProx":
            server = FedProx_Server(args, model, loss, device, current_directory)

        elif args.fl_algorithm == "MOON":
            server = FedMOON_Server(args, model, loss, device, current_directory)

        elif args.fl_algorithm == "MOON_KL":
            server = FedMOON_KL_Server(args, model, loss, device, current_directory)

        elif args.fl_algorithm == "MOON_L2":
            server = FedMOON_l2_Server(args, model, loss, device, current_directory)





        server.train()
        server.save_file()
            
        args.exp_no+=1 
    
    
    
if __name__ == "__main__":
    args = args_parser()

    print("=" * 60)
    print("Summary of training process:")
    print("FL Algorithm: {}".format(args.fl_algorithm))
    print("model: {}".format(args.model_name))
    print("optimizer: {}".format(args.optimizer))
    print("Batch size: {}".format(args.batch_size))
    print("Global_iters: {}".format(args.global_iters))
    print("Local_iters: {}".format(args.local_iters))
    print("experiments: {}".format(args.times))
    print("device : {}".format(args.gpu))
    print("Learning rate: {}".format(args.lr))
    if args.fl_algorithm == "FedProx":
        print("Proximal hyperparameter {}".format(args.lambda_prox))
    
    print("=" * 60)

    main(args)




