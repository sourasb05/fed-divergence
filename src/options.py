import argparse


def args_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--dataset", type=str, default="CIFAR10", choices=["MNIST", "FMNIST", "CIFAR10", "EMNIST", "CIFAR100", "CELEBA", "SYNTHETIC", "MOVIELENS_1m", "MOVIELENS_100k"])
    parser.add_argument("--model_name", type=str, default="CNN",  choices=["CNN", "MCLR", "DNN"])
    parser.add_argument("--exp_no", type=int, default=0)
    parser.add_argument("--times", type=int, default=1)
    parser.add_argument("--fl_algorithm", type=str, default= "MOON_KL", choices=["MOON_KL", "MOON_L2", "FedAvg","FedProx","MOON"])
    parser.add_argument("--optimizer", type=str, default="SGD", choices=["SGD", "Adam"])
    parser.add_argument("--split_method", type=str, default="digits", choices=["digits", "byclass"]) # for EMNIST dataset
    parser.add_argument("--total_labels", type=int, default=10, choices=[62, 10])   # for EMNIST dataset
    parser.add_argument("--lr", type=float, default=0.01, help="Learning rate")
    parser.add_argument("--lambda_prox", type=float, default=1.0, help="proximal_regularizer")
    parser.add_argument("--global_iters", type=int, default=10)
    parser.add_argument("--local_iters", type=int, default=10)
    parser.add_argument("--batch_size", type=int, default=124)
    parser.add_argument("--gpu", type=int, default=0, choices=[0,1,2,3,4,5,6,7,8] )
    parser.add_argument("--num_users", type=int, default=10, help="should be multiple of 10") 
    parser.add_argument("--num_users_perGR", type=int, default=10, help="should be <= num_users")
    parser.add_argument("--num_labels", type=int, default=10)  
    parser.add_argument("--iid", type=int, default=1, choices=[0, 1], help="0 : for iid , 1 : non-iid")
    parser.add_argument("--wandb", action='store_true')
    """For MOON"""
    parser.add_argument('--temperature', type=float, default=0.5, help='the temperature parameter for contrastive loss')
    parser.add_argument('--mu', type=float, default=1, help='the mu parameter for fedprox or moon')
    """For MOON_KL"""
    parser.add_argument('--lamda', type=float, default=0.5, help='the adjustment parameter between CE and KL in MOON_KL')
    """For MOON_similarity_stability"""
    parser.add_argument('--lambda_g', type=float, default=0.5, help='the adjustment parameter between local and global model')
    parser.add_argument('--lambda_s', type=float, default=0.5, help='the adjustment parameter between previous and current local model')



            

    
    args = parser.parse_args()

    return args
