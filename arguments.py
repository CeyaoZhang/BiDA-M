import argparse

def get_args():

    ############################
    

    parser = argparse.ArgumentParser()

    parser.add_argument('--mode', type=str, default='normal', choices=['none', 'normal', 'meta3', 'meta2'], 
    help='none mode just generate data, noraml mode train the NN with noraml, meta2 use two updates and meta3 use three updates')

    parser.add_argument('--train_mode', type=str, default='regular', choices=['regular', 'sigmas', 'epsilons', 'sigmas-seeds', 'epsilons-seeds'])
    parser.add_argument('--num_seeds', type=int, default='50', help="this is the setting for train_mode='sigmas-seeds' or 'epsilons-seeds'")

    parser.add_argument('--func', type=str, required=True, default='func0')
    parser.add_argument('--robust', type=str, default='ct', choices=['ct', 'ht'], help='robust type, ht=heavy-tails, ct=contamination')
    parser.add_argument('--noise_type', type=str, default='Laplace', choices=['Lognormal','Normal','Laplace','Pareto'], help='noise type')
    
    parser.add_argument('--adap_loss', type=str, default='Huber', choices=['Lp','Huber','Tukey','General'], help='loss function choose to be optimize in the inner/lower loop')
    parser.add_argument('--hypara', type=float, default='1.345', help='initial hyperparameters, 1.345 for c.95 and 0.7317 for c.85 with Huber loss')

    parser.add_argument('--train_add_val', type=int, default=0, choices=[0,1], help='1 means that add the validation data to the training data, which is a better baseline to the BiDA-M methods')
    parser.add_argument('--noisy_val', type=int, default=0, choices=[0, 1], help='0 means clean val data, 1 means that val data also contain heavy noise')
    parser.add_argument('--outer_loss', type=str, default='MSE', help='the loss in the outer/upper loop, useful only when noisy_val=1')


    parser.add_argument('--bflag', type=int, default=1, choices=[0,1], help='bflag=0 use sm package and linear model, bflag=1 use bmodel which are all NN')
    # parser.add_argument('--bmodel', type=str, default='NN', choices=['NN','Lin','MLP'], help='model choice baseline methods')
    # parser.add_argument('--model', type=str, default='NN', choices=['NN','Lin','MLP'], help='the estimator model')
    parser.add_argument('--bmodel', type=str, default='50-80-10', help='model choice baseline methods')
    parser.add_argument('--model', type=str, default='50-80-10', help='the estimator model')
    
    parser.add_argument('--epochs', type=int,  default=500, help='number of total epochs to run')
    parser.add_argument('--batch_size', type=int, default=60, help='mini-batch size')

    parser.add_argument('--lr', '--learning-rate', type=float, default=1e-3,  help='initial learning rate')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--nesterov', type=bool, default=True, help='nesterov momentum')
    parser.add_argument('--weight_decay', '--wd', type=float, default=5e-4, help='weight decay (default: 5e-4)')

    parser.add_argument('--data_seed', type=int, default=0) # this parser will be used in the train.py


    ############################

    args = parser.parse_args()
    # print('\n=======================\n')
    # print(args)
    # print('\n=======================\n')

    return args