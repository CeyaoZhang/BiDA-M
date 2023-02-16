from tkinter import NONE
import torch
import torch.nn as nn
import torch.nn.functional as F

# this is a test of synch

import os, re
from time import sleep
from datetime import datetime
from tqdm import tqdm


import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg') # Linux绘图需要
# matplotlib.use('TkAgg')
matplotlib.rcParams['figure.max_open_warning'] = 0
from matplotlib import pyplot as plt
from matplotlib.ticker import MaxNLocator
from sklearn.model_selection import train_test_split

from models import MetaLinear, MetaNN, MetaMLP
from data_generator import generate_data1

from multiprocessing import cpu_count
print('\nThe number of CPU kernels:', cpu_count())

cpu_num = 10 # 这里设置成你想运行的CPU个数
os.environ['OMP_NUM_THREADS'] = str(cpu_num)
os.environ['OPENBLAS_NUM_THREADS'] = str(cpu_num)
os.environ['MKL_NUM_THREADS'] = str(cpu_num)
os.environ['VECLIB_MAXIMUM_THREADS'] = str(cpu_num)
os.environ['NUMEXPR_NUM_THREADS'] = str(cpu_num)
torch.set_num_threads(cpu_num)

use_cuda = True # False
# torch.manual_seed(args.data_seed)
if torch.cuda.is_available():
    print("\ngpu cuda is available!")
    torch.cuda.manual_seed(1)
    torch.manual_seed(1)
    device = torch.device("cuda")
else:
    print("\ncuda is not available! cpu is available!")
    torch.manual_seed(1)
    device = torch.device("cpu")
# device = torch.device("cuda" if use_cuda and torch.cuda.is_available() else "cpu")

# func = args.func
# noise_type = args.noise_type
# seed = args.data_seed
# batch_size = args.batch_size
# epochs = args.epochs

############################


def build_model(struc, func=None):
    # print('--')
    # print(args)
    if '2d' in func:
        num_hidden = 2
    else:
        num_hidden = 1
    # model = VNet(num_dims, num_hidden, 1)
    
    if struc == 'NN':
        model = MetaNN(num_hidden, 1)
    elif struc == 'MLP':
        model = MetaMLP(30)
    elif struc == 'Lin' and ('linear0' in func):
        model = MetaLinear(num_hidden, 1)

    if torch.cuda.is_available() and use_cuda:
        model.cuda()
        # torch.backends.cudnn.benchmark = True

    return model

def lp(y_true, y_pred, hypara):
    error = y_true-y_pred
    return torch.abs(error).pow(hypara).mean()
    
def lh(y_true, y_pred, hypara):
    error = torch.abs(y_true-y_pred)
    # print('\nerror:', x.numpy())
    return torch.where(error <= hypara, 0.5 * error.pow(2), hypara * error - 0.5 * hypara**2).mean()

def lt(y_true, y_pred, hypara):
    error = torch.abs(y_true-y_pred)
    return torch.where(error<=hypara, hypara**2/6*(1-(1-error**2/hypara**2)**3),  hypara**2/6).mean()
    
def lg(y_true, y_pred, hypara):
    error = y_true - y_pred
    return ((torch.pow( (error.pow(2) / (torch.abs(hypara-2))) + 1 ,\
                                    hypara/2) -1) * torch.abs(hypara-2) / hypara).mean()

def loss_choice(loss):
    if loss == 'Lp':
        loss_fn = lp
    elif loss == 'Huber':
        loss_fn = lh
    elif loss == 'Tukey':
        loss_fn = lt
    elif loss == 'General':
        loss_fn = lg
    elif loss == 'MSE' or 'OLS':
        loss_fn = nn.MSELoss()
    elif loss == 'MAE' or 'LAD':
        loss_fn = nn.L1Loss()
    else:
        raise Exception('The loss function is wrong!')
    return loss_fn

def adjust_learning_rate(optimizer, epochs):
    # lr = args.lr * ((0.1 ** int(epochs >= 80)) * (0.1 ** int(epochs >= 100)))  # For WRN-28-10
    lr = args.lr * ((0.1 ** int(epochs >= 500)) * (0.1 ** int(epochs >= 1000)) * (0.1 ** int(epochs >= 2000)))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr

############################
# the next three func are for meta 

def meta_train(args, epoch, train_loader, val_loader, model, inner_loss, hypara, mode):

    # pseudo_lr = 1e-1
    # pseudo_lr = 1e-3 # seed 50
    # pseudo_lr = 1e-3 * ((0.1 ** int(epoch >= 500)) * (0.1 ** int(epoch >= 1000)) * (0.1 ** int(epoch >= 1500)))   
    # optimizer_meta = torch.optim.AdamW([hypara], lr=3e-4, weight_decay=1e-2)
    # optimizer_model = torch.optim.SGD(model.params(), lr=pseudo_lr, momentum=args.momentum, weight_decay=args.weight_decay)

    # # the best performance until now
    pseudo_lr = 1e-2 * ((0.1 ** int(epoch >= 500)) * (0.1 ** int(epoch >= 1000)) * (0.1 ** int(epoch >= 2000))) 
    meta_lr = 1e-3
    optimizer_meta = torch.optim.Adam([hypara], lr=meta_lr)
    optimizer_model = torch.optim.Adam(model.params(), lr=1e-3, weight_decay=1e-4)
    
    # pseudo_lr = 1e-1/(epoch+1)  
    # meta_lr = 1e-2
    # optimizer_meta = torch.optim.Adam([hypara], lr=meta_lr)
    # optimizer_model = torch.optim.Adam(model.params(), lr=pseudo_lr, weight_decay=args.weight_decay)

    # pseudo_lr = 1e-1/(epoch+1)  
    # meta_lr = 1e-2 * ((0.1 ** int(epoch>=1000)))
    # optimizer_meta = torch.optim.Adam([hypara], lr=meta_lr)
    # optimizer_model = torch.optim.Adam(model.params(), lr=pseudo_lr, weight_decay=args.weight_decay)

    # pseudo_lr = 1e-1/(epoch+1)  
    # meta_lr = 1e-3
    # optimizer_meta = torch.optim.Adam([hypara], lr=meta_lr)
    # optimizer_model = torch.optim.Adam(model.params(), lr=pseudo_lr, weight_decay=args.weight_decay)

    # pseudo_lr = 1e-2 * ((0.1 ** int(epoch >= 500)) * (0.1 ** int(epoch >= 1000)) * (0.1 ** int(epoch >= 2000)))
    # meta_lr = 1e-2 * ((0.1 ** int(epoch>=1000)))
    # optimizer_meta = torch.optim.Adam([hypara], lr=meta_lr)
    # optimizer_model = torch.optim.Adam(model.params(), lr=1e-3, weight_decay=1e-4)
    
    # adjust_learning_rate(optimizer_model, epoch)
    # adjust_learning_rate(optimizer_meta, epoch)

    if args.noisy_val:
        if 'Huber' in args.outer_loss:
            tau = float(re.findall(r"\d+\.?\d*", args.outer_loss)[0])
            meta_loss_func = nn.HuberLoss(delta=tau)
        elif args.outer_loss == 'MAE':
            meta_loss_func = nn.L1Loss()
    else:
        meta_loss_func = nn.MSELoss()
    task_loss_func = loss_choice(inner_loss)
    task_mse, meta_mse = 0, 0

    val_loader_iter = iter(val_loader)
    for batch_idx, (inputs, targets) in enumerate(train_loader):

        #############################
        ## first update

        model.train()
        inputs, targets = inputs.to(device), targets.to(device)

        pseudo_model = build_model(args.model, args.func)
        pseudo_model.load_state_dict(model.state_dict())
        outputs = pseudo_model(inputs)
        outputs = torch.squeeze(outputs)
        batch_task_loss_ = task_loss_func(outputs, targets, hypara) ## hypara.requires_grad_=True
        pseudo_model.zero_grad()
        grads = torch.autograd.grad(batch_task_loss_, (pseudo_model.params()), create_graph=True)
        
        pseudo_model.update_params(lr_inner=pseudo_lr, source_params=grads)
        del grads

        #############################
        ## second update

        try:
            inputs_val, targets_val = next(val_loader_iter)
        except StopIteration:
            val_loader_iter = iter(val_loader)
            inputs_val, targets_val = next(val_loader_iter)
        inputs_val, targets_val = inputs_val.to(device), targets_val.to(device)
        
        outputs_val = pseudo_model(inputs_val)
        outputs_val = torch.squeeze(outputs_val)
        batch_meta_loss = meta_loss_func(outputs_val, targets_val)
        optimizer_meta.zero_grad()
        batch_meta_loss.backward()
        optimizer_meta.step()
        
        if hypara.item() <= 0:
            hypara.data = torch.tensor([1e-5], requires_grad=True, device=device)

        #############################
        ## third update
        if mode == 'meta3':
            outputs = model(inputs)
            outputs = torch.squeeze(outputs)
            batch_task_loss = task_loss_func(outputs, targets, hypara) # hypara.requires_grad=False
            optimizer_model.zero_grad()
            batch_task_loss.backward()
            optimizer_model.step()
        elif mode == 'meta2':
            model.load_state_dict(pseudo_model.state_dict())
        else:
            raise Exception('The meta update mode is wrong!')
        
        batch_task_mse = F.mse_loss(outputs, targets)
        
        #############################
        task_mse += batch_task_mse.item() * len(inputs)
        meta_mse += batch_meta_loss.item()

    task_mse /= len(train_loader.dataset)
    meta_mse /= len(train_loader)
    
    return task_mse, meta_mse 

def meta_test(model, data_loader):

    model.eval()
    test_loss = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(data_loader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            outputs = torch.squeeze(outputs)
            test_loss += F.mse_loss(outputs, targets).item() * len(inputs)

    test_loss /= len(data_loader.dataset)

    return test_loss

def meta_exec(args, train_loader, val_loader, test_loader, \
    X_tr, y_tr, X_val, y_val, X_te, y_te, \
        output_folder2, name1, name2, mode):
    
    epochs = args.epochs
    # name1 = args.mode + '_' + args.loss + '(%s)'%args.hypara + '_{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    # name1 = 'meta' + '_{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S"))
    inner_loss = args.adap_loss
    if inner_loss == 'MAE' or inner_loss == 'MSE':
        raise Exception('The loss function must contain hyperparameter')
    else:
        loss_hypara = inner_loss + '(%s)'%args.hypara
        name_train_result = mode + '_' + loss_hypara # + '_' + name1+'_'+name2
    
    output_folder2 = os.path.join(output_folder2, name_train_result+'_{}'.format(datetime.now().strftime("%Y%m%d-%H%M%S")))
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)

    model = build_model(args.model, args.func)
    best_val = meta_test(model, val_loader)
    best_epoch = 0
    best_val_test = meta_test(model, test_loader)
    print('\nInitial %s result %.5f and test MSE %.3f'%(loss_hypara, best_val, best_val_test))
    
    hypara = torch.tensor([args.hypara], requires_grad=True, device=device)

    epoch_hy_list = [hypara.item()]

    epoch_task_loss_list = []
    epoch_meta_loss_list = []
    epoch_val_loss_list = [best_val]
    epoch_test_loss_list = [best_val_test]

    
    
    # for epoch in tqdm(range(args.epochs)):
    for epoch in range(epochs):
        task_mse, meta_mse = meta_train(args, epoch, \
            train_loader, val_loader, model, inner_loss, hypara, mode)
        val_loss = meta_test(model, val_loader)
        test_loss = meta_test(model, test_loader)
        
        epoch_hy_list.append(hypara.item())
        epoch_task_loss_list.append(task_mse)
        epoch_meta_loss_list.append(meta_mse)
        epoch_val_loss_list.append(val_loss)
        epoch_test_loss_list.append(test_loss)

        if val_loss <= best_val:
            best_val = val_loss
            best_epoch = epoch
            best_val_test = test_loss
            # print('%d epochs with best result %s'%(epoch+1, best_result))
            
        if epoch%int(epochs/10)==0:   
            
            name_performance = '%.3f(%d)'%(val_loss, epoch)
            
            model.eval()
            with torch.no_grad():
                y_te_hat = model(X_te.to(device))
            
            # plt.figure()
            # plt.title('%s with %.3f'%(name_performance, best_val))
            if '2d' in args.func:
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                ax.plot(X_tr[:,0].numpy(), X_tr[:,1].numpy(), y_tr.numpy(), '.', ms=10, alpha=0.6, label='train')
                ax.plot(X_val[:,0].numpy(), X_val[:,1].numpy(), y_val.numpy(), '*', ms=10, alpha=0.6, label='val')
                ax.plot(X_te[:,0].numpy(), X_te[:,1].numpy(), y_te.numpy(), '.', ms=10, alpha=0.6, label='test')
                ax.plot(X_te[:,0].tolist(), X_te[:,1].tolist(), y_te_hat[:,0].detach().cpu().tolist(), '.', ms=10, alpha=0.6, label='fitted')
            else:
                fig, ax = plt.subplots()
                ax.plot(X_tr.numpy(), y_tr.numpy(), '.', label='train')
                ax.plot(X_val.numpy(), y_val.numpy(), '*', label='val')
                ax.plot(X_te.numpy(), y_te.numpy(), '.', label='test')
                ax.plot(X_te.tolist(), y_te_hat[:,0].detach().cpu().tolist(), '.', label='fitted')
            ax.set_title('%s with %.3f'%(name_performance, best_val))
            ax.legend()
            fig.savefig(os.path.join(output_folder2, name_train_result+'_' +name_performance+'.png')) # +'_'+ name1+'_'+name2
            del fig
            plt.clf()
            plt.close('all') 


    ##################################################
    ### plot
    
    converge_hy = np.mean(epoch_hy_list[-100:])
    # fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(17,13))
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(25,10)) # 
    fig.suptitle('Train %s %s with %s(%s) converge to %.3f'%\
        (args.noise_type, args.robust, inner_loss, args.hypara, converge_hy)) # , fontsize=16

    ax1.set_title('Training Loss')
    ax1.plot(epoch_task_loss_list, '-', label='train MSE')
    # ax1.legend()
    ax1.set_xlabel('Epochs')
    # ax4.set_ylim(-1,top=9)
    # ax2.set_ylim(bottom=0)
    # ax2.set_xlim(left=0)
    # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax2.set_title('Validataion Loss')
    ax2.plot(epoch_val_loss_list, '-', label='val MSE')
    ax2.set_xlabel('Epochs')
    ax3.set_title('Testing Loss')
    ax3.plot(epoch_test_loss_list, '-', label='test MSE')
    ax3.set_xlabel('Epochs')

    fig.savefig(os.path.join(output_folder2, 'loss_'+name_train_result+'.png')) # +'_' + name1+'_'+name2
    del fig

    plt.figure()
    # plt.title('Train %s %s with %s(%s) converge to %.3f'%\
        # (args.noise_type, args.robust, loss, args.hypara, converge_hy))
    plt.plot(epoch_hy_list, '-')
    plt.xlabel('Epochs', fontsize = 14)
    plt.ylabel('The value of %s threshold'%inner_loss, fontsize = 14)
    plt.savefig(os.path.join(output_folder2, 'hypara_'+name_train_result+'_convege(%.3f)'%converge_hy+'.png')) # +'_' + name1+'_'+name2
    plt.close('all') 
    
    # print('\nThis is epoch_hy_list:', epoch_hy_list[-20:])
    # print('\n=======================\n')
    best_performance = '%.3f(%d)'%(best_val, best_epoch)
    best_test =  min(epoch_test_loss_list)
    best_test_epoch = epoch_test_loss_list.index(best_test)

    print("%s: %s Done with best val %s and its test %.3f | the best test %.3f(%d) | hy converge to %.3f!"%\
    (mode, loss_hypara, best_performance, best_val_test, best_test, best_test_epoch, converge_hy))
    # print('\n=======================\n')
    
    return best_val_test, best_performance, epoch_test_loss_list, converge_hy


############################
# the next three funcs are for non-meta, NN+GD+Huber loss 处理 robust 问题的效果

def train_loop(dataloader, model, loss, hypara, loss_fn, optimizer):

    if type(hypara) == float:
        hypara = np.array(hypara) # np.float64
        hypara = torch.from_numpy(hypara).float() # tensor.float32
        hypara = hypara.to(device)
        # hypara = np.array(hypara, dtype=np.float32)


    model.train()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    train_mse = 0
    
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        # Compute prediction and loss
        pred = model(X)
        pred = torch.squeeze(pred)
        if hypara==None:
            train_loss = loss_fn(pred, y)
        else:
            train_loss = loss_fn(pred, y, hypara) 
        train_mse += F.mse_loss(pred, y).item()

        # Backpropagation
        optimizer.zero_grad()
        train_loss.backward()
        optimizer.step()
        
    train_mse /= num_batches
    
    return train_mse


def evaluate_loop(dataloader, model, loss_fn):
    
    model.eval()
    size = len(dataloader.dataset)
    num_batches = len(dataloader)
    
    test_loss = 0

    with torch.no_grad():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            pred = model(X)
            pred = torch.squeeze(pred)
            test_loss += loss_fn(pred, y).item() * len(X) 
            
    test_loss /= size
    # print(f"Test Error: Avg loss: {test_loss:>8f} \n")

    return test_loss


def normal_exec(args, loss_hypara, train_loader, val_loader, test_loader,\
    X_tr, y_tr, X_val, y_val, X_te, y_te, \
        output_folder2, name1, name2, mode):
    
    epochs = args.epochs
    if '(' in loss_hypara:
        loss_fn, hypara, _ = re.split('\(|\)', loss_hypara)
        hypara = float(hypara)
        # hypara = float(re.findall(r"\d+\.?\d*", loss_hypara)[0])
        name_train_result = '%s_%s(%s)_%depochs'%(mode, loss_fn, hypara, epochs)#+ '_' + name1+'_'+name2
    else:
        loss_fn = loss_hypara
        hypara = None
        name_train_result =  '%s_%s_%depochs'%(mode, loss_fn, epochs)#+ '_' + name1+'_'+name2
    # print('\nStart normal mode with %s loss'%loss)
        
    
    timenow = datetime.now().strftime("%m%d-%H%M")
    output_folder3 = os.path.join(output_folder2, name_train_result+'_{}'.format(timenow))
    # print('\n', output_folder2)
    
    if not os.path.exists(output_folder3):
        os.makedirs(output_folder3)

    loss_fn2 = nn.MSELoss()
    loss_fn1 = loss_choice(loss_fn)
    
    model = build_model(args.bmodel, func=args.func)
    best_val = evaluate_loop(val_loader, model, loss_fn2)
    best_epoch = 0
    best_val_test = evaluate_loop(test_loader, model, loss_fn2)
    print('\nInitial %s result %.5f and test MSE %.3f'%(loss_hypara, best_val, best_val_test))
    
    train_loss_list = []
    val_loss_list = [best_val]
    test_loss_list = [best_val_test]
    
    # for epoch in tqdm(range(args.epochs)):
    for epoch in range(epochs):
        
        # adjust_learning_rate(optimizer, epoch)
        # normal_lr = 1e-2 * ((0.1 ** int(epoch >= 500)) * (0.1 ** int(epoch >= 1000)) * (0.1 ** int(epoch >= 2000)))
        # normal_lr = 1e-2/(epoch+1)
        if loss_fn == 'MSE' or 'OLS':
            normal_lr = 1e-1/(epoch+1) ## 之前用了这个
        elif loss_fn == 'MAE' or 'LAD':
            normal_lr = 1e-2
        # normal_lr = 1e-4
        optimizer = torch.optim.Adam(model.params(), lr=normal_lr, weight_decay=1e-4)
        # optimizer = torch.optim.SGD(model.params(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
        train_loss = train_loop(train_loader, model, loss_fn, hypara, loss_fn1, optimizer)
        val_loss = evaluate_loop(val_loader, model, loss_fn2)
        test_loss = evaluate_loop(test_loader, model, loss_fn2)
        
        
        train_loss_list.append(train_loss)
        val_loss_list.append(val_loss)
        test_loss_list.append(test_loss)
        
        if val_loss <= best_val:
            best_val = val_loss
            best_epoch = epoch
            best_val_test = test_loss
            # print('%d epochs with best result %s'%(epoch+1, best_result))
            # name_performance = '%depoch(%.3f)'%(best_epoch, best_result)
            

        if epoch%int(epochs/10)==0:

            name_performance = '%.3f(%d)'%(val_loss, epoch)
            
            model.eval()
            with torch.no_grad():
                y_te_hat = model(X_te.to(device))
                # y_tr_hat = model(X_tr.to(device))
            
            # plt.figure()
            
            if '2d' in args.func:
                # ax = fig.add_subplot(projection='3d')
                fig, ax = plt.subplots(subplot_kw={"projection": "3d"})
                # ax = plt.axes(projection='3d')
                ax.plot(X_tr[:,0].numpy(), X_tr[:,1].numpy(), y_tr.numpy(), '.', ms=10, alpha=0.6, label='train')
                ax.plot(X_val[:,0].numpy(), X_val[:,1].numpy(), y_val.numpy(), '*', ms=10, alpha=0.6, label='val')
                ax.plot(X_te[:,0].numpy(), X_te[:,1].numpy(), y_te.numpy(), '.', ms=10, alpha=0.6, label='test')
                ax.plot(X_te[:,0].tolist(), X_te[:,1].tolist(), y_te_hat[:,0].detach().cpu().tolist(), '.', ms=10, alpha=0.6, label='fitted')
            else:
                fig, ax = plt.subplots()
                ax.plot(X_tr.numpy(), y_tr.numpy(), '.', label='train')
                ax.plot(X_val.numpy(), y_val.numpy(), '*', label='val')
                ax.plot(X_te.numpy(), y_te.numpy(), '.', label='test')
                ax.plot(X_te.tolist(), y_te_hat[:,0].detach().cpu().tolist(), '.', label='fitted')
            ax.set_title('%s with best %.3f'%(name_performance, best_val))
            ax.legend()
            fig.savefig(os.path.join(output_folder3, name_performance+'.png')) #name_train_result+'_'+
            del fig
            plt.clf()
            plt.close('all') 

    ##################################################
    ### plot
    
    fig, ax1 = plt.subplots(1, 1) #(ax1, ax2), figsize=(25,10)
    if hypara == None:
        fig.suptitle('Train %s %s with %s'%(args.noise_type, args.robust, loss_fn), fontsize=16)
    else:
        fig.suptitle('Train %s %s with %s(%s)'%(args.noise_type, args.robust, loss_fn, hypara), fontsize=16)

    # ax1.set_title('Training Loss')
    ax1.plot(train_loss_list, '-', label='train')
    # ax1.set_xlabel('Epochs')
    # ax1.legend()
    # ax4.set_ylim(-1,top=9)
    # ax2.set_ylim(bottom=0)
    # ax2.set_xlim(left=0)
    # ax1.xaxis.set_major_locator(MaxNLocator(integer=True))
    # ax2.set_title('Validataion Loss')
    ax1.plot(val_loss_list, '-', label='val')
    ax1.plot(test_loss_list, '-', label='test')
    ax1.set_xlabel('Epochs')
    ax1.set_ylabel('MSE')
    ax1.legend()
    
    fig.savefig(os.path.join(output_folder3, 'loss_'+name_train_result+'.png')) # +'_' + name1+'_'+name2
    del fig
    
    best_performance = '%.3f(%d)'%(best_val, best_epoch)
    best_test =  min(test_loss_list)
    best_test_epoch = test_loss_list.index(best_test)

    try:
        print("%s: %s Done with best val %s and its test %.3f | the best test %.3f(%d)!"%\
            (mode, loss_hypara, best_performance, best_val_test, best_test, best_test_epoch))
    except UnboundLocalError:
        print('UnboundLocalError')
    print('\n=======================\n')
    
    

    return best_val_test, best_performance, test_loss_list
    




if __name__ == '__main__':

    start_time = datetime.now()

    ############################
    from arguments import get_args
    args = get_args()
    
    print('\n=======================\n')
    print(device, args)
    print('\n=======================')

    ##############################################
    sigma=0.5 ### 1.2
    epsilon=0.25 ###
    noise_seed = 47

    # methods = ['OLS', 'LAD', 'Huber(1.345)', 'Tukey(4.685)']
    taus = [100, 50, 10, 5, 1, 0.5, 0.1, 0.01, 0.001, 0.00001]
    methods = ['OLS', 'LAD']+['Huber(%.6f)'%tau for tau in taus] + ['Tukey(%.6f)'%tau for tau in taus]
    # methods = ['OLS', 'LAD']
    best_val_testes = {}

    (X_tr, y_tr), (X_val,y_val), (X_te, y_te), (name, name1, name2),\
         (output_folder, output_folder1, output_folder2), (output_folder_, output_folder1_, output_folder2_),\
              (train_loader, val_loader, test_loader) = generate_data1(args, noise_seed, sigma, epsilon)
    # print('\n%d train, %d val, %d test'%(num_train, num_val, num_test))
    # print('\nX & y shape :', X_tr.shape, X_te.shape)
    
    timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
    print('timenow: %s'%timenow)
    f = open(os.path.join(output_folder2,'result-%s.txt'%timenow), 'w')
    f.write('args:%s'%name+ '\n')
    f.write('noise_seed:%s'%name1+ '\n')
    f.write('sigma&epsilon:%s'%name2+ '\n')

    if args.mode == 'none':
        pass
    else:
        if args.mode == 'normal':
            for loss_hypara in methods:
                best_val_test, best_performance, _ = normal_exec(args, loss_hypara, \
                    train_loader, val_loader, test_loader, \
                        X_tr, y_tr, X_val, y_val, X_te, y_te, \
                            output_folder2, name1, name2, args.mode)   
        else:
            best_val_test, best_performance, _, _ = meta_exec(args, \
                train_loader, val_loader, test_loader,\
                 X_tr, y_tr, X_val, y_val, X_te, y_te, \
                     output_folder2, name1, name2, args.mode)
        
        best_val_testes[loss_hypara] = best_val_test
        f.write('%s: val %s and test %.3f'%(loss_hypara, best_performance, best_val_test)+ '\n')
    
    f.close
    end_time = datetime.now()
    print("\nCost time : {:.4f}s".format((end_time-start_time).seconds))
    print(best_val_testes)

    
    # pass
    # python train.py --robust contamination --func linear0 --noise laplace --data_seed 10 --epochs 40 --loss MSE --hypara 2. --mode none
     
    # python train.py --func linear0 --data_seed 11 --mode normal
    # python train.py --robust contamination --func sin0 --noise laplace --data_seed 10 --epochs 100 --loss MAE --hypara 1. --mode normal
    
    # python train.py --func linear0 --data_seed 11 --mode meta
    # python train.py --func linear01 --data_seed 20 --noise laplace --mode meta
    # python train.py --robust contamination --func poly5 --noise laplace --data_seed 10 --epochs 1500 --loss Huber --hypara 1.23 --mode meta

    # python train.py --func poly5 --data_seed 11 --mode meta
    
    # python train.py --func sin01 --data_seed 40 --mode normal
    # python train.py --func tan01 --data_seed 666 --mode normal --epochs 2000 --noise_type Lognormal

    # python train.py --func log01 --data_seed 6 --mode meta3 --epochs 2000
    # python train.py --func hd01 --data_seed 6 --mode none --noise_type Lognormal

    
    