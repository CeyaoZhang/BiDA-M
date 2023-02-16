import os
import copy

import numpy as np
import pandas as pd

import matplotlib
# matplotlib.use('TkAgg') 
from matplotlib import pyplot as plt

import torch
from torch.utils.data import TensorDataset, DataLoader, random_split


def generate_data1(args, noise_seed, sigma, epsilon):

    if 'real' not in args.func:
        num_train, num_val, num_test = 600, 120, 400
        num_samples = num_train + num_val + num_test
        # test_size = num_val + num_test

        np.random.seed(args.data_seed) # seed for generate grouth truth
        if args.func == 'linear0':

            bias = np.array([-3.])
            weights = np.array([[-2.]])
            num_dims = weights.shape[0]
            # X = np.random.randn(num_samples, num_dims)
            X = np.random.uniform(-2, 2, size=(num_samples, num_dims))
            y = np.dot(X, weights) + bias # y(N, 1)
            y = np.squeeze(y) # y(N,)
            # print('\n%s is y=%d+%dX\n'%(args.func, bias, weights))
            
        elif args.func == 'linear01':

            bias = np.array([3.])
            weights = np.array([[-2.]])
            num_dims = weights.shape[0]
            # X = np.random.randn(num_samples, num_dims)
            X = np.random.uniform(-2, 2, size=(num_samples, num_dims))
            y = np.dot(X, weights) + bias # y(N, 1)
            y = np.squeeze(y) # y(N,)
            # print('\n%s is y=%d+%dX\n'%(args.func, bias, weights))

        elif args.func == 'poly5':
            num_dims = 1
            X = np.random.uniform(-1., 1., size=(num_samples, num_dims))
            y = -4*X**5 + 3*X**4 - 6*X**3 + 2*X**2  - 2*X + 1.
            y = np.squeeze(y) # y(N,)
            
        elif args.func == 'sin0':
            num_dims = 1
            X = np.random.uniform(-2, 2, size=(num_samples, num_dims))
            y = np.sin(2*np.pi*X)
            y = np.squeeze(y) # y(N,)
        
        elif args.func == 'sin01':
            num_dims = 1
            X = np.random.uniform(0, 6, size=(num_samples, num_dims))
            # X = np.linspace(0, 6, num_samples)[:,np.newaxis]
            y = np.sin(X).ravel() + np.sin(6 * X).ravel() 

        elif args.func == 'tan01':
            num_dims = 1
            X = np.random.uniform(-1.5, 1.5, size=(num_samples, num_dims))
            y = np.tan(X).ravel() # y(N,)

        elif args.func == 'log01':
            num_dims = 1
            X = np.random.uniform(1, 140, size=(num_samples, num_dims))
            y = np.log(X).ravel() # y(N,)

        elif '2d' in args.func:
            num_dims = 2
            X = np.random.uniform(-1, 1, size=(num_samples, num_dims))

            if args.func == '2d01':
                # y = np.sin(X[:,0]) + np.cos(X[:,1]) # y(N,)
                y = np.cos(X[:,0] + X[:,1]) # y(N,)
            elif args.func == '2d02':
                y = np.sinh(X[:,0] + X[:,1])

            xx = np.arange(-1, 1, 0.1)
            yy = np.arange(-1, 1, 0.1)
            x0, x1 = np.meshgrid(xx, yy)
            Z = np.cos(x0+x1)
        
        else:
            raise SystemExit('\nThe func is wrong')
        
        y += np.random.normal(0, 0.1, num_samples) # standard deviation

        X_tr, X_te, X_val = X[:num_train], X[num_train:num_train+num_test], X[num_train+num_test:]
        y_tr, y_te, y_val = y[:num_train], y[num_train:num_train+num_test], y[num_train+num_test:]

    else:
        if args.func == 'real-rail':
            df = pd.read_csv('real-data/rail-miles.csv', header=0) #, 

        # print(df.head())
        # print(df.tail())
        X = np.array(df.iloc[:,0])
        y = np.array(df.iloc[:,1])
        # print(type(X), X.shape)
        # print(type(y), y.shape)
        X = X.reshape(-1,1)
        # print(type(X), X.shape)
        
        num_samples = len(y)
        num_train, num_val = int(num_samples*0.7), int(num_samples*0.1)
        num_test = num_samples - num_train - num_val
        
        
        index_train = np.random.choice(num_samples, num_train, replace=False)
        index_train.sort()
        
        index_ = list(set(list(range(num_samples))) - set(index_train))
        index_val = np.random.choice(index_, num_val, replace=False) 
        index_val.sort()
        index_test = list(set(index_) - set(index_val))
        index_test.sort()
        
        # print('\n%d train, %d val, %d test'%(num_train, num_val, num_test))
        # print(index_train)
        # print(index_val)
        # print(index_test)

        X_tr, X_te, X_val = X[index_train], X[index_test], X[index_val]
        y_tr, y_te, y_val = y[index_train], y[index_test], y[index_val]
    
    
    y_tr = copy.copy(y_tr) # 这一步关键，保证之后y_tr的改变不会改变y。in order to not change y, as y_r is the slice of y 

    rng = np.random.RandomState(noise_seed) # seed for generate outliers, this can be change with input seed
    if args.robust == 'ht': # heavy-tails
        if args.noise_type == 'Normal':
            e = rng.normal(0., sigma, num_train)
            sd = sigma
            # print('\n%d-th sigma=%s sd=%s:'%(i, sigma, sd))
        elif args.noise_type == 'Laplace':
            e = rng.laplace(0., sigma, num_train)
            sd = np.sqrt(2*sigma**2)
        elif args.noise_type == 'Lognormal':
            e = rng.lognormal(0., sigma, int(num_samples*0.6))
            sd = np.sqrt((np.exp(sigma**2)-1)*np.exp(sigma**2))
        elif args.noise_type == 'Pareto':
            e = rng.pareto(sigma, num_train)
            sd = sd_hat = np.std(y_tr)  
        else:
            raise Exception('The noise type is wrong!')
        print('\nsigma=%.3f sd=%.3f:'%(sigma, sd))
        y_tr = y_tr + e

    elif args.robust == 'ct': # contanimation

        num_outlier = int(num_train*epsilon)
        # outlier_index = random.sample(range(num_train), num_outlier)
        outlier_index = np.random.choice(num_train, num_outlier, replace=False)

        mu = 0
        if args.noise_type == 'Normal':
            e = rng.normal(mu, sigma, num_outlier)
        elif args.noise_type == 'Laplace':
            e = rng.laplace(mu, sigma, num_outlier)  
        elif args.noise_type == 'Lognormal':
            e = rng.lognormal(mu, sigma, num_outlier)
        elif args.noise_type == 'Pareto':
            e = rng.pareto(sigma, num_outlier)
        else:
            raise Exception('The noise type is wrong!')
        y_tr[outlier_index] += e

        if args.noisy_val: 
            num_outlier = int(num_val*epsilon)
            outlier_index = np.random.choice(num_val, num_outlier, replace=False)

            mu = 0
            if args.noise_type == 'Normal':
                e = rng.normal(mu, sigma, num_outlier)
            elif args.noise_type == 'Laplace':
                e = rng.laplace(mu, sigma, num_outlier)  
            elif args.noise_type == 'Lognormal':
                e = rng.lognormal(mu, sigma, num_outlier)
            elif args.noise_type == 'Pareto':
                e = rng.pareto(sigma, num_outlier)
            else:
                raise Exception('The noise type is wrong!')
            y_val[outlier_index] += e

    output_folder = os.path.dirname(os.path.dirname(os.getcwd()))
    output_folder = os.path.join(output_folder, 'adaLoss_results')
    output_folder_ = os.path.dirname(os.getcwd())
    output_folder_ = os.path.join(output_folder_, 'results')
    
    fun_name = args.func+'_%dseed_%dnum(%d-%d-%d)'%(args.data_seed, num_samples, num_train, num_val, num_test)
    output_folder = os.path.join(output_folder, fun_name)
    output_folder_ = os.path.join(output_folder_, fun_name)

    name = args.robust+'_'+args.noise_type+'(%sbias)'%mu
    output_folder = os.path.join(output_folder, name)
    output_folder_ = os.path.join(output_folder_, name)

    name = '%dt+v'%args.train_add_val+'_'+'%dnoisy-val'%args.noisy_val+'_'+"%sbmodel"%(args.bmodel)+'_'+args.adap_loss+'+'+args.outer_loss
    output_folder = os.path.join(output_folder, name)
    output_folder_ = os.path.join(output_folder_, name)
    if not os.path.exists(output_folder_):
        os.makedirs(output_folder_)

    name1 = '%dseed'%noise_seed
    output_folder1 = os.path.join(output_folder, name1)
    output_folder1_ = os.path.join(output_folder_, name1)
    try:
        if 'epsilons' in args.train_mode:
            output_folder1 = os.path.join(output_folder1, 'epsilons')
            output_folder1_ = os.path.join(output_folder1_, 'epsilons')
        elif 'sigmas' in args.train_mode:
            output_folder1 = os.path.join(output_folder1, 'sigmas')
            output_folder1_ = os.path.join(output_folder1_, 'sigmas')
    except AttributeError:
        pass
    if args.robust == 'ht':
        name2 = '%.3fsigma'%sigma
    elif args.robust == 'ct':
        name2 = '%.3fsigma+'%sigma+'%.3fepsilon'%epsilon
    output_folder2 = os.path.join(output_folder1, name2)
    output_folder2_ = os.path.join(output_folder1_, name2)
    
    if not os.path.exists(output_folder2):
        os.makedirs(output_folder2)
        print(output_folder2)
        
    # print('\n', name, name1, name2)
    
    if num_dims == 1:
        plt.figure()
        plt.title(name1+'_'+name2)
        plt.plot(X_tr, y_tr, '.', label='train corrupt') #markersize
        plt.plot(X_val, y_val, '*', label='validation')
        plt.plot(X_te, y_te, '.', label='test')
        # plt.plot(X, y, 'o',  label='ground truth')
        plt.legend()
        # plt.ylim(bottom=0)
        plt.savefig(os.path.join(output_folder2, 'data-%s.png'%(name+'_'+name1+'_'+name2)))
        # plt.savefig(os.path.join(output_folder2_, 'data-%s.png'%(name+'_'+name1+'_'+name2)))
        plt.close('all') 

    elif num_dims == 2:
        plt.figure()
        ax1 = plt.axes(projection='3d')
        ax1.set_title(name1+'_'+name2)
        ax1.plot_surface(x0,x1,Z,color='w') # cmap='rainbow''coolwarm'
        ax1.scatter3D(X_tr[:,0], X_tr[:,1], y_tr, '.', s=10, alpha=0.6, label='train corrupt')
        ax1.scatter3D(X_val[:,0], X_val[:,1], y_val, '*', s=10, alpha=0.6, label='validation')
        ax1.scatter3D(X_te[:,0], X_te[:,1], y_te, '.', s=10, alpha=0.6, label='test')
        ax1.legend()
        # 调整角度
        # plt.show()
        # ax1.view_init(elev=5,    # 仰角
        #      azim=121    # 方位角
        #     )
        plt.savefig(os.path.join(output_folder2, 'data-%s.png'%(name+'_'+name1+'_'+name2)))
        
        plt.close('all') 


    X_tr = torch.from_numpy(X_tr).float()
    y_tr = torch.from_numpy(y_tr).float()
    X_te = torch.from_numpy(X_te).float()
    y_te = torch.from_numpy(y_te).float()
    X_val = torch.from_numpy(X_val).float()
    y_val = torch.from_numpy(y_val).float()

    train_data = TensorDataset(X_tr, y_tr)
    test_data = TensorDataset(X_te, y_te)
    val_data = TensorDataset(X_val, y_val)
    # val_data, test_data = random_split(test_data, lengths=[num_val, num_test])

    train_loader = DataLoader(train_data, args.batch_size, shuffle=True)
    val_loader = DataLoader(val_data, args.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, args.batch_size, shuffle=True)
    
    # print('\nnum of training data %d = batch size %d * iteration %d\n'%(len(train_loader.dataset),  args.batch_size, len(train_loader)))
    
    return (X_tr, y_tr), (X_val,y_val), (X_te, y_te), (name, name1, name2), \
            (output_folder, output_folder1, output_folder2), \
                (output_folder_, output_folder1_, output_folder2_), \
                    (train_loader, val_loader, test_loader)

