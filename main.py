from calendar import c
from math import fabs
import os
import re
import random
import time
from datetime import datetime
from tqdm import tqdm

import numpy as np
import matplotlib
matplotlib.use('Agg') # Linux绘图需要
from matplotlib import pyplot as plt
plt.rcParams.update({'figure.max_open_warning': 0})
import pandas as pd

# import statsmodels.api as sm
from sklearn.metrics import mean_squared_error as mse
# from sklearn.metrics import mean_absolute_error as mae

from arguments import get_args
from train import meta_exec, normal_exec
from data_generator import generate_data1


############################


def test_all_methods(noise_seed, sigma=3., epsilon=0.3):

    (X_tr, y_tr), (X_val,y_val), (X_te, y_te),  (name, name1, name2), \
    (output_folder, output_folder1, output_folder2), (output_folder_, output_folder1_, output_folder2_), \
    (train_loader, val_loader, test_loader) = generate_data1(args, noise_seed, sigma, epsilon)
    
    timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
    name_df = '%s_%s_%s'%(name1, name2, timenow)
    print(name_df)

    f = open(os.path.join(output_folder2,'result-%s.txt'%name_df), 'w')
    f.write('args:%s'%name+ '\n')
    f.write('noise_seed:%s'%name1+ '\n')
    f.write('sigma&epsilon:%s'%name2+ '\n')
    f.write('\n')

    methods_pred_mse_list = []
    methods_predmse_epochs_dict = {}
    
    if 'linear0' in args.func and (not args.bflag):
        
        if args.train_add_val:
            XX_tr = sm.add_constant(np.vstack((X_tr.numpy(), X_val.numpy()))) # X_tr (N,1)
            XX_te = sm.add_constant(X_te.numpy())
            yy_tr = np.hstack((y_tr.numpy(), y_val.numpy())) # y_tr (N,)
            yy_te = y_te.numpy()
        else:
            XX_tr = sm.add_constant(X_tr.numpy())
            XX_te = sm.add_constant(X_te.numpy())
            yy_tr, yy_te = y_tr.numpy(), y_te.numpy()
        if len(XX_tr) == len(yy_tr):
                pass
        else:
            raise Exception('The training data size is mismatch!') 

        plt.figure(figsize=(17,13)) 
        for (j, method) in enumerate(method1):
            if method == 'OLS':
                result = sm.OLS(yy_tr, XX_tr).fit()
                # LinearRegression(fit_intercept=False).fit(X, y).coef_
            elif method == 'LAD':
                result = sm.QuantReg(yy_tr, XX_tr).fit(q=.5)
            elif method == 'Tukey':
                result = sm.RLM(yy_tr, XX_tr, M=sm.robust.norms.TukeyBiweight(c=4.685)).fit()
            elif method == 'Huber':
                result = sm.RLM(yy_tr, XX_tr, M=sm.robust.norms.HuberT(t=1.345)).fit()
            # elif method == 'Huber2':
                # if args.noise_type == 'pareto':
                    # T = sd_hat*np.sqrt(int(train_size)/d)
                # else:
                    # T = sd*np.sqrt(int(train_size)/d)
                # print('T:', T)
                # result = sm.RLM(yy_tr, XX_tr, M=sm.robust.norms.HuberT(T)).fit()
            elif method == 'Huber-LAD':
                result = sm.RLM(yy_tr, XX_tr, M=sm.robust.norms.HuberT(t=0.01)).fit()
            elif method == 'Huber-OLS':
                result = sm.RLM(yy_tr, XX_tr, M=sm.robust.norms.HuberT(t=100)).fit()
            elif 'Huber(' in method:
                # print(method)
                tau = float(re.findall(r"\d+\.?\d*", method)[0])
                result = sm.RLM(yy_tr, XX_tr, M=sm.robust.norms.HuberT(t=tau)).fit()
            elif 'Tukey(' in method:
                tau = float(re.findall(r"\d+\.?\d*", method)[0])
                result = sm.RLM(yy_tr, XX_tr, M=sm.robust.norms.TukeyBiweight(c=tau)).fit()
            
                
            yy_te_hat = result.predict(XX_te) # array(N, )
            # pred_mae = mae(yy_te_hat, yy_te)
            # list_pred_mae.append(pred_mae)
            pred_mse = mse(yy_te_hat, yy_te)
            f.write('%s: %.4f'%(method, pred_mse)+ '\n')
            methods_pred_mse_list.append(pred_mse) 
            # dict_pred_mse[method] = pred_mse
            
            num_standard_method = len(method1)
            plt.subplot(2, int(num_standard_method/2)+num_standard_method%2, j+1)
            plt.title(method)
            plt.plot(XX_tr[:,1], yy_tr, 'o', label='train')
            if args.train_add_val:
                pass
            else:
                plt.plot(X_val.numpy(), y_val.numpy(), 'o', label='val')
            plt.plot(XX_te[:,1], yy_te, 'o', label='test')
            plt.plot(XX_te[:,1], yy_te_hat, 'o', label='fitted')
            
        plt.legend(loc=1)
        plt.savefig(os.path.join(output_folder2, 'fit-'+name_df+'.png'))
        plt.close('all') 
        
    else:
        for method in method2:
            pred_mse, best_performance, pred_mse_epochs_list = normal_exec(args, method, train_loader, val_loader, test_loader, \
                X_tr, y_tr, X_val, y_val, X_te, y_te, output_folder2, name1, name2, mode='normal')
            
            best_test =  min(pred_mse_epochs_list)
            best_test_epoch = pred_mse_epochs_list.index(best_test)
            f.write('%s: best val %s and its test %.3f | the best test %.3f(%d)!'%\
                (method, best_performance, pred_mse, best_test, best_test_epoch)+ '\n')

            methods_pred_mse_list.append(pred_mse)
            methods_predmse_epochs_dict[method] = pred_mse_epochs_list
    
    hy_dict = {}
    if len(method3) != 0:
        for mode in method3:
            # print('\nStart the %s mode'%mode)
            pred_mse, best_performance, pred_mse_epochs_list, converge_hy = meta_exec(args, train_loader, val_loader, test_loader,\
                X_tr, y_tr, X_val, y_val, X_te, y_te, output_folder2, name1, name2, mode)
            
            best_test =  min(pred_mse_epochs_list)
            best_test_epoch = pred_mse_epochs_list.index(best_test)
            f.write('%s: best val %s and its test %.3f | the best test %.3f(%d) | hy converge to %.3f!!'%\
                (mode, best_performance, pred_mse, best_test, best_test_epoch, converge_hy)+ '\n')
            
            methods_pred_mse_list.append(pred_mse)
            methods_predmse_epochs_dict[mode] = pred_mse_epochs_list
            hy_dict[mode] = converge_hy
            
        print('\n==========meta Done=============\n')
    
    
    f.close

    df_epochs = pd.DataFrame(methods_predmse_epochs_dict)
    df_epochs.to_csv(os.path.join(output_folder2, 'epochs-'+name_df+'.csv'), index=False)
    
    plt.figure()
    plt.title('Prediction in test data')
    method_name = list(df_epochs.columns)
    for _ in method_name:
        plt.plot(list(df_epochs[_]), label='%s'%_)
    plt.legend()
    plt.xlabel('Epochs')
    plt.ylabel('MSE')
    plt.ylim(top=2) 
    plt.savefig(os.path.join(output_folder2, 'epochs-'+name_df+'.png'))
    plt.close('all') 
    
    return methods_pred_mse_list, hy_dict, (output_folder, output_folder1), (output_folder_, output_folder1_), (name, name1, name2)

### This func works as father func for test_all_methods_sigmas & test_all_methods_epsilons
def test_all_methods_ses(name_, list_, val_, noise_seed, flag=True):
    '''
    name_ shows this is with different sigma or epsilon
    list_ is the list of sigma or list of epsilon
    val_ is counterpart value of name_
    '''
    if args.robust == 'ht':
        raise Exception('This only works for contamination case')
    # name_: 'sigmas' or 'epsilons'
    if flag:
        if name_ == 'sigmas':
            print('=Start all methods with different sigmas %s and epsilon %s in seed %s'%(list_, val_, noise_seed))
        elif name_ == 'epsilons':
            print('=Start all methods with different epsilons %s and sigma %s in seed %s'%(list_, val_, noise_seed))
    
    num_ = len(list_)
    matrix_pred_mse = np.zeros((num_, num_method))
    hy_dict1 = {'seed':noise_seed}
    hy_dict2 = {}
    
    for (i, _) in enumerate(list_):
        
        if name_ == 'sigma':
            if flag:
                print('===start %d-th all methods with sigma(%.4f) & epsilon(%s)'%(i, _, val_))
            methods_pred_mse_list, hy_dict, (output_folder, output_folder1), \
                (output_folder_, output_folder1_), (name, name1, name2)\
            = test_all_methods(noise_seed, sigma=_, epsilon=val_)
        elif name_ == 'epsilon':
            if flag:
                print('===start %d-th all methods with epsilon(%.4f) & sigma(%s)'%(i, _, val_))
            methods_pred_mse_list, hy_dict, (output_folder, output_folder1), \
                (output_folder_, output_folder1_), (name, name1, name2)\
            = test_all_methods(noise_seed, epsilon=_, sigma=val_)
        # print(len(methods_pred_mse_list))
        matrix_pred_mse[i, :] = methods_pred_mse_list
        if 'meta3' in method3:
            hy_dict1[name2] = hy_dict['meta3']
        if 'meta2' in method3:
            hy_dict2[name2] = hy_dict['meta2']
            
    # save the table
    timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
    if name_ == 'sigma':
        name_plt = os.path.join(output_folder1, 'predmse_%.3fepsilons_%dmethods_%s_%s_%s'%(val_, num_method, name, name1, timenow))
        # name_plt_ = os.path.join(output_folder1_, 'predmse_%.3fepsilons_%dmethods_%s_%s_%s'%(val_, num_method, name, name1, timenow))
    elif name_ == 'epsilon':
        name_plt = os.path.join(output_folder1, 'predmse_%.3fsigmas_%dmethods_%s_%s_%s'%(val_, num_method, name, name1, timenow))
        # name_plt_ = os.path.join(output_folder1_, 'predmse_%.3fsigmas_%dmethods_%s_%s_%s'%(val_, num_method, name, name1, timenow))
    df = np.column_stack((list_, matrix_pred_mse))
    df = pd.DataFrame(df)
    df.columns = [name_] + methods
    # name_df = os.path.join(output_folder1, name+'_'+name1+'.xlsx')
    name_df = name_plt + '.csv'
    # name_df_ = name_plt_ + '.csv'
    df.to_csv(name_df, index=False)
    # df.to_csv(name_df_, index=False)

    # visualize all the methods
    fig, ax = plt.subplots() #figsize=(15,12)
    # fig.suptitle(name1)
    # ax.set_title('pred MSE')
    for i in range(num_method): 
        ax.plot(list_, matrix_pred_mse[:,i], 'o-', label='%s'%methods[i])
    ax.set_xlabel((r'%s distribution with value of $\%s$'%(args.noise_type, name_)))
    # ax.set_ylabel('Prediction MSE')
    ax.set_ylabel('Test MSE')
    ax.legend(loc=2)
    fig.savefig(name_plt+'.png')
    # fig.savefig(name_plt_+'.png')
    del fig
    plt.close('all') 

    #  visualize without the OLS and meta2
    if name_ == 'sigma':
        name_plt = os.path.join(output_folder1, 'predmse_%.3fepsilons_%dmethods_%s_%s_%s'%(val_, num_method-2, name, name1, timenow))
        # name_plt_ = os.path.join(output_folder1_, 'predmse_%.3fepsilons_%dmethods_%s_%s_%s'%(val_, num_method-1, name, name1, timenow))
    elif name_ == 'epsilon':
        name_plt = os.path.join(output_folder1, 'predmse_%.3fsigmas_%dmethods_%s_%s_%s'%(val_, num_method-2, name, name1, timenow))
        # name_plt_ = os.path.join(output_folder1_, 'predmse_%.3fsigmas_%dmethods_%s_%s_%s'%(val_, num_method-1, name, name1, timenow))
    plt.figure()
    for method in methods[1:-1]:
        plt.plot(list_, df[method].values, 'o-', label='%s'%method)
    plt.xlabel((r'%s distribution with value of $\%s$'%(args.noise_type, name_)))
    plt.ylabel('Prediction MSE')
    plt.legend(loc=2)
    plt.savefig(name_plt+'.png')
    # plt.savefig(name_plt_+'.png')
    plt.close('all') 
    
    return matrix_pred_mse, (hy_dict1, hy_dict2), (output_folder, output_folder_, name)


def test_all_methods_ses_seeds(num_seeds, name_, list_, val_):

    

    # num_method = len(methods)
    num_ = len(list_)
    tensor_pred_mse = np.zeros((num_seeds, num_, num_method))
    random.seed(0)
    noise_seeds = random.sample(range(1, 100), num_seeds)
    
    hy_list1 = []
    hy_list2 = []

    for (i, noise_seed) in enumerate(noise_seeds):
        print('\n=====================>>><<<<=======================')
        start_time = time.time()
        tensor_pred_mse[i,:,:], (hy_dict1, hy_dict2),\
             (output_folder, output_folder_, name) = test_all_methods_ses(name_, list_, val_, noise_seed, flag=True) # flag=False
        end_time = time.time()
        print("\n---> Done {:d}-th seed {:d} cost : {:.4f}s\n".format(i, noise_seed, end_time-start_time))
        # print('\n==========================')
        
        hy_list1.append(hy_dict1)
        hy_list2.append(hy_dict2)
    
    ave_matrix_pred_mse = np.mean(tensor_pred_mse, axis=0)
    std_matrix_pred_mse = np.std(tensor_pred_mse, axis=0)
    
    timenow = datetime.now().strftime("%Y%m%d-%H%M%S")
    if name_ == 'sigma':
        name_df = '%s+%.3f%s_ave_%dseed_%dmethods_%s_%s'%(name_, val_, 'epsilons',num_seeds, num_method, name, timenow)
    elif name_ == 'epsilon':
        name_df = '%s+%.3f%s_ave_%dseed_%dmethods_%s_%s'%(name_, val_, 'sigmas',num_seeds, num_method, name, timenow)
    name_df1 = os.path.join(output_folder, name_df)
    name_df1_ = os.path.join(output_folder_, name_df)
    print(name_df1)
    print(name_df1_)
    
    hy_df1 = pd.DataFrame(hy_list1)
    hy_df2 = pd.DataFrame(hy_list2)
    hy_df = pd.concat([hy_df1, hy_df2], axis=1)
    hy_df.to_csv(os.path.join(output_folder, 'hy-'+name_df+'.csv'), index=False)
    hy_df.to_csv(os.path.join(output_folder_, 'hy-'+name_df+'.csv'), index=False)
    
    df1_pred = np.column_stack((list_, ave_matrix_pred_mse))
    df2_pred = np.column_stack((list_, std_matrix_pred_mse))
    df_pred = np.row_stack((df1_pred, df2_pred))
    df_pred = pd.DataFrame(df_pred)
    df_pred.columns = [name_] + methods
    df_pred.to_csv(name_df1+'.csv', index=False)
    df_pred.to_csv(name_df1_+'.csv', index=False)
    
    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize=(15,12))
    # fig.suptitle(name1)
    # ax.set_title('ave %d seed pred MSE'%num_seeds)
    for i in range(num_method): 
        ax.errorbar(list_, ave_matrix_pred_mse[:,i], \
            yerr=std_matrix_pred_mse[:,i], fmt='o-', label='%s'%methods[i], capsize=3, capthick=2)
    ax.set_xlabel(r'%s distribution with value of $\%s$'%(args.noise_type, name_))
    ax.set_ylabel('Prediction MSE')
    ax.legend() # loc=2
    fig.savefig(name_df1+'.png')
    fig.savefig(name_df1_+'.png')


    #  visualize without the OLS and meta2
    fig, ax = plt.subplots()
    # fig, ax = plt.subplots(figsize=(15,12))
    # fig.suptitle(name1)
    # ax.set_title('ave %d seed pred MSE'%num_seeds)
    for i in range(1, num_method-1): 
        ax.errorbar(list_, ave_matrix_pred_mse[:,i], \
            yerr=std_matrix_pred_mse[:,i], fmt='o-', label='%s'%methods[i], capsize=3, capthick=2)
    ax.set_xlabel(r'%s distribution with value of $\%s$'%(args.noise_type, name_))
    ax.set_ylabel('Prediction MSE')
    ax.legend() # loc=2
    if name_ == 'sigma':
        name_df = '%s+%.3f%s_ave_%dseed_%dmethods_%s_%s'%(name_, val_, 'epsilons',num_seeds, num_method-2, name, timenow)
    elif name_ == 'epsilon':
        name_df = '%s+%.3f%s_ave_%dseed_%dmethods_%s_%s'%(name_, val_, 'sigmas',num_seeds, num_method-2, name, timenow)
    name_df2 = os.path.join(output_folder, name_df)
    name_df2_ = os.path.join(output_folder_, name_df)
    fig.savefig(name_df2+'.png')
    fig.savefig(name_df2_+'.png')

    print(noise_seeds)


if __name__ == '__main__':

    args = get_args()
    print('\n=======================\n')
    print(args)
    print('\n=======================')

    # # method3 = ['meta3', 'meta2']
    method3 = ['meta3']
    # method1 = ['OLS', 'LAD', 'Huber(1.345)', 'Tukey(4.685)']
    if args.adap_loss == 'Huber':
        # taus = np.linspace(0.001,100, 10)
        taus = [2.0102, 1.345, 0.7317, 0.307, 0.158] + [0.05, 0.01]
        method1 = ['OLS', 'LAD', 'Tukey(4.685)'] + ['Huber(%.3f)'%tau for tau in taus]
    elif args.adap_loss == 'Tukey':
        taus = [7.041, 4.685, 3.444, 1.722, 0.861]
        method1 = ['OLS', 'LAD', 'Huber(1.345)'] + ['Tukey(%.3f)'%tau for tau in taus]


    

    if not args.bflag and 'linear0' in args.func:
        print('Use method1')
        methods = method1+method3
    else:
        print('Use method2')
        method2 = method1
        methods = method2+method3
        
    print('All methods contain:', methods)
    num_method = len(methods)  # this is a global variable, which will be used in the functions above
    num_seeds = args.num_seeds

    num_sigma = 5
    if args.noise_type == 'Normal':
        list_sigma = np.linspace(1., 4., num_sigma)
        sigma = 3.5
    elif args.noise_type == 'Laplace':
        list_sigma = np.linspace(0.5, 4.5, num_sigma)
        sigma = 3.5
    elif args.noise_type == 'Lognormal':
        list_sigma = np.linspace(0.2, 1.3, num_sigma) # 1.5 will be too variate
        # [0.2, 0.475, 0.75, 1.025, 1.3]
        sigma = 1.2
    elif args.noise_type == 'Pareto':
        list_sigma = np.linspace(1.5, 3., num_sigma)
    
    epsilon = 0.25
    list_epsilon = [0.1, 0.2, 0.3, 0.4, 0.5]

    #==========================================#
    start_time = time.time()

    if args.train_mode == 'regular':
        methods_pred_mse_list, hy_dict, \
        (output_folder, output_folder1), (output_folder_, output_folder1_), (name, name1, name2) \
            = test_all_methods(noise_seed=65, sigma=sigma, epsilon=epsilon) #random.randint(0,100)
        
    elif 'sigmas' in args.train_mode:
        
        print('\nThis is epsilon %s & list_sigma %s'%(epsilon, list_sigma))

        if args.train_mode == 'sigmas':
            matrix_pred_mse = test_all_methods_ses('sigma', list_sigma, epsilon, noise_seed=random.randint(0,100))
        elif args.train_mode == 'sigmas-seeds':
            test_all_methods_ses_seeds(num_seeds, 'sigma', list_sigma, epsilon)

    elif 'epsilons' in args.train_mode: 

        print('\nThis is sigma %s & list_epsilon %s'%(sigma, list_epsilon))
        
        if args.train_mode == 'epsilons':
            matrix_pred_mse = test_all_methods_ses('epsilon', list_epsilon, sigma, noise_seed=random.randint(0,100))
        elif args.train_mode == 'epsilons-seeds':
            test_all_methods_ses_seeds(num_seeds, 'epsilon', list_epsilon, sigma)
        
      
    end_time = time.time()
    print("\nCost time : {:.4f}s\n".format(end_time - start_time))