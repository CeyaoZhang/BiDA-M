# run linear problem without sm package but NN model to fit
# python main.py --func linear01 --data_seed 555 --train_mode sigmas-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0 
# python main.py --func linear01 --data_seed 555 --train_mode epsilons-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0 

# python main.py --func linear01 --data_seed 555 --train_mode sigmas-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0 
# python main.py --func linear01 --data_seed 555 --train_mode epsilons-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0

#----- func 2d01 40min 25 min one seed
# python main.py --func 2d01 --data_seed 555 --train_mode regular --epochs 600 --num_seeds 3 --noise_type Lognormal
# nohup python -u main.py --func 2d01 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 3 --noise_type Lognormal
# nohup python -u main.py --func 2d01 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 3 --noise_type Lognormal > ../results/2d01_epislons_Lognormal.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 2d01 --data_seed 55 --train_mode sigmas-seeds --epochs 600 --num_seeds 3 --noise_type Lognormal > ../results/2d01_sigmas_Lognormal.log 2>&1 & 

# nohup python -u main.py --func 2d01 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 3 --noise_type Laplace > ../results/2d01_epislons_Laplace.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 2d01 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 3 --noise_type Laplace > ../results/2d01_sigmas_Laplace.log 2>&1 &  

#----- func 2d02
python train.py --func 2d02 --data_seed 555 --mode none --noise_type Lognormal