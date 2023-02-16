# run linear problem without sm package but NN model to fit
# python main.py --func linear01 --data_seed 555 --train_mode sigmas-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0 
# python main.py --func linear01 --data_seed 555 --train_mode epsilons-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0 

# python main.py --func linear01 --data_seed 555 --train_mode sigmas-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0 
# python main.py --func linear01 --data_seed 555 --train_mode epsilons-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0


# python main.py --func 2d01 --data_seed 555 --train_mode regular --epochs 600 --num_seeds 3 --noise_type Lognormal
python main.py --func 2d01 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 3 --noise_type Lognormal
python main.py --func 2d01 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 3 --noise_type Lognormal