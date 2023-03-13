# run linear problem without sm package but NN model to fit
# python main.py --func linear01 --data_seed 555 --train_mode sigmas-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0 
# python main.py --func linear01 --data_seed 555 --train_mode epsilons-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0 

# python main.py --func linear01 --data_seed 555 --train_mode sigmas-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0 
# python main.py --func linear01 --data_seed 555 --train_mode epsilons-seeds --epochs 3000 --num_seeds 15 --noise_type Lognormal --bflag 0



# func='5d02'
func='real-rail'
# model='50-80-10'
# model='80-100-30'
model='160-80-10'

# nohup python -u main.py --func ${func} --data_seed 555 --epochs 600 --num_seeds 5 --noise_type Lognormal --model ${model} --bmodel ${model} --train_mode sigmas-seeds > ../results/${func}_huber_sigmas_Lognormal_5sd.log 2>&1 & 
# nohup python -u main.py --func ${func} --data_seed 555 --epochs 600 --num_seeds 5 --noise_type Lognormal --model ${model} --bmodel ${model} --train_mode epsilons-seeds > ../results/${func}_huber_epsilons_Lognormal_5sd.log 2>&1 &

# nohup python -u main.py --func ${func} --data_seed 555 --epochs 600 --num_seeds 5 --noise_type Lognormal --model ${model} --bmodel ${model} --train_mode sigmas-seeds --adap_loss Tukey --hypara 4.685 > ../results/${func}_tukey_sigmas_Lognormal_5sd.log 2>&1 & 
# nohup python -u main.py --func ${func} --data_seed 555 --epochs 600 --num_seeds 5 --noise_type Lognormal --model ${model} --bmodel ${model} --train_mode epsilons-seeds --adap_loss Tukey --hypara 4.685 > ../results/${func}_tukey_epsilons_Lognormal_5sd.log 2>&1 &

# ------

# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func ${func} --data_seed 555 --epochs 600 --num_seeds 5 --noise_type Lognormal --model ${model} --bmodel ${model} --train_mode sigmas-seeds > ../results/${func}_huber_sigmas_Lognormal_5sd.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func ${func} --data_seed 555 --epochs 600 --num_seeds 5 --noise_type Lognormal --model ${model} --bmodel ${model} --train_mode epsilons-seeds > ../results/${func}_huber_epsilons_Lognormal_5sd.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func ${func} --data_seed 555 --epochs 600 --num_seeds 5 --noise_type Lognormal --model ${model} --bmodel ${model} --train_mode sigmas-seeds --adap_loss Tukey --hypara 4.685 > ../results/${func}_tukey_sigmas_Lognormal_5sd.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func ${func} --data_seed 555 --epochs 600 --num_seeds 5 --noise_type Lognormal --model ${model} --bmodel ${model} --train_mode epsilons-seeds --adap_loss Tukey --hypara 4.685 > ../results/${func}_tukey_epsilons_Lognormal_5sd.log 2>&1 &

#----- func 2d01 40min 25 min one seed

# python train.py --func 2d02 --data_seed 555 --mode none --noise_type Lognormal
# python main.py --func 2d01 --data_seed 555 --train_mode regular --epochs 600 --num_seeds 3 --noise_type Lognormal

# nohup python -u main.py --func 2d01 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 3 --noise_type Lognormal
# nohup python -u main.py --func 2d01 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 3 --noise_type Lognormal > ../results/2d01_epsilons_Lognormal.log 2>&1 &

# nohup python -u main.py --func 2d01 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 > ../results/2d01_sigmas_Lognormal_tukey.log 2>&1 &
# nohup python -u main.py --func 2d01 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 > ../results/2d01_epsilons_Lognormal_tukey.log 2>&1 &  

# nohup python -u main.py --func 2d01 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 3 --noise_type Laplace > ../results/2d01_epsilons_Laplace.log 2>&1 &
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 2d01 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 3 --noise_type Laplace > ../results/2d01_sigmas_Laplace.log 2>&1 &  

#----- func 2d02 | 25-30min/seed [50, 98, 54, 6, 34]
# python train.py --func 2d02 --data_seed 555 --mode none --noise_type Lognormal

# nohup python -u main.py --func 2d02 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal > ../results/2d02_sigmas_Lognormal_5sd.log 2>&1 & 
# nohup python -u main.py --func 2d02 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal > ../results/2d02_epsilons_Lognormal_5sd.log 2>&1 &

# nohup python -u main.py --func 2d02 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 > ../results/2d02_sigmas_Lognormal_5sd_tukey.log 2>&1 & 
# nohup python -u main.py --func 2d02 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 > ../results/2d02_epsilons_Lognormal_5sd_tukey.log 2>&1

#----- func 2d03 | 25-30min/seed [50, 98, 54, 6, 34]

# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 2d03 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal > ../results/2d03_sigmas_Lognormal_5sd.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 2d03 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal > ../results/2d03_epsilons_Lognormal_5sd.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 2d03 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 > ../results/2d03_sigmas_Lognormal_5sd_tukey.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 2d03 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 > ../results/2d03_epsilons_Lognormal_5sd_tukey.log 2>&1


#----- func 3d02 | 
# nohup python -u main.py --func 3d02 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal > ../results/3d02_sigmas_Lognormal_5sd.log 2>&1 & 
# nohup python -u main.py --func 3d02 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal > ../results/3d02_epsilons_Lognormal_5sd.log 2>&1 &

# nohup python -u main.py --func 3d02 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 > ../results/3d02_sigmas_Lognormal_5sd_tukey.log 2>&1 & 
# nohup python -u main.py --func 3d02 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 > ../results/3d02_epsilons_Lognormal_5sd_tukey.log 2>&1

#----- func 3d03 
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 3d03 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal > ../results/3d03_huber_sigmas_Lognormal_5sd.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 3d03 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal > ../results/3d03_huber_epsilons_Lognormal_5sd.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 3d03 --data_seed 555 --train_mode sigmas-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 > ../results/3d03_tukey_sigmas_Lognormal_5sd.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 3d03 --data_seed 555 --train_mode epsilons-seeds --epochs 600 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 > ../results/3d03_tukey_epsilons_Lognormal_5sd.log 2>&1


#----- func 10d02 | 25-30min/seed [50, 98, 54, 6, 34]

# python train.py --func 10d02 --data_seed 555 --mode none --noise_type Lognormal

# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 10d02 --data_seed 555 --epochs 500 --num_seeds 5 --noise_type Lognormal --train_mode sigmas-seeds > ../results/10d02_sigmas_Lognormal_5sd_huber.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 10d02 --data_seed 555 --epochs 500 --num_seeds 5 --noise_type Lognormal --train_mode epsilons-seeds > ../results/10d02_epsilons_Lognormal_5sd_huber.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 10d02 --data_seed 555 --epochs 500 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 --train_mode sigmas-seeds > ../results/10d02_sigmas_Lognormal_5sd_tukey.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 10d02 --data_seed 555 --epochs 500 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 --train_mode epsilons-seeds > ../results/10d02_epsilons_Lognormal_5sd_tukey.log 2>&1

# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 10d02 --data_seed 555 --epochs 800 --num_seeds 5 --noise_type Lognormal --model 80-100-30 --bmodel 80-100-30 --train_mode sigmas-seeds > ../results/10d02_sigmas_Lognormal_5sd_huber.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 10d02 --data_seed 555 --epochs 800 --num_seeds 5 --noise_type Lognormal --model 80-100-30 --bmodel 80-100-30 --train_mode epsilons-seeds > ../results/10d02_epsilons_Lognormal_5sd_huber.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 10d02 --data_seed 555 --epochs 800 --num_seeds 5 --noise_type Lognormal --model 80-100-30 --bmodel 80-100-30 --train_mode sigmas-seeds --adap_loss Tukey --hypara 4.685 > ../results/10d02_sigmas_Lognormal_5sd_tukey.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 10d02 --data_seed 555 --epochs 800 --num_seeds 5 --noise_type Lognormal --model 80-100-30 --bmodel 80-100-30 --train_mode epsilons-seeds --adap_loss Tukey --hypara 4.685 > ../results/10d02_epsilons_Lognormal_5sd_tukey.log 2>&1



#----- func 10d03 

# python train.py --func 10d03 --data_seed 555 --mode none --noise_type Lognormal

# nohup python -u main.py --func 10d03 --data_seed 555 --epochs 500 --num_seeds 5 --noise_type Lognormal --train_mode sigmas-seeds > ../results/10d03_sigmas_Lognormal_5sd_huber.log 2>&1 & 
# nohup python -u main.py --func 10d03 --data_seed 555 --epochs 500 --num_seeds 5 --noise_type Lognormal --train_mode epsilons-seeds > ../results/10d03_epsilons_Lognormal_5sd_huber.log 2>&1 &

# nohup python -u main.py --func 10d03 --data_seed 555 --epochs 500 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 --train_mode sigmas-seeds > ../results/10d03_sigmas_Lognormal_5sd_tukey.log 2>&1 & 
# nohup python -u main.py --func 10d03 --data_seed 555 --epochs 500 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 --train_mode epsilons-seeds > ../results/10d03_epsilons_Lognormal_5sd_tukey.log 2>&1


# 80-100-30
# nohup python -u main.py --func 10d03 --data_seed 555 --epochs 800 --num_seeds 5 --noise_type Lognormal --model 80-100-30 --bmodel 80-100-30 --train_mode sigmas-seeds > ../results/10d03_sigmas_Lognormal_5sd_huber.log 2>&1 & 
# nohup python -u main.py --func 10d03 --data_seed 555 --epochs 800 --num_seeds 5 --noise_type Lognormal --model 80-100-30 --bmodel 80-100-30 --train_mode epsilons-seeds > ../results/10d03_epsilons_Lognormal_5sd_huber.log 2>&1 &

# nohup python -u main.py --func 10d03 --data_seed 555 --epochs 800 --num_seeds 5 --noise_type Lognormal --model 80-100-30 --bmodel 80-100-30 --adap_loss Tukey --hypara 4.685 --train_mode sigmas-seeds > ../results/10d03_sigmas_Lognormal_5sd_tukey.log 2>&1 & 
# nohup python -u main.py --func 10d03 --data_seed 555 --epochs 800 --num_seeds 5 --noise_type Lognormal --model 80-100-30 --bmodel 80-100-30 --adap_loss Tukey --hypara 4.685 --train_mode epsilons-seeds > ../results/10d03_epsilons_Lognormal_5sd_tukey.log 2>&1


#----- func 10d05 

# python train.py --func 10d05 --data_seed 555 --mode none --noise_type Lognormal

# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 10d05 --data_seed 555 --epochs 500 --num_seeds 5 --noise_type Lognormal --train_mode sigmas-seeds > ../results/10d05_sigmas_Lognormal_5sd_huber.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 10d05 --data_seed 555 --epochs 500 --num_seeds 5 --noise_type Lognormal --train_mode epsilons-seeds > ../results/10d05_epsilons_Lognormal_5sd_huber.log 2>&1 &

# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 10d05 --data_seed 555 --epochs 500 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 --train_mode sigmas-seeds > ../results/10d05_sigmas_Lognormal_5sd_tukey.log 2>&1 & 
# CUDA_VISIBLE_DEVICES=1 nohup python -u main.py --func 10d05 --data_seed 555 --epochs 500 --num_seeds 5 --noise_type Lognormal --adap_loss Tukey --hypara 4.685 --train_mode epsilons-seeds > ../results/10d05_epsilons_Lognormal_5sd_tukey.log 2>&1


#----- real-rail
# python train.py --func real-rail --data_seed 555 --mode none --noise_type Lognormal

# nohup python -u main.py --func real-rail --data_seed 555 --epochs 600 --num_seeds 5 --noise_type Lognormal --train_mode sigmas-seeds > ../results/real-rail_sigmas_Lognormal_5sd_huber.log 2>&1 & 
# nohup python -u main.py --func real-rail --data_seed 555 --epochs 600 --num_seeds 5 --noise_type Lognormal --train_mode epsilons-seeds > ../results/real-rail_epsilons_Lognormal_5sd_huber.log 2>&1 &

# nohup python -u main.py --func real-rail --data_seed 555 --epochs 600 --num_seeds 5 --noise_type Lognormal --train_mode sigmas-seeds --adap_loss Tukey --hypara 4.685 > ../results/real-rail_sigmas_Lognormal_5sd_tukey.log 2>&1 & 
# nohup python -u main.py --func real-rail --data_seed 555 --epochs 600 --num_seeds 5 --noise_type Lognormal --train_mode epsilons-seeds --adap_loss Tukey --hypara 4.685 > ../results/real-rail_epsilons_Lognormal_5sd_tukey.log 2>&1 &