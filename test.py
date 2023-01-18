import torch
import torch.nn as nn

from data_generator import generate_data1
from train import build_model, test_loop
from arguments import get_args


args = get_args()


(X_tr, y_tr), (X_val,y_val), (X_te, y_te), (name, name1, name2),\
         (output_folder, output_folder1, output_folder2), (output_folder_, output_folder1_, output_folder2_),\
              (train_loader, val_loader, test_loader) = generate_data1(args, noise_seed=6, sigma=1.2, epsilon=0.4)

loss_fn2 = nn.MSELoss()
torch.manual_seed(0)
for _ in range(2):
    
    model = build_model(args.bmodel)
    best_result = test_loop(test_loader, model, loss_fn2)
    print('%d: %s'%(_, best_result))