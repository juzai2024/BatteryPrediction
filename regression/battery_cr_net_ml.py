import os
os.environ['CUDA_VISIBLE_DEVICES'] = '4'
import torch
import numpy as np
from sklearn.metrics import r2_score
from batter_fine_net_CR import prepare_data, fine as fine_cr
from ablation_exp import fine as fine_ablation
import re
import scipy.io as sio
import random
import json
random.seed(42)

def gen_data_path(battery_type="LCO", trial_index=1):
    base_path = "/path/to/mae/data/fixed_data_final"
    mixup_data_path = "/path/to/mae/data/mae_train_all"
    json_dir = "/path/to/mae/data/LCO_test_vs_ablation"
    base_data = os.listdir(base_path)
    base_data = [f for f in base_data if f.startswith(battery_type)]
    
    choice_data_file = [f for f in base_data if 'wo_10' not in f]
    test_indices = random.sample(range(len(choice_data_file)), 8)
    test_files = [choice_data_file[idx] for idx in test_indices]
    train_files = [f for f in base_data if f not in test_files]
    
    file_groups = {}

    for file in train_files:
            parts = file.split('_wo_')[0]
            battery_name, state = parts.split('_', 1)
            
            # Create key for grouping
            group_key = (battery_name, state)
            
            # Add file to group
            if group_key not in file_groups:
                file_groups[group_key] = []
            file_groups[group_key].append(file)
            
    save_path = mixup_data_path
    train_data = []
    expand_factor = 10
    # Generate weights with finer granularity based on expand_factor
    step = 1 / expand_factor  # smaller step size for higher granularity
    weight = []
    for i in range(1, expand_factor):  # more pairs based on expand_factor
        w1 = round(i * step, 3)  # round to 3 decimal places
        w2 = round(1 - w1, 3)
        weight.append((w1, w2))

    for (battery, state), files in file_groups.items():
        files = sorted(files)
        if len(files) < 2:
            print(f"Not enough files for {battery}_{state} to perform mixup.")
            continue
        for idx in range(len(files)-1):
            
            stage1 = int(re.search(r'wo_(\d+)', files[idx]).group(1))
            stage2 = int(re.search(r'wo_(\d+)', files[idx+1]).group(1))
            
            filename1 = os.path.basename(files[idx])
            train_data.append(os.path.join(save_path, filename1))
            
            filename2 = os.path.basename(files[idx+1])
            train_data.append(os.path.join(save_path, filename2))
            
            for wt1, wt2 in weight:
                new_filename = f"{battery}_{state}_wo_{stage1}_w{wt1}_{stage2}_w{wt2}.mat"
                train_data.append(os.path.join(save_path, new_filename))
            
    test_data = [os.path.join(base_path, f) for f in test_files]
    
    trial_json = {
        battery_type : {
            'train': train_data,
            'test': test_data
        }
    }
      
    with open(os.path.join(json_dir, f'trial_{trial_index}.json'), 'w') as f:
        json.dump(trial_json, f, indent=4)
    
            
    return os.path.join(json_dir, f'trial_{trial_index}.json')



def run_experiment(data_path, save_path, batter="LCO", r2_threshold=0.2, version=1):
    """
    Run experiment by combining methods from batter_fine_net_CR and ablation_exp.
    """
    # Run fine method from batter_fine_net_CR
    weight_path = "/path/to/mae/mae_weight_out/mae_train_all/checkpoint-final.pth"
    r2_cr = fine_cr(data_path, version=version, batter=batter, weight_path=weight_path, save_path=save_path)

    # Run fine method from ablation_exp
    r2_ablation = fine_ablation(data_path, version=version, batter=batter, weight_path=weight_path, save_path=save_path)

    # Compare R² scores
    if r2_cr > r2_ablation and r2_cr > r2_threshold:
        print(f"Experiment successful with R²: {r2_cr:.4f}")
        return True
    else:
        print(f"Experiment failed. R² (CR): {r2_cr:.4f}, R² (Ablation): {r2_ablation:.4f}")
        return False
        
        
def search_trial():
    trial_index = 1
    while True:
        data_path = gen_data_path("LCO", trial_index=trial_index)
        save_path = "/path/to/mae/batter_output/LCO_vs_abaltion"
        if run_experiment(data_path, save_path, batter="LCO", r2_threshold=0.2, version=trial_index):
            trial_index += 1
        else:
            continue
    
    

if __name__ == "__main__":
    search_trial()  