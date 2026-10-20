import scipy.io as sio
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import re
import random
import json

random.seed(42)  # For reproducibility

def save_data(save_path, filename, tf):
    if os.path.exists(os.path.join(save_path, filename)):
        return
    else:
        sio.savemat(os.path.join(save_path, filename), tf)

def gen_test_files(files, cur_point):
    # Create dictionary to store files by type and state
    file_dict = {}
    for file_path in files:
        filename = os.path.basename(file_path)
        parts = filename.split('_wo_')[0]
        battery_name, state = parts.split('_', 1)
        group_key = (battery_name, state)
        if group_key not in file_dict:
            file_dict[group_key] = []
        file_dict[group_key].append(file_path)
        
    if cur_point % 3 == 0:
        test_files = []
        # 随机取一个lco#2 charge的文件
        lco_files = file_dict.get(('LCO#2', 'charge'), [])
        test_file = random.choice(lco_files)
        test_files.append(test_file)
        
        # 随机取一个 LCO#3 charge 的文件
        lco_files = file_dict.get(('LCO#3', 'charge'), [])
        test_file = random.choice(lco_files)
        test_files.append(test_file)
        
        # 随机取5个 LCO#4 charge 的文件
        lco_files = file_dict.get(('LCO#4', 'charge'), [])
        test_files.extend(random.sample(lco_files, 5))
        
        # 随机取1个 LCO#4 discharge 的文件
        lco_files = file_dict.get(('LCO#4', 'discharge'), [])
        test_file = random.choice(lco_files)
        test_files.append(test_file)
        return test_files
    
    elif cur_point % 3 == 1:
        test_files = []
        # 随机取2个 LCO#2 charge 的文件
        lco_files = file_dict.get(('LCO#2', 'charge'), [])
        test_files.extend(random.sample(lco_files, 2))
        
        # 随机取一个 LCO#3 charge 的文件
        lco_files = file_dict.get(('LCO#3', 'charge'), [])
        test_files.extend(random.sample(lco_files, 2))
        
        # 随机取5个 LCO#4 charge 的文件
        lco_files = file_dict.get(('LCO#4', 'charge'), [])
        test_file = random.choice(lco_files)
        test_files.append(test_file)
        
        # 随机取1个 LCO#4 discharge 的文件
        lco_files = file_dict.get(('LCO#4', 'discharge'), [])
        test_files.extend(random.sample(lco_files, 3))
        
        return test_files
    
    elif cur_point % 3 == 2:
        test_files = []
        # 随机取1个 LCO#2 charge 的文件
        lco_files = file_dict.get(('LCO#2', 'charge'), [])
        test_file = random.choice(lco_files)
        test_files.append(test_file)
        
        # 随机取2个 LCO#3 charge 的文件
        lco_files = file_dict.get(('LCO#3', 'charge'), [])
        test_files.extend(random.sample(lco_files, 2))
        
        # 随机取5个 LCO#4 charge 的文件
        lco_files = file_dict.get(('LCO#4', 'charge'), [])
        test_files.extend(random.sample(lco_files, 3))
        
        # 随机取1个 LCO#4 discharge 的文件
        lco_files = file_dict.get(('LCO#4', 'discharge'), [])
        test_files.extend(random.sample(lco_files, 2))
        
        return test_files



  
data_path = "/path/to/mae/data/fixed_data_final"
tot_files = glob.glob(os.path.join(data_path, "LCO*.mat"))

total_path = "/path/to/mae/data/only_lco_test"

# 随机选取8个LCO文件作为测试集，并且首先排除discharge_wo_10的数据
# Exclude files that contain 'discharge_wo_10' in their names
choice_files = [f for f in tot_files if 'discharge_wo_10' not in f]
# 生成10组索引，每一组8个，然后按照索引获取测试集
for i in range(51):
    # 随机选择8个文件作为测试集
    test_files = gen_test_files(choice_files, i)
    train_files = [f for f in tot_files if f not in test_files]
    train_files.sort()
    
    file_groups = {}
    for file_path in train_files:
        # Extract filename from path
        filename = os.path.basename(file_path)
        
        # Extract battery name and state
        # Pattern: {battery_name}_{state}_wo_{number}.mat
        parts = filename.split('_wo_')[0]
        battery_name, state = parts.split('_', 1)
        
        # Create key for grouping
        group_key = (battery_name, state)
        
        # Add file to group
        if group_key not in file_groups:
            file_groups[group_key] = []
        file_groups[group_key].append(file_path)
    #-----mixup data generation-----#    
    # 定义保存路径
    save_path = os.path.join(total_path, f"trial_data_{i+1}")
    os.makedirs(save_path, exist_ok=True)
    train_save_path = os.path.join(save_path, "train_data")
    os.makedirs(train_save_path, exist_ok=True)
    test_save_path = os.path.join(save_path, "test_data")
    os.makedirs(test_save_path, exist_ok=True)
    
    expand_factor = 1
    # Generate weights with finer granularity based on expand_factor
    step = 1 / (10 * expand_factor)  # smaller step size for higher granularity
    weight = []
    for j in range(1, 10 * expand_factor):  # more pairs based on expand_factor
        w1 = round(j * step, 3)  # round to 3 decimal places
        w2 = round(1 - w1, 3)
        weight.append((w1, w2))

    for (battery, state), files in file_groups.items():
        files = sorted(files)
        if len(files) < 2:
            print(i)
            print(f"Not enough files for {battery}_{state} to perform mixup.")
            
        for idx in range(len(files)-1):
            
            stage1 = int(re.search(r'wo_(\d+)', files[idx]).group(1))
            stage2 = int(re.search(r'wo_(\d+)', files[idx+1]).group(1))
            
            tf1 = sio.loadmat(files[idx])
            filename1 = os.path.basename(files[idx])
            save_data(train_save_path, filename1, tf1)
            data1= tf1['data']
            mah1 = tf1['ah'][0][0]
            
            tf2 = sio.loadmat(files[idx+1])
            filename2 = os.path.basename(files[idx+1])
            save_data(train_save_path, filename2, tf2)
            data2 = tf2['data']
            mah2 = tf2['ah'][0][0]
            for wt1, wt2 in weight:
                new_data = wt1 * data1 + wt2 * data2
                new_mah = wt1 * mah1 + wt2 * mah2
                new_tf = {'data': new_data, 'ah': np.array([[new_mah]])}
                new_filename = f"{battery}_{state}_wo_{stage1}_w{wt1}_{stage2}_w{wt2}.mat"
                save_data(train_save_path, new_filename, new_tf)
    # Save test data
    for file_path in test_files:
        tf = sio.loadmat(file_path)
        filename = os.path.basename(file_path)
        save_data(test_save_path, filename, tf)
        
multi_test_dir = os.listdir(total_path)
multi_test_dir.sort()

for i in range(len(multi_test_dir)):
    multi_test_dir[i] = os.path.join(total_path, multi_test_dir[i])
    train_files = glob.glob(os.path.join(multi_test_dir[i], "train_data", "*.mat"))
    test_files = glob.glob(os.path.join(multi_test_dir[i], "test_data", "*.mat"))

    lco_train_files = [f for f in train_files if "LCO" in f]
    lco_test_files = [f for f in test_files if "LCO" in f]
    nmc_train_files = [f for f in train_files if "NMC" in f]
    nmc_test_files = [f for f in test_files if "NMC" in f]

    split_data = {
        "LCO": {
            "train": lco_train_files,
            "test": lco_test_files
        },
        "NMC": {
            "train": nmc_train_files,
            "test": nmc_test_files
        }
    }

    with open(f'{total_path}/trial_{i+1}.json', 'w') as f:
        json.dump(split_data, f, indent=4)
    
    
    
