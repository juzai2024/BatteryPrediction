import glob 
import os
import json
import numpy as np
import re


def load_json(json_path):
    with open(json_path, 'r') as f:
        data = json.load(f)
    return data

def save_json(data, json_path):
    with open(json_path, 'w') as f:
        json.dump(data, f, indent=4)
        
def get_json_files(directory):
    return glob.glob(os.path.join(directory, '*.json'))

def selction_mixup(json_data, expand_factor=5):
    """
    Selects a mixup in JSON data entries based on the specified ratio.
    The JSON file is finished with a 'mixup' key.
    """
    result = {
        'train': [],
        'test' : [],
    }
    step = 1 / expand_factor
    weight = []
    for i in range(1, expand_factor):  
        w1 = round(i * step, 3) 
        w2 = round(1 - w1, 3)
        weight.append((w1, w2))
    weight = np.array(weight)
    
    for data in json_data['train']:
        try:
            weight1, weight2 = re.findall(r'_w(\d*\.?\d+)', data)
            weight1 = float(weight1)
            weight2 = float(weight2)
            # 元组可以直接和np数组比较
            weight_cur = (weight1, weight2)
            if weight_cur in weight:
                result['train'].append(data)
            
        except ValueError:
            result['train'].append(data)
            continue
    result['test'] = json_data['test']
    
    return result
    

if __name__ == "__main__":
    np.random.seed(42)  
    save_path = "/path/to/mae/data/mixup_test"
    expand_factor = 5
    save_path = os.path.join(save_path, str(expand_factor))
    os.makedirs(save_path, exist_ok=True)
    lco_js_dir = '/path/to/mae/data/LCO_test_vs_ablation_mixuptest'
    nmc_js_dir = '/path/to/mae/data/final_test/NMC_json'
    lco_json_files = get_json_files(lco_js_dir)
    lco_json_files.sort(key=lambda x: int(re.findall(r'trial_(\d+)\.json', x)[0]))
    lco_json_files = lco_json_files[:31]
    print(lco_json_files)
    nmc_json_files = get_json_files(nmc_js_dir)
    nmc_json_files.sort()
    for i in range(len(lco_json_files)):
        lco_res = load_json(lco_json_files[i])['LCO']
        nmc_res = load_json(nmc_json_files[i])['NMC']
        lco_res = selction_mixup(lco_res, expand_factor=expand_factor)
        nmc_res = selction_mixup(nmc_res, expand_factor=expand_factor)
        tol_res = {
            'LCO': lco_res,
            'NMC': nmc_res
        }
        # save_json(tol_res, os.path.join(save_path,f'trial_{i}.json'))
    
    
  
    
   