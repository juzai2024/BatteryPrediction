import torch
from torch.utils.data import Dataset
import numpy as np
import os
import scipy.io as sio
import glob
import json

 
# class BatterDataset(Dataset):
#     # 添加类变量用于存储每个数据路径对应的已打乱列表
#     _shuffled_lists = {}
    
#     def __init__(self, data_path, transform=None, is_train=True, batter=None, 
#                  batter_s=None , pattern=None, split_ratio=0.9):
#         """
#         Initialize the dataset
#         Args:
#             data_path (str): Path to the data
#             transform (callable, optional): Optional transform to be applied to samples
#             batter (str, optional): Battery type to filter the dataset \\ LCO, NMC 
#             batter_s (str, optional): Battery specific  \\ LCO#1, NMC#4
#             pattern (str, optional): exclude or inculed
            
#         """
#         self.data_path = data_path
#         self.transform = transform
#         data_path_list = os.listdir(data_path)
#         target_list = [os.path.join(data_path, i) for i in data_path_list]
        
#         if batter is not None:
#             target_list = [i for i in target_list if batter in i]
        
#         # 创建唯一键，用于在类变量中存储对应的打乱列表
#         key = f"{data_path}_{batter}"
        
#         # 如果这个键对应的打乱列表不存在，则创建它
#         if key not in BatterDataset._shuffled_lists:
#             fixed_shuffled_list = target_list.copy()
#             np.random.shuffle(fixed_shuffled_list)
#             BatterDataset._shuffled_lists[key] = fixed_shuffled_list
        
#         # 使用已存储的打乱列表
#         fixed_shuffled_list = BatterDataset._shuffled_lists[key]
        
#         if batter_s is not None:
#             if pattern == "exclude":
#                 # Exclude the specific battery
#                 self.target_list = [i for i in fixed_shuffled_list if batter_s not in i]
                
#             elif pattern == "include":
#                 self.target_list = [i for i in fixed_shuffled_list if batter_s in i] 
            
#         else:
#             if is_train:
#                 # Use 90% of data for training
#                 split_idx = int(split_ratio * len(fixed_shuffled_list))
#                 self.target_list = fixed_shuffled_list[:split_idx]
#             else:
#                 # Use 10% of data for testing/validation
#                 split_idx = int(split_ratio * len(fixed_shuffled_list))
#                 self.target_list = fixed_shuffled_list[split_idx:]

#     def __len__(self):
#         """
#         Returns the total number of samples
#         """
#         return len(self.target_list)

#     def __getitem__(self, idx):
#         """
#         Returns one sample of data
#         Args:
#             idx (int): index of the sample to return
#         """
            
#         data = sio.loadmat(self.target_list[idx])
#         gt = data['data']
#         ah = data['ah'].reshape(-1)
#         if self.transform:
#             gt = self.transform(gt)
#         return gt, ah


class BatterDataset(Dataset):
    
    def __init__(self, data_path, transform=None, is_train=True, batter=None,):
        """
        Initialize the dataset
        Args:
            data_path (str): Path to the data
            transform (callable, optional): Optional transform to be applied to samples
            batter (str, optional): Battery type to filter the dataset \\ LCO, NMC 
            batter_s (str, optional): Battery specific  \\ LCO#1, NMC#4
            pattern (str, optional): exclude or inculed
            
        """
        if data_path.endswith('json'):
            json_data = json.load(open(data_path, 'r'))[batter]
            if is_train:
                self.target_list = json_data['train']
            else:
                self.target_list = json_data['test']
                   
        else:
            self.data_path = data_path
            self.target_list = glob.glob(os.path.join(data_path,'**','*.mat'), recursive=True)
        
        self.transform = transform

    def __len__(self):
        """
        Returns the total number of samples
        """
        return len(self.target_list)

    def __getitem__(self, idx):
        """
        Returns one sample of data
        Args:
            idx (int): index of the sample to return
        """
            
        data = sio.loadmat(self.target_list[idx])
        gt = data['data']
        ah = data['ah'].reshape(-1)
        if self.transform:
            gt = self.transform(gt)
        return gt, ah
    

if __name__ == "__main__":
    # Example usage
    # data_path = "/path/to/mae/data/mixup_data"
    data_path = "/path/to/mae/data/trial_0.json"
    dataset_val = BatterDataset(
            data_path=data_path,
            is_train=False,     
            
        )
    
    for data, ah in dataset_val:
        print(data.shape, ah)
        
    