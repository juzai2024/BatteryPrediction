import json
import os
import glob

train_data_path = "/path/to/mae/data/lco_unknown/mae_train"
test_path = "/path/to/mae/data/Batter/LCO#0charge"

train_files = glob.glob(os.path.join(train_data_path, "*.mat"))
test_files = glob.glob(os.path.join(test_path, "**", "*.mat"), recursive=True)

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

with open(f'data/lco_unknown/LCO_unknown_#0.json', 'w') as f:
    json.dump(split_data, f, indent=4)
    
    