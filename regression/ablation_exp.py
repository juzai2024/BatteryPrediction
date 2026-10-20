import os
os.environ["CUDA_VISIBLE_DEVICES"] = "4"
import torch
from data.batter import BatterDataset
from models_vit import vit_batter_patch16 as vit
from util import misc
from torchvision import transforms
from sklearn.metrics import r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from torch import nn
import torch.nn.functional as F
import glob


# 数据集
def prepare_data(data_path, batter):
    transform = transforms.ToTensor()
    train_set = BatterDataset(data_path=data_path, transform=transform, is_train=True, batter=batter)
    val_set = BatterDataset(data_path=data_path, transform=transform, is_train=False, batter=batter)
    num_tasks = misc.get_world_size()
    global_rank = misc.get_rank()
    sampler_train = torch.utils.data.DistributedSampler(train_set, num_replicas=num_tasks, rank=global_rank, shuffle=True)
    loader_train = torch.utils.data.DataLoader(train_set, sampler=sampler_train, batch_size=5, num_workers=2, drop_last=True)
    loader_val = torch.utils.data.DataLoader(val_set, batch_size=3, shuffle=False, num_workers=1)
    return loader_train, loader_val

def draw_plot(x, y, x_label="True Values", y_label="Predictions", result="picture/tt.jpg", title=None):
    if not os.path.exists(os.path.dirname(result)):
        os.makedirs(os.path.dirname(result))
    plt.figure(figsize=(12, 6))
    plt.plot(x, label=x_label, color='#1976D2', linewidth=2) 
    # 使用seismic色彩映射中偏红一侧更红的颜色作为y的颜色
    plt.plot(y, label=y_label, color='#d7626b', linestyle='--', linewidth=2)
    plt.xlabel('Sample Index')
    plt.ylabel('Value')
    plt.legend(loc='upper left')
    r2 = r2_score(x, y)
    plt.text(0.95, 0.95, f'R² = {r2:.4f}', transform=plt.gca().transAxes,
             horizontalalignment='right', verticalalignment='top')
    plt.savefig(result, dpi=300)
    mse = np.mean((x - y) ** 2)
    mae = np.mean(np.abs(x - y))
    print(f'MSE: {mse:.4f}, MAE: {mae:.4f}, R2: {r2:.4f}')

def cnn_train(loader_train, loader_val, save_path):
    print("Training CNN model...")
    # Define a simple CNN model for regression
    class CNNRegressor(nn.Module):
        def __init__(self):
            super(CNNRegressor, self).__init__()
            self.conv1 = nn.Conv2d(6, 32, kernel_size=3, padding=1)
            self.conv2 = nn.Conv2d(32, 16, kernel_size=3, padding=1)
            self.conv3 = nn.Conv2d(16, 8, kernel_size=3, padding=1)
            self.pool = nn.MaxPool2d(2)
            self.fc1 = nn.Linear(16 * 16 * 8, 128)
            self.fc2 = nn.Linear(128, 1)
            self.dropout = nn.Dropout(0.5)

        def forward(self, x):
            x = F.relu(self.conv1(x))
            x = self.pool(x)
            x = F.relu(self.conv2(x))
            x = self.pool(x)
            x = F.relu(self.conv3(x))
            x = self.pool(x)
            x = x.view(x.size(0), -1)
            x = F.relu(self.fc1(x))
            x = self.dropout(x)
            x = self.fc2(x)
            return x

    # Initialize model, optimizer and loss function
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CNNRegressor().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.001)

    # Training loop
    num_epochs = 100
    model.train()
    for epoch in range(num_epochs):
        running_loss = 0.0
        for batch_data, batch_labels in loader_train:
            batch_data = batch_data.to(device).float()
            batch_labels = batch_labels.to(device).float()
            
            optimizer.zero_grad()
            outputs = model(batch_data)
            loss = criterion(outputs, batch_labels) 
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()

        if (epoch + 1) % 10 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(loader_train):.4f}')

        # Validation
        if (epoch + 1) % 10 == 0:
            model.eval()
            val_outputs = []
            val_labels = []
            with torch.no_grad():
                for batch_data, batch_labels in loader_val:
                    batch_data = batch_data.to(device).float()
                    batch_labels = batch_labels.to(device).float()
                    outputs = model(batch_data)
                    val_outputs.extend(outputs.cpu().numpy())
                    val_labels.extend(batch_labels.cpu().numpy())
            
            val_r2 = r2_score(val_labels, val_outputs)
            print(f'Validation R2 score: {val_r2:.4f}')
            model.train()
    # Final evaluation
    model.eval()
    val_outputs = []
    val_labels = []
    with torch.no_grad():   
        for batch_data, batch_labels in loader_val:
            batch_data = batch_data.to(device).float()
            batch_labels = batch_labels.to(device).float()
            outputs = model(batch_data)
            val_outputs.extend(outputs.cpu().numpy())
            val_labels.extend(batch_labels.cpu().numpy())
    val_r2 = r2_score(val_labels, val_outputs)
    print(f'Final Validation R2 score: {val_r2:.4f}')
    draw_plot(np.array(val_labels).squeeze(), np.array(val_outputs).squeeze(),
              x_label="True Values", y_label="Predictions",
              result=f"{save_path}/cnn_regression_plot.jpg")
    np.save(os.path.join(save_path, 'cnn_val_out.npy'), np.array(val_outputs).squeeze())
    return val_r2
    
def rf_train(loader_train, loader_val, save_path):
    from sklearn.ensemble import RandomForestRegressor
    print("Training Random Forest model...")
    # Prepare training data
    train_features = []
    train_labels = []
    for batch_data, batch_labels in loader_train:
        batch_data = batch_data.view(batch_data.size(0), -1).numpy()
        train_features.extend(batch_data)
        train_labels.extend(batch_labels.numpy())
    train_features = np.array(train_features)
    train_labels = np.array(train_labels).squeeze()
    # Train Random Forest model
    rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
    rf_model.fit(train_features, train_labels)

    # Prepare validation data
    val_features = []
    val_labels = []
    for batch_data, batch_labels in loader_val:
        batch_data = batch_data.view(batch_data.size(0), -1).numpy()
        val_features.extend(batch_data)
        val_labels.extend(batch_labels.numpy())
    
    val_features = np.array(val_features)
    val_labels = np.array(val_labels).squeeze()
    # Predict and evaluate
    val_predictions = rf_model.predict(val_features)
    val_r2 = r2_score(val_labels, val_predictions)
    print(f'Random Forest Validation R2 score: {val_r2:.4f}')  
    draw_plot(val_labels, val_predictions,
              x_label="True Values", y_label="Predictions",
              result=f"{save_path}/rf_regression_plot.jpg") 
    np.save(os.path.join(save_path, 'rf_val_out.npy'), val_predictions)
    return val_r2
    
def lr_train(loader_train, loader_val, save_path):
    from sklearn.linear_model import LinearRegression
    print("Training Linear Regression model...")
    # Prepare training data
    train_features = []
    train_labels = []
    for batch_data, batch_labels in loader_train:
        batch_data = batch_data.view(batch_data.size(0), -1).numpy()
        train_features.extend(batch_data)
        train_labels.extend(batch_labels.numpy())
    train_features = np.array(train_features)
    train_labels = np.array(train_labels).squeeze()

    # Train Linear Regression model
    lr_model = LinearRegression()
    lr_model.fit(train_features, train_labels)

    # Prepare validation data
    val_features = []
    val_labels = []
    for batch_data, batch_labels in loader_val:
        batch_data = batch_data.view(batch_data.size(0), -1).numpy()
        val_features.extend(batch_data)
        val_labels.extend(batch_labels.numpy())
    
    val_features = np.array(val_features)
    val_labels = np.array(val_labels).squeeze()
    # Predict and evaluate
    val_predictions = lr_model.predict(val_features)
    val_r2 = r2_score(val_labels, val_predictions)
    print(f'Linear Regression Validation R2 score: {val_r2:.4f}')
    draw_plot(val_labels, val_predictions,
              x_label="True Values", y_label="Predictions",
              result=f"{save_path}/linear_regression_plot.jpg")
    np.save(os.path.join(save_path, 'lr_val_out.npy'), val_predictions)
    return val_r2
    
    
def gpr_train(loader_train, loader_val, save_path):
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel as C
    print("Training Gaussian Process Regression model...")
    # Prepare training data
    train_features = []
    train_labels = []
    for batch_data, batch_labels in loader_train:
        batch_data = batch_data.view(batch_data.size(0), -1).numpy()
        train_features.extend(batch_data)
        train_labels.extend(batch_labels.numpy())
    train_features = np.array(train_features)
    train_labels = np.array(train_labels).squeeze()

    # Define and train GPR model
    kernel = C(1.0) * RBF(length_scale=1.0)
    gpr_model = GaussianProcessRegressor(kernel=kernel, random_state=42)
    gpr_model.fit(train_features, train_labels)

    # Prepare validation data
    val_features = []
    val_labels = []
    for batch_data, batch_labels in loader_val:
        batch_data = batch_data.view(batch_data.size(0), -1).numpy()
        val_features.extend(batch_data)
        val_labels.extend(batch_labels.numpy())

    val_features = np.array(val_features)
    val_labels = np.array(val_labels).squeeze()

    # Predict and evaluate
    val_predictions = gpr_model.predict(val_features)
    val_r2 = r2_score(val_labels, val_predictions)
    print(f'Gaussian Process Regression Validation R2 score: {val_r2:.4f}')
    draw_plot(val_labels, val_predictions,
              x_label="True Values", y_label="Predictions",
              result=f"{save_path}/gpr_regression_plot.jpg")
    np.save(os.path.join(save_path, 'gpr_val_out.npy'), val_predictions)
    return val_r2

def svr_train(loader_train, loader_val, save_path):
    from sklearn.svm import SVR
    print("Training Support Vector Regression model...")
    # Prepare training data
    train_features = []
    train_labels = []
    for batch_data, batch_labels in loader_train:
        batch_data = batch_data.view(batch_data.size(0), -1).numpy()
        train_features.extend(batch_data)
        train_labels.extend(batch_labels.numpy())
    train_features = np.array(train_features)
    train_labels = np.array(train_labels).squeeze()

    # Train SVR model
    svr_model = SVR(kernel='rbf', C=1.0, epsilon=0.1)
    svr_model.fit(train_features, train_labels)

    # Prepare validation data
    val_features = []
    val_labels = []
    for batch_data, batch_labels in loader_val:
        batch_data = batch_data.view(batch_data.size(0), -1).numpy()
        val_features.extend(batch_data)
        val_labels.extend(batch_labels.numpy())

    val_features = np.array(val_features)
    val_labels = np.array(val_labels).squeeze()

    # Predict and evaluate
    val_predictions = svr_model.predict(val_features)
    val_r2 = r2_score(val_labels, val_predictions)
    print(f'Support Vector Regression Validation R2 score: {val_r2:.4f}')
    draw_plot(val_labels, val_predictions,
              x_label="True Values", y_label="Predictions",
              result=f"{save_path}/svr_regression_plot.jpg")
    np.save(os.path.join(save_path, 'svr_val_out.npy'), val_predictions)
    return val_r2
    
# 主训练函数
def fine(data_path, version=1, batter="NMC", weight_path=None, save_path=None):
    os.makedirs(save_path, exist_ok=True)
    save_path = os.path.join(save_path, f"{batter}_trial_{version}")
    loader_train, loader_val = prepare_data(data_path, batter)
    
    # Prepare validation data
    # val_features = []
    # val_labels = []
    # for batch_data, batch_labels in loader_val:
    #     batch_data = batch_data.view(batch_data.size(0), -1).numpy()
    #     val_features.extend(batch_data)
    #     val_labels.extend(batch_labels.numpy())
    # val_labels = np.array(val_labels).squeeze()
    # np.save(os.path.join(save_path, 'val_labels.npy'), val_labels)
    
    # --- train ----
    # cnn_r2 = cnn_train(loader_train, loader_val, save_path)
    lr_r2 = lr_train(loader_train, loader_val, save_path)
    # gpr_r2 = gpr_train(loader_train, loader_val, save_path)
    # svr_r2 = svr_train(loader_train, loader_val, save_path)
    # rf_r2 = rf_train(loader_train, loader_val, save_path)
    results = {
        # 'CNN_R2': cnn_r2,
        'LR_R2': lr_r2,
        # 'GPR_R2': gpr_r2,
        # 'SVR_R2': svr_r2,
        # 'RF_R2': rf_r2
    }
    return lr_r2


def _fine(data_path, batter="NMC", weight_path=None, save_path=None):
    os.makedirs(save_path, exist_ok=True)
    loader_train, loader_val = prepare_data(data_path, batter)
    
    # Prepare validation data
    # val_features = []
    # val_labels = []
    # for batch_data, batch_labels in loader_val:
    #     batch_data = batch_data.view(batch_data.size(0), -1).numpy()
    #     val_features.extend(batch_data)
    #     val_labels.extend(batch_labels.numpy())
    # val_labels = np.array(val_labels).squeeze()
    # np.save(os.path.join(save_path, 'val_labels.npy'), val_labels)
    
    # --- train ----
    cnn_r2 = cnn_train(loader_train, loader_val, save_path)
    # lr_r2 = lr_train(loader_train, loader_val, save_path)
    gpr_r2 = gpr_train(loader_train, loader_val, save_path)
    svr_r2 = svr_train(loader_train, loader_val, save_path)
    rf_r2 = rf_train(loader_train, loader_val, save_path)
    results = {
        'CNN_R2': cnn_r2,
        # 'LR_R2': lr_r2,
        'GPR_R2': gpr_r2,
        'SVR_R2': svr_r2,
        'RF_R2': rf_r2
    }
    return results

def run_lco():
    lco_trial_path = "/path/to/mae/data/LCO_test_vs_ablation"
    trials_path = glob.glob(os.path.join(lco_trial_path, "*.json"))[:30]
    for trial in trials_path:
        trial_name = os.path.basename(trial).split('.')[0]
        print(f"Processing {trial_name}...")
        _fine(data_path=trial, batter="LCO",
             save_path=f'batter_output/LCO_vs_abaltion/LCO_{trial_name}')
        
def run_single():
    # trial = '/path/to/mae/data/lco_unknown/LCO_unknown_#0.json'
    trial = '/path/to/mae/data/LCO_test_vs_ablation/trial_5.json'
    trial_name = os.path.basename(trial).split('.')[0]
    print(f"Processing {trial_name}...")
    _fine(data_path=trial, batter="LCO",
             save_path=f'batter_output/LCO_vs_abaltion/LCO_{trial_name}')
    
def run_nmc():
    nmc_trial_path = "/path/to/mae/data/final_test/NMC_json"
    trials_path = glob.glob(os.path.join(nmc_trial_path, "*.json"))
    for trial in trials_path:
        trial_name = os.path.basename(trial).split('.')[0]
        print(f"Processing {trial_name}...")
        fine(data_path=trial, version="v9", batter="NMC",
             save_path=f'batter_output/ablation_exp/NMC_{trial_name}')


        
if __name__ == "__main__":
    # run_lco()
    run_single()
    # run_nmc()