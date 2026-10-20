import os
os.environ["CUDA_VISIBLE_DEVICES"] = "7"
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

# 绘图函数
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

# 对比损失函数
def contrastive_loss(proj, labels, alpha=10.0):
    sim = F.cosine_similarity(proj.unsqueeze(1), proj.unsqueeze(0), dim=-1)
    label_diff = torch.abs(labels.unsqueeze(1) - labels.unsqueeze(0))
    weights = torch.exp(-alpha * label_diff)
    mask = torch.eye(len(labels), device=labels.device).bool()
    sim = sim.masked_fill(mask, 0.0)
    weights = weights.masked_fill(mask, 0.0)
    loss = (1 - sim) * weights
    return loss.sum() / (weights.sum() + 1e-6)

# 加载预训练ViT
def prepapre_model(ckpt):
    checkpoint = torch.load(ckpt, map_location='cuda')
    vit_model = vit(img_size=128, in_chans=6)
    vit_model.load_state_dict(checkpoint['model'], strict=False)
    return vit_model.cuda() if torch.cuda.is_available() else vit_model.cpu()

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

# 修改后的 MLP 模型（带投影头）
class MLPWithProj(nn.Module):
    def __init__(self, input_dim, output_dim, proj_dim=128):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 512)
        self.reg_head = nn.Linear(512, output_dim)
        self.proj_head = nn.Linear(512, proj_dim)
        self.relu = nn.ReLU()
        self.drop = nn.Dropout(0.2)
        
    def forward(self, x):
        x = self.drop(self.relu(self.fc1(x)))
        reg_out = self.reg_head(x)
        proj_out = F.normalize(self.proj_head(x), dim=-1)
        return reg_out, proj_out

# 主训练函数
def fine(data_path, version=1, batter="NMC", weight_path=None, save_path=None):
    encoder = prepapre_model(ckpt=weight_path)
    for p in encoder.parameters():
        p.requires_grad = False
    regressor = MLPWithProj(input_dim=encoder.embed_dim, output_dim=1).cuda()
    loader_train, loader_val = prepare_data(data_path, batter)

    optimizer = torch.optim.AdamW(regressor.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    criterion = nn.MSELoss()
    
    num_epochs, lam = 30, 0.1
    
    best_r2 = -float('inf')
    for epoch in range(num_epochs):
        regressor.train()
        train_preds, train_labels, total_loss = [], [], 0
        for images, labels in loader_train:
            images = images.cuda().float()
            labels = labels.cuda().float()
            with torch.no_grad():
                feats = encoder(images)
            preds, proj = regressor(feats)
            loss_reg = criterion(preds, labels)
            loss_contrast = contrastive_loss(proj, labels)
            loss = loss_reg + lam * loss_contrast
            optimizer.zero_grad(); loss.backward(); optimizer.step()
            train_preds.extend(preds.detach().cpu().numpy())
            train_labels.extend(labels.detach().cpu().numpy())
            total_loss += loss.item()
        print(f"Epoch {epoch+1}/{num_epochs}, Train Loss: {total_loss / len(loader_train):.4f}")
        scheduler.step()

        # 验证
        regressor.eval()
        val_preds, val_labels, val_loss = [], [], 0
        with torch.no_grad():
            for images, labels in loader_val:
                images = images.cuda().float()
                labels = labels.cuda().float()
                feats = encoder(images)
                preds, _ = regressor(feats)
                loss = criterion(preds, labels)
                val_preds.extend(preds.cpu().numpy())
                val_labels.extend(labels.cpu().numpy())
                val_loss += loss.item()
        print(f"Epoch {epoch+1}, Val Loss: {val_loss / len(loader_val):.4f}")
        if r2_score(val_labels, val_preds) > best_r2:
            best_r2 = r2_score(val_labels, val_preds)
            best_preds, best_labels = val_preds, val_labels

    out_dir = f"{save_path}/{batter}_trial_{version}"
    os.makedirs(out_dir, exist_ok=True)
    train_labels = np.array(train_labels)
    train_preds = np.array(train_preds)
    best_labels = np.array(best_labels)
    best_preds = np.array(best_preds)
    
    draw_plot(train_labels, train_preds, result=os.path.join(out_dir, "train.jpg"))
    draw_plot(best_labels, best_preds, result=os.path.join(out_dir, "val.jpg"))
    # np.save(os.path.join(out_dir, "train_labels.npy"), train_labels)
    # np.save(os.path.join(out_dir, "train_preds.npy"), train_preds)
    np.save(os.path.join(out_dir, "val_labels.npy"), best_labels)
    np.save(os.path.join(out_dir, "val_preds.npy"), best_preds)
    
    return r2_score(best_labels, best_preds), np.mean(np.abs(best_labels - best_preds))

def multi_test(weight_path, battery="NMC", target_path=None, save_path=r'batter_output\tmp'):
    json_files = glob.glob(os.path.join(target_path, "*.json"))
    json_files.sort()
    for idx, trial_path in enumerate(json_files):
        print(f"Processing {trial_path}...")
        version = os.path.basename(trial_path).split('.')[0]
        fine(data_path=trial_path, version=version, batter=battery, weight_path=weight_path, save_path=save_path)
        print(f"Finished processing {trial_path}.\n")

def single_test(weight_path, battery="NMC", trial_path=None, save_path=r'batter_output\tmp'):
    
    print(f"Processing {trial_path}...")
    version = os.path.basename(trial_path).split('.')[0]
    r2_s = fine(data_path=trial_path, version=version, batter=battery, weight_path=weight_path, save_path=save_path)
    print(f"Finished processing {trial_path}.\n")
    print(f"R2 score for {version}: {r2_s:.4f}")
        
def mixup_test(weight_path, battery="NMC", target_path=None, save_path=r'batter_output\tmp'):
    json_files = glob.glob(os.path.join(target_path, "*.json"))
    json_files.sort()
    data = []
    for idx, trial_path in enumerate(json_files):
        print(f"Processing {trial_path}...")
        version = os.path.basename(trial_path).split('.')[0]
        r2 = fine(data_path=trial_path, version=version, batter=battery, weight_path=weight_path, save_path=save_path)
        print(f"Finished processing {trial_path}.\n")
        data.append({"version": version, "r2": r2})
    # Calculate mean R2 score
    # mean_r2 = np.mean([d['r2'] for d in data])
    # Add mean to the DataFrame
    df = pd.DataFrame(data)
    # df.loc[len(df)] = {'version': 'mean', 'r2': mean_r2}
    # Save to CSV
    df.to_csv(os.path.join(save_path, f"{battery}_mixup_results.csv"), index=False)

def diff_num_test(weight_path="/path/to/mae/mae_weight_out/mae_train_all/checkpoint-final.pth", 
                  battery="LCO", 
                  target_path=None, 
                  save_path=r'batter_output\tmp'):
    json_files = glob.glob(os.path.join(target_path, "*.json"))
    json_files.sort()
    data = []
    for idx, trial_path in enumerate(json_files):
        print(f"Processing {trial_path}...")
        version = os.path.basename(trial_path).split('.')[0]
        r2, mae = fine(data_path=trial_path, version=version, batter=battery, weight_path=weight_path, save_path=save_path)
        print(f"Finished processing {trial_path}.\n")
        data.append({"version": version, "r2": r2, "mae": mae})
    # Calculate mean R2 score
    mean_r2 = np.mean([d['r2'] for d in data])
    mean_mae = np.mean([d['mae'] for d in data])
    # Add mean to the DataFrame
    df = pd.DataFrame(data)
    df.loc[len(df)] = {'version': 'mean', 'r2': mean_r2}
    df.loc[len(df)+1] = {'version': 'mean', 'mae': mean_mae}
    # Save to CSV
    df.to_csv(os.path.join(save_path, f"{battery}_diff_num_results.csv"), index=False)
    
    
    
if __name__ == "__main__":
    battery = 'NMC'
    diff_num = 15
    target_path = f"/path/to/mae/data/diff_num_train_data/regression_data/{battery}_diff_{diff_num}"
    # target_path = "/path/to/mae/data/LCO_test_vs_ablation_mixuptest"
    save_path = f'batter_output/diff_num_test/{battery}_{diff_num}'
    os.makedirs(save_path, exist_ok=True)
    
    diff_num_test( battery=battery, target_path=target_path, save_path=save_path)
    
    
    # num_mixup = 5
    # weight_path = f'/path/to/mae/mae_weight_out/mix_{num_mixup}/checkpoint-399.pth'
    # json_path = f'/path/to/mae/data/mixup_test/{num_mixup}'
    # save_path = f'batter_output/mixup_test/{battery}_{num_mixup}'
    # mixup_test(weight_path=weight_path, battery=battery, 
    #          target_path=json_path, save_path=save_path)
    
    # json_path = "/path/to/mae/data/mixup_test/5/trial_8.json"
    # save_path = f'batter_output/mixup_test/{battery}_{num_mixup}_single'
    # single_test(weight_path=weight_path, battery=battery, 
    #            trial_path=json_path, save_path=save_path)
    
    # weight_path = '/path/to/mae/mae_weight_out/lco_unknown/checkpoint-final.pth'
    # json_path = '/path/to/mae/data/lco_unknown/LCO_unknown_#0.json'
    # save_path = f'batter_output/lco_unknown/test_on_LCO#0_v4'
    # single_test(weight_path=weight_path, battery=battery, 
    #            trial_path=json_path, save_path=save_path)
    