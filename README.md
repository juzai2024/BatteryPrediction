# mae_battery

从训练服务器上的 mae 项目整理出的实际使用代码（2026-10-20）。原始镜像见 ../mae_server/。
数据和权重不在这里：预训练数据 data/mae_train_all、权重 mae_weight_out/mae_train_all/checkpoint-final.pth 都在服务器上。
代码和 json 里的绝对路径统一写成占位符 `/path/to/mae`，使用前替换成自己的项目根目录。

| 目录 | 内容 | 运行方式 |
|---|---|---|
| pretrain/ | MAE 预训练：batter_mae.py + model_batter.py（ViT-L/16，128×128×6，mask 0.75，增强仅 ToTensor+RandomHorizontalFlip）；logs/ 为各次预训练的 log | `cd pretrain && python batter_mae.py --data_path <mat目录> --output_dir <输出>` |
| regression/ | 冻结 encoder + MLPWithProj 回归：batter_fine_net_CR.py（主脚本，含 multi/mixup/diff_num 批量实验）、_unknown（未知电池）、_grad_cam、ablation_exp.py（CNN/RF/LR/GPR/SVR 对比）、battery_cr_net_ml.py（trial 搜索） | `cd regression && python batter_fine_net_CR.py` |
| data_gen/ | mixup 数据生成、train/test 划分 json（splits/ 对应服务器 data/ 下各实验目录） | — |
| vis/ | ViT-GradCAM（实际用 attention rollout） | — |
| release_tools/ | 发布包 battery_public_export 的导出/训练头/评估脚本与 predict.py、reports/ | 发布包本体（fp16 encoder+头，592M）在服务器 battery_public_export/final_package |

pretrain/ 和 regression/ 各自带一份 util/、data/batter.py，保证在各自目录下 import 可用。
