import torch
import torch.nn as nn
import numpy as np

import os
import yaml
import json
import importlib
from tqdm import tqdm
from typing import Type, Dict

from dataloader import build_dataloader
from model import *
from util import (
    load_mappings, 
    compute_class_weights,
    get_current_timestamp,
    causal_decision,
    f_score,
    edit_score,
    plot_temporal_results
)


device = "cuda" if torch.cuda.is_available() else "cpu"

def save_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer, epoch: int):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    checkpoint = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(checkpoint, path)

def load_checkpoint(path: str, model: torch.nn.Module, optimizer: torch.optim.Optimizer):
    checkpoint = torch.load(path, map_location=device)
    model.load_state_dict(checkpoint["model"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    return checkpoint["epoch"] + 1


def train(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    config: Dict,
    resume: str = None,
    verbose: bool = False,
):
    model.train()

    start_epoch = 1
    if resume:
        start_epoch = load_checkpoint(resume, model, optimizer)

    model_name = config["model"]["name"]
    epochs = config["train"]["epochs"]

    for epoch in tqdm(range(start_epoch, epochs + 1)):
        for x, y in dataloader:
            x, y = x.to(device), y.to(device)

            logits = model(x)                 # (B, w, C)
            last_logits = logits[:, -1, :]    # (B, C)
    
            loss = criterion(last_logits, y)
    
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        if verbose and epoch % 5 == 0:
            print(f"Epoch {epoch:3d}  Loss {loss.item():.4f}")

        if epoch % 5 == 0:
            save_dir = config["train"]["checkpoint_dir"]
            path = f"{save_dir}/{model_name}-{epoch}-{get_current_timestamp()}.ckpt"
            save_checkpoint(path, model, optimizer, epoch)

    save_dir = config["train"]["weight_dir"]
    if save_dir:
        os.makedirs(save_dir, exist_ok=True)
        model_path = f"{save_dir}/{model_name}-final-{get_current_timestamp()}.pt"
        config["model"]["model_path"] = model_path
        torch.save(model.state_dict(), model_path)

def eval(
    model: nn.Module, 
    dataloader: torch.utils.data.DataLoader, 
    config: Dict,
    visualize: bool = False,
):
    model.eval()

    video_preds = []
    video_gts = []

    for ds in tqdm(dataloader.dataset.datasets):
        T = ds.x.shape[0]
        window = ds.w
        preds = np.zeros(T, dtype=np.int64)

        # 补全前 window-1 帧
        idle_id = 0
        preds[:window-1] = idle_id

        # ===== 决策状态 =====
        pending = None     # 待确认状态
        stable = idle_id   # 已确认状态（连续帧）
        count = 0
        
        with torch.no_grad():
            for i in range(len(ds)):
                x, _ = ds[i]
                x = x.unsqueeze(0).to(device)

                logits = model(x)            # (1, w, C)
                last = logits[0, -1]

                pending, stable, count = causal_decision(last, pending, stable, count)
                preds[i + window - 1] = stable

        video_preds.append(preds)
        video_gts.append(ds.y.numpy())

    # ===== 合并所有视频 =====
    all_preds = np.concatenate(video_preds)
    all_gts   = np.concatenate(video_gts)

    mappings = load_mappings("data/Endo_Project/mapping0.txt")
    idx_to_action = {v : k for k, v in mappings.items()}
    pred_labels = [idx_to_action[p] for p in all_preds]
    gt_labels   = [idx_to_action[g] for g in all_gts]

    # Accuracy
    correct = sum(p == g for p, g in zip(all_preds, all_gts))
    acc = 100.0 * correct / len(all_preds)

    # Edit score
    edit = edit_score(pred_labels, gt_labels)

    # F1 scores
    overlap = [0.1, 0.25, 0.5]
    tp = np.zeros(3)
    fp = np.zeros(3)
    fn = np.zeros(3)

    for i, o in enumerate(overlap):
        tp1, fp1, fn1 = f_score(pred_labels, gt_labels, o)
        tp[i] += tp1
        fp[i] += fp1
        fn[i] += fn1

    f1_scores = {}
    for i, o in enumerate(overlap):
        precision = tp[i] / (tp[i] + fp[i]) if (tp[i] + fp[i]) > 0 else 0.0
        recall    = tp[i] / (tp[i] + fn[i]) if (tp[i] + fn[i]) > 0 else 0.0
        f1 = 2.0 * precision * recall / (precision + recall + 1e-8)
        f1_scores[o] = f1 * 100

    model_name = config["model"]["name"]
    result = {
        "timestamp": get_current_timestamp(),
        "model": config["model"],
        "num_params": sum(p.numel() for p in model.parameters()),
        "acc": round(acc, 2),
        "edit": round(edit, 2),
        "f1": {k: round(v, 2) for k, v in f1_scores.items()}
    }

    print(json.dumps(result, indent=4, ensure_ascii=False))

    output_dir = config["eval"]["result_dir"]
    if visualize and output_dir:
        os.makedirs(output_dir, exist_ok=True)
        plot_temporal_results(
            pred_labels, gt_labels,
            metadata=result,
            output_path=f"{output_dir}/{model_name}-{get_current_timestamp()}.png",
        )

    return result


def import_class(class_path: str) -> Type:
    """根据完整类路径导入并返回类对象。"""
    parts = class_path.rsplit(".", 1)
    if len(parts) != 2:
        raise ValueError(f"无效的类路径: {class_path}")
    module_path, class_name = parts
    try:
        module = importlib.import_module(module_path)
        return getattr(module, class_name)
    except ImportError as e:
        raise ImportError(f"无法导入模块 {module_path}: {e}")
    except AttributeError as e:
        raise AttributeError(f"模块 {module_path} 中不存在类 {class_name}: {e}")


def build_model(config: Dict) -> nn.Module:
    cls = import_class(config["class"])
    model: nn.Module = cls(**config["params"])
    model_path = config["model_path"]
    if model_path:
        model.load_state_dict(torch.load(model_path, map_location=device))
    model = model.to(device)
    return model

def build_optimizer(config: Dict, model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.Adam(
        model.parameters(),
        lr=float(config["lr"]),
        weight_decay=float(config["weight_decay"]),
    )

class TemporalLoss(nn.Module):
    def __init__(self, weight):
        super().__init__()
        self.ce = nn.CrossEntropyLoss(
            weight=torch.tensor(weight, dtype=torch.float32)
        ).to(device)

    def forward(self, logits, targets):
        return self.ce(logits, targets)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Causal Action Recognition")

    parser.add_argument("--mode", type=str, default="full", help="模式选择",
                        choices=["full", "train", "eval"])
    parser.add_argument("--config", type=str, help="模型训练配置文件")
    parser.add_argument("--resume", type=str, default=None, help="从指定检查点恢复训练")
    parser.add_argument("--verbose", action="store_true", help="是否打印训练损失")
    parser.add_argument("--visualize", action="store_true", help="是否生成可视化结果")

    args = parser.parse_args()

    # ===== Config loading  ======
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)

    model = build_model(config["model"])
    optimizer = build_optimizer(config["train"], model)

    # ===== Training =====
    if args.mode in ["full", "train"]:
        train_loader = build_dataloader(config["data"]["train"], "train")

        criterion = TemporalLoss(compute_class_weights(train_loader))

        train(model, train_loader, optimizer, criterion, 
              config=config, resume=args.resume, verbose=args.verbose)

    # ===== Evaluation =====
    if args.mode in ["full", "eval"]:
        test_loader = build_dataloader(config["data"]["test"], "test")

        eval(model, test_loader,
             config=config, visualize=True)
