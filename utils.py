import re
import torch
from collections import Counter

def normalize(text):
    """去除冗余空白，合并多行并转为小写字符以执行公平指标比对"""
    return re.sub(r"\s+", " ", text).strip().lower()

def normalized_edit_distance(pred, gt):
    """计算归一化编辑距离 (Normalized Edit Distance)"""
    if not gt and not pred: return 1.0
    if not gt or not pred:  return 0.0
    m, n = len(gt), len(pred)
    if m > n:
        gt, pred, m, n = pred, gt, n, m
    dp = list(range(n + 1))
    for i in range(1, m + 1):
        prev = dp[0]
        dp[0] = i
        for j in range(1, n + 1):
            temp = dp[j]
            dp[j] = min(prev + (gt[i-1] != pred[j-1]),
                        dp[j] + 1, dp[j-1] + 1)
            prev = temp
    return 1.0 - dp[n] / max(m, n)

def char_f1(pred, gt):
    """计算字符维度的 F1-score 指标"""
    pc, gc = Counter(pred), Counter(gt)
    tp = sum((pc & gc).values())
    if sum(pc.values()) == 0 or sum(gc.values()) == 0:
        return 0.0
    p = tp / sum(pc.values())
    r = tp / sum(gc.values())
    return 2 * p * r / (p + r) if p + r else 0.0

def check_gpu():
    """检查系统的显卡支持及显存情况"""
    if torch.cuda.is_available():
        print(f"✅ GPU 验证就绪: {torch.cuda.get_device_name(0)} "
              f"({torch.cuda.get_device_properties(0).total_memory/1024**3:.1f} GB)")
        return True
    else:
        print("⚠️ 警告: 系统未能检测到活动的 CUDA 显卡，此微调流程可能受阻。")
        return False
