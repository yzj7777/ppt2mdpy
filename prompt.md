针对 Kaggle 运行环境，最稳妥且推荐的修改方案是**“环境自适应”与“动态路径重建”的双重保障设计**。

这样修改不仅能自动适配 Kaggle 默认的 `/kaggle/working` 目录，还能防止由于云端拉取的数据集内硬编码绝对路径导致的路径不一致问题。

以下是具体的修改建议，请直接替换对应文件中的相关代码段：

---

### 1. 修改 `config.py`（环境自动识别）

**修改目的**：自动检测是否在 Kaggle 环境中运行。如果是，默认将工作根目录指向 `/kaggle/working`（与云端数据集中的路径保持一致）；如果不是（如在本地或 AutoDL），则退回 `./workspace`。

**修改位置**：定位到 `config.py` 中定义 `BASE_DIR` 的地方，修改为：

```python
# ==================== 基础目录自适应配置 ====================
# 自动检测是否处于 Kaggle 环境中
IS_KAGGLE = os.path.exists("/kaggle/working")
DEFAULT_BASE_DIR = "/kaggle/working" if IS_KAGGLE else "./workspace"

BASE_DIR = os.getenv("WORKSPACE_DIR", DEFAULT_BASE_DIR)
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
PNG_DIR = os.path.join(BASE_DIR, "slides_png")
GT_DIR  = os.path.join(BASE_DIR, "ground_truth")
TRAIN_JSONL = os.path.join(BASE_DIR, "train.jsonl")
VAL_JSONL = os.path.join(BASE_DIR, "val.jsonl")
```

---

### 2. 修改 `dataset_loader.py`（训练集路径安全解析）

**修改目的**：在加载训练和验证数据时，不直接读取 JSONL 文件中可能写死的旧绝对路径，而是动态提取文件名（basename），并拼装到当前配置的 `PNG_DIR` 下。这可以较好地兼容不同平台之间的路径差异。

**修改位置**：定位到 `SlideDataset` 类的 `__getitem__` 方法，修改为：

```python
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 动态修复可能由环境不同导致的图片路径不一致问题
        from config import PNG_DIR
        filename = os.path.basename(sample["image"])
        actual_image_path = os.path.join(PNG_DIR, filename)
        
        image = Image.open(actual_image_path).convert("RGB")
        return [
            {
                "role": "user",
                "content": [
                    {"type": "image", "image": image},
                    {"type": "text", "text": "Extract all content from this slide into markdown."}
                ]
            },
            {
                "role": "assistant",
                "content": [
                    {"type": "text", "text": sample["text"]}
                ]
            }
        ]
```

---

### 3. 修改 `evaluate.py`（评测流程路径安全解析）

**修改目的**：在运行双模型指标对比评估（Baseline vs 微调模型）时，采用同样的机制动态解析验证集的图片路径，避免此处抛出 `FileNotFoundError` 导致整个 pipeline 中断。

**修改位置**：定位到 `evaluate_model_pipeline` 函数，将读取图片的循环部分修改为：

```python
    total_ned, total_f1, total_lat = 0, 0, 0
    eval_model.eval()

    print(f"📐 针对 {len(val_items)} 个样本执行推理指标分析...")
    from config import PNG_DIR  # 引入当前配置的图片目录
    for idx, item in enumerate(val_items):
        # 提取文件名并重构本地真实路径
        filename = os.path.basename(item["image"])
        actual_image_path = os.path.join(PNG_DIR, filename)
        
        image = Image.open(actual_image_path).convert("RGB")
        gt_text = normalize(item["text"])

        start = time.time()
        # 后续推理逻辑保持不变...
```

---

### 为什么推荐这个方案？

1. **零人工干预**：Kaggle 笔记本启动后直接运行 `python main.py`，无需手动在终端或代码里声明 `export WORKSPACE_DIR=/kaggle/working`，代码能够自行识别并完成目录对齐。
2. **容错性强**：即使未来云端备份的数据集转移到其他地方，或者在本地离线运行时解压到了不同的子目录，`os.path.basename` 的动态拼接机制都能确保程序始终在当前配置的 `slides_png` 文件夹中寻找图片，系统整体表现会更加健壮。
