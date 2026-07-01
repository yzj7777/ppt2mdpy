为了全面、稳妥地解决您在 Kaggle 运行中遇到的 `FileNotFoundError` 路径报错问题，并确保系统优先、正确地读取和解析您在右侧添加的 Kaggle 本地数据集（`pptx_files`），以下为您整理了**4 个核心文件**的完整修改建议。

这些修改建立了“**本地 PPT 优先扫描**”与“**运行期动态路径还原**”的双重保障机制，能够适应多变的工作路径（如 `%cd ppt2mdpy` 切换）。

---

### 1. 修改 `config.py`
**修改目的**：引入环境自适应。无论工作目录如何切换，在 Kaggle 容器中统一将基础工作空间（`BASE_DIR`）锁定为绝对路径，避免由于 `./workspace` 的相对路径改变导致文件读写混淆。

请替换 `config.py` 的相应位置：

```python
# ==================== 基础目录自适应配置 ====================
# 限制 CUDA 显存碎片，提高显存复用效率，防止碎片化引起的 OOM
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# 自动识别 Kaggle 环境，统一解析为绝对路径
IS_KAGGLE = os.path.exists("/kaggle/working")
if IS_KAGGLE:
    # 强制将 Kaggle 下的根文件夹指到绝对路径，避免 git 克隆子目录切换导致相对路径失效
    DEFAULT_BASE_DIR = "/kaggle/working/workspace"
else:
    DEFAULT_BASE_DIR = "./workspace"

BASE_DIR = os.path.abspath(os.getenv("WORKSPACE_DIR", DEFAULT_BASE_DIR))
PDF_DIR = os.path.join(BASE_DIR, "pdfs")
PNG_DIR = os.path.join(BASE_DIR, "slides_png")
GT_DIR  = os.path.join(BASE_DIR, "ground_truth")
TRAIN_JSONL = os.path.join(BASE_DIR, "train.jsonl")
VAL_JSONL = os.path.join(BASE_DIR, "val.jsonl")
```

---

### 2. 修改 `dataset_prep.py`
**修改目的**：使本地扫描优先级高于云端下载。系统将首先在 Kaggle 挂载的 `/kaggle/input` 下递归扫描所有的原始 `.pptx` 数据，一旦发现直接进入转换解析，只有在本地没有任何 PPTX 时才退回云端备份。

请替换 `execute_data_pipeline` 函数：

```python
def execute_data_pipeline(local_input_dir=None):
    """数据准备的主控调度函数（优先支持 Kaggle 本地 PPTX 数据集解析）"""
    prepare_directories()

    # 1. 优先扫描 Kaggle 挂载目录（或用户指定目录），递归搜寻 PPTX 原始文件
    pptx_files = []
    search_dir = local_input_dir or "/kaggle/input"
    if os.path.exists(search_dir):
        for dirpath, _, filenames in os.walk(search_dir):
            for f in filenames:
                if f.endswith(".pptx"):
                    pptx_files.append(os.path.join(dirpath, f))

    # 2. 如果检测到本地有原始 PPTX，则优先执行本地解析转换，不下载云端旧备份
    if pptx_files:
        # 如果本地已经有转换好的结果，则直接放行，避免重复耗时转换
        if (os.path.exists(TRAIN_JSONL) and 
            os.path.exists(VAL_JSONL) and 
            os.path.exists(PNG_DIR) and 
            len(os.listdir(PNG_DIR)) > 0):
            print("✅ 本地已检测到现成切分数据集（JSONL与PNG），跳过解析转换。")
            return

        print(f"🚀 成功检测到本地 PPTX 原始文件共 {len(pptx_files)} 个。开始启动图文原生抽取与渲染流程...")
        extract_and_align_local_pptx(pptx_files)
        return

    # 3. 只有在本地检测不到 PPTX 时，才降级尝试拉取云端旧数据集作为兜底
    print("⚠️ 本地未检索到 PPTX 原始文件。正在尝试从云端拉取备份数据...")
    if (os.path.exists(TRAIN_JSONL) and 
        os.path.exists(VAL_JSONL) and 
        os.path.exists(PNG_DIR) and 
        len(os.listdir(PNG_DIR)) > 0):
        print("✅ 本地已检测到拉取好的云端备份数据集（JSONL与PNG），跳过下载。")
        return

    if retrieve_cloud_dataset():
        return

    raise FileNotFoundError("❌ 无法准备数据集：本地未扫描到 PPTX 原始文件，且云端没有可用备份。")
```

---

### 3. 修改 `dataset_loader.py`
**修改目的**：规避 JSONL 描述文件中写死的旧绝对路径。在数据加载器执行 `__getitem__` 时，通过 `os.path.basename` 仅截取文件名，并动态将其拼接至当前运行环境配置的 `PNG_DIR` 中。

请替换 `__getitem__` 方法：

```python
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # 【动态重构路径】提取文件名并结合当前配置的绝对 PNG_DIR 路径，防止路径写死冲突
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

### 4. 修改 `evaluate.py`
**修改目的**：对评测和验证指标阶段（`evaluate_model_pipeline`）应用相同的动态路径解算逻辑，彻底断绝因为克隆位置变动、下载历史数据路径残存而导致的 `FileNotFoundError` 报错。

请替换 `evaluate_model_pipeline` 函数：

```python
def evaluate_model_pipeline(eval_model, eval_processor, val_jsonl):
    if not os.path.exists(val_jsonl):
        print(f"❌ 评测验证集数据缺失：{val_jsonl}")
        return 0.0, 0.0, 0.0

    with open(val_jsonl, "r", encoding="utf-8") as f:
        val_items = [json.loads(line) for line in f]

    if not val_items:
        print("⚠️ 验证集中没有有效条目。")
        return 0.0, 0.0, 0.0

    total_ned, total_f1, total_lat = 0, 0, 0
    eval_model.eval()

    print(f"📐 针对 {len(val_items)} 个样本执行推理指标分析...")
    from config import PNG_DIR  # 动态引入
    for idx, item in enumerate(val_items):
        # 【动态路径解析】提取文件名并与本地 PNG_DIR 重组
        filename = os.path.basename(item["image"])
        actual_image_path = os.path.join(PNG_DIR, filename)
        
        image = Image.open(actual_image_path).convert("RGB")
        gt_text = normalize(item["text"])

        start = time.time()
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image},
            {"type": "text", "text": "Extract all content from this slide into markdown."}
        ]}]
        text = eval_processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        inputs = eval_processor(text=[text], images=[image], return_tensors="pt").to(eval_model.device)

        with torch.no_grad():
            output_ids = eval_model.generate(**inputs, max_new_tokens=512, do_sample=False)

        pred_text = eval_processor.decode(output_ids[0][inputs["input_ids"].shape[1]:], skip_special_tokens=True)
        lat = time.time() - start

        pred_norm = normalize(pred_text)
        ned = normalized_edit_distance(pred_norm, gt_text)
        f1 = char_f1(pred_norm, gt_text)

        total_ned += ned
        total_f1 += f1
        total_lat += lat
        print(f"  - [{idx+1}/{len(val_items)}] NED: {ned:.4f} | F1: {f1:.4f} | 时延: {lat:.2f}s")

    n = len(val_items)
    return total_ned / n, total_f1 / n, total_lat / n
```