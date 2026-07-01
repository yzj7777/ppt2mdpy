import os
import re
import json
import random
import tarfile
from concurrent.futures import ProcessPoolExecutor, as_completed
from pptx import Presentation
from huggingface_hub import HfApi, hf_hub_download, list_repo_files

from config import (
    BASE_DIR, PDF_DIR, PNG_DIR, GT_DIR, TRAIN_JSONL, VAL_JSONL,
    DATASET_REPO, VERSION
)
from converter import (
    convert_single_pptx, pdf_to_png_high_speed, slide_to_markdown_layout_aware
)

def prepare_directories():
    for d in [PDF_DIR, PNG_DIR, GT_DIR]:
        os.makedirs(d, exist_ok=True)

def retrieve_cloud_dataset():
    """动态检索 HuggingFace 云端仓库并下载最新可用版本的数据压缩包"""
    print(f"🔍 正在检索云端仓库 {DATASET_REPO} 中的所有数据集版本...")
    try:
        files = list_repo_files(repo_id=DATASET_REPO, repo_type="dataset")
        # 筛选所有的 tar 包
        dataset_files = [f for f in files if f.startswith("ppt2md_dataset") and f.endswith(".tar.gz")]

        if dataset_files:
            def get_dataset_version_key(filename):
                match = re.search(r"ppt2md_dataset_(v\d+_\d+)\.tar\.gz", filename)
                return match.group(1) if match else "v00000000_000000"

            # 提取时间戳排序找出最新版
            dataset_files.sort(key=get_dataset_version_key)
            latest_dataset_file = dataset_files[-1]
            print(f"📌 云端最新发现的数据集文件为: {latest_dataset_file}")

            downloaded_tar = hf_hub_download(
                repo_id=DATASET_REPO,
                filename=latest_dataset_file,
                repo_type="dataset",
                local_dir=BASE_DIR,
                local_dir_use_symlinks=False
            )
            print(f"📥 成功下载数据集包: {downloaded_tar}")
            print("📦 正在执行解压...")
            with tarfile.open(downloaded_tar, "r:gz") as tar:
                tar.extractall(path=BASE_DIR)
            print("✅ 成功拉取云端数据集并完成部署。已跳过 PPT 原生抽取逻辑。")
            return True
        else:
            print("⚠️ 仓库内没有找到任何版本的数据集。将转向本地重构流程。")
            return False
    except Exception as e:
        print(f"⚠️ 云端备份数据集检索失败 (原因: {e})，系统将进入本地生成流程...")
        return False

def extract_and_align_local_pptx(pptx_files):
    """基于本地 PPT 原生文档并行抽离图文对，执行分割，并归档上传"""
    max_workers = min(2, os.cpu_count() or 1)
    print(f"🚀 启动并行 LibreOffice PDF 转换, 文件数: {len(pptx_files)}...")
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        futures = [executor.submit(convert_single_pptx, f, PDF_DIR) for f in pptx_files]
        for fut in as_completed(futures):
            path, success = fut.result()
            print(f"  - [{'Success' if success else 'Failed'}] {os.path.basename(path)}")

    pdf_files = sorted([os.path.join(PDF_DIR, f) for f in os.listdir(PDF_DIR) if f.endswith('.pdf')])

    print("🚀 渲染 PDF 为 PNG 图集 (PyMuPDF)...")
    png_paths = []
    with ProcessPoolExecutor(max_workers=os.cpu_count() or 1) as executor:
        futures = [executor.submit(pdf_to_png_high_speed, pdf, PNG_DIR) for pdf in pdf_files]
        for fut in as_completed(futures):
            png_paths.extend(fut.result())
    print(f"Generated {len(png_paths)} PNG images.")

    print("🧹 清理中间产物 PDF...")
    for pdf in pdf_files:
        if os.path.exists(pdf):
            os.remove(pdf)

    print("🚀 抽取幻灯片板式 Markdown 标签...")
    gt_count = 0
    for pptx in pptx_files:
        base_name = os.path.splitext(os.path.basename(pptx))[0]
        try:
            prs = Presentation(pptx)
            slide_width = prs.slide_width
            slide_height = prs.slide_height
            for i, slide in enumerate(prs.slides):
                md = slide_to_markdown_layout_aware(slide, slide_width, slide_height)
                gt_name = f"{base_name}_slide_{i:04d}.md"
                with open(os.path.join(GT_DIR, gt_name), "w", encoding="utf-8") as f:
                    f.write(md)
                gt_count += 1
        except Exception as e:
            print(f"抽取 {pptx} 失败: {e}")
    print(f"完成 {gt_count} 个 Markdown 标注的输出。")

    pptx_names = list(set([os.path.splitext(os.path.basename(f))[0] for f in pptx_files]))
    all_items = []
    if os.path.exists(PNG_DIR):
        for filename in sorted(os.listdir(PNG_DIR)):
            if filename.endswith(".png"):
                base = os.path.splitext(filename)[0]
                md_path = os.path.join(GT_DIR, base + ".md")
                if os.path.exists(md_path):
                    with open(md_path, "r", encoding="utf-8") as f:
                        md_content = f.read().strip()
                    if md_content:
                        all_items.append({
                            "image": os.path.join(PNG_DIR, filename),
                            "text": md_content,
                            "base": base
                        })

    train_data = []
    val_data = []

    # 按文件集切分，防止同个 PPT 内的幻灯片同时泄漏进训练与验证集
    if len(pptx_names) > 1:
        random.seed(42)
        random.shuffle(pptx_names)
        split_idx = max(1, int(len(pptx_names) * 0.9))
        train_pptxs = set(pptx_names[:split_idx])

        for item in all_items:
            belonging_pptx = None
            for p in pptx_names:
                if item["base"].startswith(p):
                    belonging_pptx = p
                    break

            clean_item = {"image": item["image"], "text": item["text"]}
            if belonging_pptx in train_pptxs:
                train_data.append(clean_item)
            else:
                val_data.append(clean_item)
    else:
        random.seed(42)
        random.shuffle(all_items)
        split_idx = max(1, int(len(all_items) * 0.8))
        for i, item in enumerate(all_items):
            clean_item = {"image": item["image"], "text": item["text"]}
            if i < split_idx:
                train_data.append(clean_item)
            else:
                val_data.append(clean_item)

    with open(TRAIN_JSONL, "w", encoding="utf-8") as f:
        for item in train_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    with open(VAL_JSONL, "w", encoding="utf-8") as f:
        for item in val_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")

    print(f"数据切分完毕，训练集: {len(train_data)} 个样本, 验证集: {len(val_data)} 个样本")

    # 本地存档与版本上传至 HF dataset 备份
    versioned_tar_name = f"ppt2md_dataset_{VERSION}.tar.gz"
    versioned_tar_path = os.path.join(BASE_DIR, versioned_tar_name)
    print(f"📦 正在打包备份版本归档文件: {versioned_tar_name}...")
    with tarfile.open(versioned_tar_path, "w:gz") as tar:
        tar.add(PNG_DIR, arcname="slides_png")
        tar.add(TRAIN_JSONL, arcname="train.jsonl")
        tar.add(VAL_JSONL, arcname="val.jsonl")

    try:
        api = HfApi()
        api.create_repo(repo_id=DATASET_REPO, repo_type="dataset", exist_ok=True)
        print(f"📤 正在上传版本化数据集压缩包 {versioned_tar_name}...")
        api.upload_file(
            path_or_fileobj=versioned_tar_path,
            path_in_repo=versioned_tar_name,
            repo_id=DATASET_REPO,
            repo_type="dataset"
        )
        print(f"🎉 新版数据集成功备份上传至：https://huggingface.co/datasets/{DATASET_REPO}")
    except Exception as e:
        print(f"⚠️ 数据集云端备份上传失败 (原因: {e})")

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
