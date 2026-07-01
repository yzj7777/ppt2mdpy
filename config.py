import os
import datetime

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

# 托管模型与数据集信息配置
MODEL_NAME = "opendatalab/MinerU2.5-Pro-2604-1.2B"
DATASET_REPO = "spacecomputer777/ppt2md_dataset"
MODEL_REPO = "spacecomputer777/ppt2md_model"

# ==================== 全局版本号定义 ====================
# 基于当前运行时间戳生成唯一版本号
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
VERSION = f"v{timestamp}"

# 用于推理阶段指定需要评估的历史云端标签版本 (默认采用本次运行生成的新版本)
EVAL_VERSION = os.getenv("EVAL_VERSION", VERSION)

print(f"🏷️ 项目全局加载完成。本次微调自动生成的版本号: {VERSION}")


def authenticate_huggingface():
    """安全读取 HF_TOKEN 并调用 login 授权"""
    hf_token = os.getenv("HF_TOKEN")
    if not hf_token:
        try:
            from kaggle_secrets import UserSecretsClient
            user_secrets = UserSecretsClient()
            hf_token = user_secrets.get_secret("HF_TOKEN")
        except ImportError:
            pass

    if hf_token:
        os.environ["HF_TOKEN"] = hf_token
        from huggingface_hub import login
        login(token=hf_token, add_to_git_credential=False)
        print("✅ 已成功载入 HF 身份令牌验证")
    else:
        print("⚠️ 未能在环境中检测到 'HF_TOKEN' 密匙，微调成果上传可能需要手动登录验证。")
