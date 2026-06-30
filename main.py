import argparse
from config import authenticate_huggingface
from utils import check_gpu
from dataset_prep import execute_data_pipeline
from train import start_training
from evaluate import perform_comparative_evaluation

def main():
    parser = argparse.ArgumentParser(description="MinerU2.5-Pro PPT to Markdown QLoRA Fine-tuning Toolkit")
    parser.add_argument(
        "--action",
        type=str,
        default="all",
        choices=["prep-data", "train", "evaluate", "all"],
        help="指定执行阶段: 数据预准备 (prep-data) | 适配器微调 (train) | 效果评估 (evaluate) | 完整执行 (all)"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        default=None,
        help="原生 PPTX 输入路径。当云端无可用备份集且本地无现成切片时，用于重构本地数据集。"
    )
    args = parser.parse_args()

    # 安全凭证认证与 GPU 设备状态初始化
    authenticate_huggingface()
    check_gpu()

    if args.action in ["prep-data", "all"]:
        print("\n=== 阶段 1: 正在运行数据集加载与校验流程 ===")
        execute_data_pipeline(local_input_dir=args.input_dir)

    if args.action in ["train", "all"]:
        print("\n=== 阶段 2: 正在载入预热并运行模型微调 ===")
        start_training()

    if args.action in ["evaluate", "all"]:
        print("\n=== 阶段 3: 正在开展双模型指标效能分析 ===")
        perform_comparative_evaluation()

if __name__ == "__main__":
    main()
