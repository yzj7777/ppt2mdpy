import os
import gc
import time
import json
import torch
from PIL import Image
from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration
from peft import PeftModel

from config import MODEL_NAME, MODEL_REPO, VAL_JSONL, BASE_DIR, EVAL_VERSION
from utils import normalize, normalized_edit_distance, char_f1
from model_utils import apply_inference_patch, configure_processor_pixels

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
    for idx, item in enumerate(val_items):
        image = Image.open(item["image"]).convert("RGB")
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

def perform_comparative_evaluation():
    """彻底回收先前训练资源，对比 Baseline 与特定 Tag 模型的抽取能力"""
    gc.collect()
    torch.cuda.empty_cache()

    config = AutoConfig.from_pretrained(MODEL_NAME, trust_remote_code=True)
    config.tie_word_embeddings = True

    # 1. Baseline 数据表现评估
    print("\n🔄 评估流程开始。优先评估 Baseline (Pre-fine-tune) 底模表现...")
    baseline_model = Qwen2VLForConditionalGeneration.from_pretrained(
        MODEL_NAME,
        config=config,
        torch_dtype=torch.float16,
        trust_remote_code=True
    ).to("cuda")
    apply_inference_patch(baseline_model)

    baseline_processor = AutoProcessor.from_pretrained(MODEL_NAME, trust_remote_code=True)
    baseline_processor = configure_processor_pixels(baseline_processor)

    pre_ned, pre_f1, pre_lat = evaluate_model_pipeline(baseline_model, baseline_processor, VAL_JSONL)

    # 销毁首个模型显存
    del baseline_model, baseline_processor
    gc.collect()
    torch.cuda.empty_cache()

    # 2. 注入指定版本 Tag 的 LoRA 适配器并评估
    print(f"\n🔄 加载云端微调完毕后的模型并合并 (目标版本 Tag: {EVAL_VERSION})...")
    try:
        base_for_ft = Qwen2VLForConditionalGeneration.from_pretrained(
            MODEL_NAME,
            config=config,
            torch_dtype=torch.float16,
            trust_remote_code=True
        ).to("cuda")
        apply_inference_patch(base_for_ft)

        # 🌟 revision 指定装载的云端版本分支 Git Tag 🌟
        ft_adapter_model = PeftModel.from_pretrained(
            base_for_ft,
            MODEL_REPO,
            revision=EVAL_VERSION
        )
        ft_model = ft_adapter_model.merge_and_unload()

        ft_processor = AutoProcessor.from_pretrained(
            MODEL_REPO,
            trust_remote_code=True,
            revision=EVAL_VERSION
        )
        ft_processor = configure_processor_pixels(ft_processor)

        post_ned, post_f1, post_lat = evaluate_model_pipeline(ft_model, ft_processor, VAL_JSONL)

        # ==================== 打印多版本评估对比结果 ====================
        print("\n" + "=" * 60)
        print(f"          MinerU2.5-Pro QLoRA Fine-tuning Evaluation ({EVAL_VERSION})")
        print("=" * 60)
        print(f"{'Metric':<20} {'Pre-fine-tune':>12} {'Post-fine-tune':>12} {'Improvement':>12}")
        print("-" * 60)
        print(f"{'Avg NED (↑)':<20} {pre_ned:>12.4f} {post_ned:>12.4f} {post_ned - pre_ned:>+12.4f}")
        print(f"{'Avg F1 (↑)':<20} {pre_f1:>12.4f} {post_f1:>12.4f} {post_f1 - pre_f1:>+12.4f}")
        print(f"{'Avg Latency (s)':<20} {pre_lat:>12.2f} {post_lat:>12.2f} {post_lat - pre_lat:>+12.2f}")
        print("=" * 60)

        report_name = os.path.join(BASE_DIR, f"comparison_report_{EVAL_VERSION}.txt")
        with open(report_name, "w") as f:
            f.write(f"MinerU2.5-Pro QLoRA Fine-tuning Comparison Report ({EVAL_VERSION})\n")
            f.write(f"Pre-fine-tune NED: {pre_ned:.4f}, F1: {pre_f1:.4f}\n")
            f.write(f"Post-fine-tune ({EVAL_VERSION}) NED: {post_ned:.4f}, F1: {post_f1:.4f}\n")
            f.write(f"NED Improvement: {post_ned - pre_ned:+.4f}\n")
            f.write(f"F1 Improvement:  {post_f1 - pre_f1:+.4f}\n")

        print(f"✅ 成功输出评估对比报告到: {report_name}")
    except Exception as e:
        print(f"❌ 无法从在线仓库 {MODEL_REPO} (Revision: {EVAL_VERSION}) 加载对应权重: {e}")
