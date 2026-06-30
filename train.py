import os
import torch
import transformers
from transformers import Trainer, TrainingArguments
from huggingface_hub import HfApi

from config import MODEL_NAME, MODEL_REPO, BASE_DIR, TRAIN_JSONL, VAL_JSONL, VERSION
from model_utils import load_base_model_and_processor, prepare_model_for_qlora
from dataset_loader import SlideDataset, get_collate_fn

def apply_final_patch(model):
    visual_module = None
    if hasattr(model, "visual"):
        visual_module = model.visual
    elif hasattr(model, "base_model") and hasattr(model.base_model.model, "visual"):
        visual_module = model.base_model.model.visual

    if visual_module:
        print("✅ 锁定视觉模块类级别 FP16 特性...")
        type(visual_module).dtype = property(lambda x: torch.float16)
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

def start_training():
    transformers.logging.set_verbosity_warning()

    print("🔄 开始加载预训练模型...")
    model, processor = load_base_model_and_processor(MODEL_NAME)
    model = prepare_model_for_qlora(model)
    apply_final_patch(model)

    train_dataset = SlideDataset(TRAIN_JSONL)
    val_dataset = SlideDataset(VAL_JSONL)
    collate_fn = get_collate_fn(processor)

    try:
        import bitsandbytes
        optimizer_choice = "paged_adamw_8bit"
        print("✅ 已成功挂载 bitsandbytes 8-bit Paged 优化器进行显存优化。")
    except ImportError:
        optimizer_choice = "adafactor"
        print("⚠️ 未能在本地导入 bitsandbytes 优化模块。系统自动退回默认 Adafactor。")

    training_args = TrainingArguments(
        output_dir=os.path.join(BASE_DIR, "mineru_ft"),
        num_train_epochs=8,
        per_device_train_batch_size=1,
        gradient_accumulation_steps=4,
        learning_rate=2e-5,
        fp16=True,
        logging_steps=5,
        save_strategy="epoch",
        eval_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="loss",
        greater_is_better=False,
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        remove_unused_columns=False,
        report_to="none",
        dataloader_num_workers=0,
        dataloader_pin_memory=False,
        dataloader_persistent_workers=False,
        optim=optimizer_choice,

        # 验证显存防护
        per_device_eval_batch_size=1,
        eval_accumulation_steps=1,
        prediction_loss_only=True,
    )
    training_args._n_gpu = 1

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=collate_fn,
    )

    print("🚀 启动 QLoRA 微调训练进程...")
    try:
        trainer.train()
        print("✅ 本地训练完美闭环完成！")

        local_save_dir = os.path.join(BASE_DIR, "mineru_ft_final")
        trainer.save_model(local_save_dir)
        processor.save_pretrained(local_save_dir)
        print(f"✅ 微调适配器已保存在本地目录: {local_save_dir}")

        print(f"☁️ 正在推送最新微调参数至仓库 {MODEL_REPO} 的 main 主干...")
        model.push_to_hub(MODEL_REPO, private=True)
        processor.push_to_hub(MODEL_REPO, private=True)

        # 🌟 自动生成对应的 Release Git Tag 🌟
        try:
            api = HfApi()
            print(f"🏷️ 正在向远程库打上此次微调的新版本 Tag 标签: {VERSION}...")
            api.create_tag(
                repo_id=MODEL_REPO,
                tag=VERSION,
                tag_type="model",
                comment=f"MinerU QLoRA Fine-tuned model at tag release: {VERSION}."
            )
            print(f"🎉 包含 Tag {VERSION} 标签的模型已完美上传至：https://huggingface.co/{MODEL_REPO}/tree/{VERSION}")
        except Exception as tag_err:
            print(f"⚠️ 云端模型版本 Tag 构建失败 (原因: {tag_err})")

    except Exception as e:
        import traceback
        traceback.print_exc()
