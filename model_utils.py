import torch
from transformers import AutoConfig, AutoProcessor, Qwen2VLForConditionalGeneration
from peft import LoraConfig, get_peft_model, TaskType

def apply_inference_patch(target_model):
    """视觉模块推理类型转换补丁"""
    visual_module = None
    for name, module in target_model.named_modules():
        if name.endswith("visual") or name == "visual":
            visual_module = module
            break
    if visual_module:
        type(visual_module).dtype = property(lambda x: torch.float16)

def configure_processor_pixels(processor):
    """通过限制视觉最高分辨率压缩显存"""
    min_pixels = 256 * 28 * 28
    max_pixels = 512 * 28 * 28
    if hasattr(processor, "image_processor"):
        processor.image_processor.min_pixels = min_pixels
        processor.image_processor.max_pixels = max_pixels
        print(f"✅ Processor 像素限制已成功设为: {max_pixels}")

    if processor.tokenizer.pad_token_id is None:
        processor.tokenizer.pad_token_id = processor.tokenizer.eos_token_id

    processor.tokenizer.padding_side = "right"
    return processor

def load_base_model_and_processor(model_name):
    """加载原始底模、应用强制 SDPA 及视觉模块类型纠正"""
    config = AutoConfig.from_pretrained(model_name, trust_remote_code=True)
    config.tie_word_embeddings = True

    model = Qwen2VLForConditionalGeneration.from_pretrained(
        model_name,
        config=config,
        torch_dtype=torch.float16,
        device_map="auto",
        attn_implementation="sdpa",
        trust_remote_code=True,
    )
    processor = AutoProcessor.from_pretrained(model_name, trust_remote_code=True)
    processor = configure_processor_pixels(processor)

    # 注册 _dtype_fix 确保 FP16 参数校验兼容性
    vis_module = model.model.visual
    vis_module.register_parameter(
        '_dtype_fix',
        torch.nn.Parameter(torch.zeros(1, dtype=torch.float16), requires_grad=False)
    )

    return model, processor

def prepare_model_for_qlora(model):
    """注入 LoRA 参数，切断视觉塔的反向传播梯度"""
    model.config.use_cache = False
    model.gradient_checkpointing_enable()
    model.enable_input_require_grads()

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        target_modules=[
            "q_proj", "k_proj", "v_proj", "o_proj",
            "gate_proj", "up_proj", "down_proj"
        ],
        lora_dropout=0.05,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    model = get_peft_model(model, lora_config)

    # 阻断视觉塔参数梯度以防止激活缓存占用多余显存
    if hasattr(model, "visual") or (hasattr(model, "model") and hasattr(model.model, "visual")):
        v_tower = model.visual if hasattr(model, "visual") else model.model.visual
        for param in v_tower.parameters():
            param.requires_grad = False
        print("✅ 整个视觉塔已设为只读。")

    model.print_trainable_parameters()
    return model
