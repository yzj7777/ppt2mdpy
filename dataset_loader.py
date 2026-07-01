import os
import json
import torch
from PIL import Image

class SlideDataset(torch.utils.data.Dataset):
    """载入幻灯片图片与 target Markdown 文本对的数据集"""
    def __init__(self, jsonl_path):
        self.samples = []
        if os.path.exists(jsonl_path):
            with open(jsonl_path, "r", encoding="utf-8") as f:
                for line in f:
                    self.samples.append(json.loads(line.strip()))
        print(f"📦 共检测到并装载了 {len(self.samples)} 条微调样本数据")

    def __len__(self):
        return len(self.samples)

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

def get_collate_fn(processor):
    """获取自定义批处理逻辑，只对助理回复段落执行计算"""
    def collate_fn(batch):
        texts = [
            processor.apply_chat_template(msg, tokenize=False, add_generation_prompt=False)
            for msg in batch
        ]

        images = []
        for msg in batch:
            for turn in msg:
                if turn["role"] == "user":
                    for content in turn["content"]:
                        if content["type"] == "image":
                            images.append(content["image"])

        inputs = processor(
            text=texts,
            images=images,
            padding=True,
            return_tensors="pt"
        )

        labels = inputs["input_ids"].clone()
        labels[labels == processor.tokenizer.pad_token_id] = -100

        image_tokens = [
            processor.tokenizer.convert_tokens_to_ids(""),
            processor.tokenizer.convert_tokens_to_ids("<think>"),
            processor.tokenizer.convert_tokens_to_ids("</think>"),
        ]
        for token_id in image_tokens:
            if token_id is not None:
                labels[labels == token_id] = -100

        assistant_prefix = processor.tokenizer.encode("</think>", add_special_tokens=False)
        n_prefix = len(assistant_prefix)

        for i in range(len(batch)):
            input_id_list = inputs["input_ids"][i].tolist()
            found_idx = -1
            for idx in range(len(input_id_list) - n_prefix + 1):
                if input_id_list[idx : idx + n_prefix] == assistant_prefix:
                    found_idx = idx + n_prefix
                    if idx + n_prefix < len(input_id_list) and input_id_list[idx + n_prefix] in [198, 271]:
                        found_idx += 1
                    break

            if found_idx != -1:
                labels[i, :found_idx] = -100

        inputs["labels"] = labels
        return inputs
    return collate_fn
