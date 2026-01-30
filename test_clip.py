from transformers import CLIPModel, CLIPProcessor
import torch

import os
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"  #解决symlinks警告
# 测试模型加载（首次会下载 ~600MB）
print("正在加载 CLIP 模型...")
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")
print("✅ CLIP 模型加载成功！")

# 测试简单推理
from PIL import Image
import requests

# 下载示例图片
url = "http://images.cocodataset.org/val2017/000000039769.jpg"
image = Image.open(requests.get(url, stream=True).raw)
text = ["a photo of a cat", "a photo of a dog"]

inputs = processor(text=text, images=image, return_tensors="pt", padding=True)
with torch.no_grad():
    outputs = model(**inputs)
logits = outputs.logits_per_image  # 图像-文本相似度
probs = logits.softmax(dim=1)
print(f"预测概率: cat={probs[0][0]:.2f}, dog={probs[0][1]:.2f}")