import os
#解决symlinks警告
os.environ["HF_HUB_DISABLE_SYMLINKS_WARNING"] = "1"
#离线模式（首次运行需要联网）
#os.environ["HF_HUB_OFFLINE"] = "1"

import torch
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
from utils.data_loader import load_cifar10_cat_dog_bird
import pandas as pd

# ========== 配置 ==========
DEVICE = "cpu"
MODEL_NAME = "openai/clip-vit-base-patch32"
PROMPTS = [
    "a photo of a cat",
    "a photo of a dog",
    "a photo of a bird",
]
# CIFAR-10的标签 -> 三类的索引映射
LABEL_TO_IDX = {3:0,5:1,2:2} #cat->0,dog->1,bird->2
IDX_TO_NAME = {0:"cat", 1:"dog", 2:"bird"}

# ========== 主流程 ==========
def main():
    # 1. 加载数据
    print("【步骤1】加载CIFAR-10数据")
    images,true_labels,_ = load_cifar10_cat_dog_bird()

    # 2. 加载CLIP模型
    print("【步骤2】加载CLIP模型...")
    model = CLIPModel.from_pretrained(MODEL_NAME).to(DEVICE)
    processor = CLIPProcessor.from_pretrained(MODEL_NAME)
    model.eval() # 推理模式

    # 3. 推理
    print("【步骤3】开始推理（约1-2分钟）...")
    predictions = []

    with torch.no_grad(): # 关闭梯度计算，加速推理
        for i,image in enumerate(tqdm(images,desc="推理进度")):
            # 处理单张照片
            inputs = processor(
                text =PROMPTS,
                images = image,
                return_tensors="pt",
                padding=True,
            )
            inputs={k:v.to(DEVICE) for k,v in inputs.items()}

            # 前向传播
            outputs = model(**inputs)
            logits = outputs.logits_per_image # shape: [1,3]

            #softmax + argmax（任务要求）
            probs = logits.softmax(dim=1)
            pred_idx = probs.argmax(dim=1).item() # 0,1,2

            predictions.append(pred_idx)

    # 4. 评估
    print("【步骤4】计算准确率...")
    # 转换真实标签为0/1/2 索引
    true_indices = [LABEL_TO_IDX[i] for i in true_labels]

    # 计算每类准确率
    results = []
    for idx in range(3):
        class_name = IDX_TO_NAME[idx]
        # 筛选该类别的样本
        indices = [i for i,label in enumerate(true_indices) if label == idx]
        correct = sum(1 for i in indices if predictions[i] == idx)
        total = len(indices)
        acc = correct / total if total > 0 else 0
        results.append({
            "类别":class_name,
            "准确率":f"{acc*100:.2f}%",
            "正确数":correct,
            "总数":total
        })
        print(f"{class_name:5s}:{acc*100:6.2f}%({correct}/{total})")

    # 总体准确率
    total_correct = sum(1 for p,t in zip(predictions,true_indices) if p==t)
    total_samples = len(true_indices)
    overall_acc = total_correct / total_samples
    results.append({
        "类别":"总体",
        "准确率":f"{overall_acc*100:6.2f}%",
        "正确数":total_correct,
        "总数":total_samples
    })
    print(f"{'总体':5s}:{overall_acc*100:6.2f}%({total_correct}/{total_samples})")

    # 5. 保存结果
    print("【步骤5】保存结果...")
    os.makedirs("results", exist_ok=True)
    df = pd.DataFrame(results)
    df.to_csv("results/accuracy_table.csv", index=False,encoding="utf-8-sig")
    print("准确率表格已保存至：results/accuracy_table.csv")

    # 6. 生成考核要求的纯文本表格
    print("\n【考核提交用表格】")
    print(df.to_string(index=False))

    return overall_acc >= 0.9 and all(
        float(r["准确率"].rstrip("%")) >= 90 for r in results[:-1]
    )

if __name__ == "__main__":
    success = main()
    if success:
        print("✅ 阶段一任务完成！所有类别准确率 ≥90%")

    print("="*50)


