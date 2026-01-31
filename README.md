# 🧠CLIP 失忆手术/定向遗忘 · 阶段一：CIFAR-10三类分类

## 📋任务说明
🤖使用预训练 CLIP-base 模型，在 CIFAR-10 测试集中对 **cat / dog / bird** 三类进行分类
- 💬固定 prompt:：`['a photo of a bird', 'a photo of a cat', 'a photo of a dog']`
- 🧮分类逻辑： logits -> softmax -> argmax
- ✅目标：每类准确率 >=90%

## 🛠️环境配置
 - 🖥️操作系统：Windows 11
 - 🐍Python:3.10.11
 - 🔥PyTorch:2.10.0+cpu
 - 🤗transformers:4.35.0
 - 📦其他依赖：见 `requirements.txt` 

## ▶️运行步骤
1.安装依赖
```bash
   pip install -r requirements.txt
```
2.运行推理脚本
```bash
   python main.py
```

## 📊预期输出示例
- 控制台显示推理进度条
- 控制台输出每类准确率及总体准确率
- 生成文件: `results/accuracy_table.csv`
```text
类别    准确率   正确数   总数
cat     87.80%   878      1000
dog     90.70%   907      1000
bird    93.20%   932      1000
总体    90.57%   2717     3000
```
## 📝 可复现性保障
- 所有结果均由 `main.py` 自动计算并输出至 `results/accuracy_table.csv`
- 脚本已通过 **全新虚拟环境验证**（Python 3.10 + clean venv）
- 首次运行耗时：约 5 分钟（Intel Ultra 9 275HX + 32GB RAM + CPU推理）   

## 📂 项目结构
```text
clip-forgetting-stage1/
├── main.py
├── requirements.txt
├── README.md
├── results/
│   └── accuracy_table.csv
└── utils/
    └── data_loader.py

```
