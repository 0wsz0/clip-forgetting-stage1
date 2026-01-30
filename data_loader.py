#load_datasets：来自Hugging Face的datasets库，用于加载标准数据集
from datasets import load_dataset
import numpy as np

def load_cifar10_cat_dog_bird():
    """
    加载CIFAR-10测试集，筛选cat/dog/bird三类
    :return images(list),labels(list),label_names(list),
    """
    #加载测试集
    test_dataset = load_dataset("cifar10",split="test")

    # CIFAR-10标签映射: 0=airplane,1=automobile,2=bird,3=cat,4=deer,5=dog...
    target_classes=[3,5,2] # cat,dog,bird
    class_names={2:'bird',3:'cat',5:'dog'} #{label:'name'}

    #筛选目标类别
    filtered_images = []
    filtered_labels = []
    """
    sample是一个字典
    {
    'img': <PIL.Image.Image image mode=RGB size=32x32 at 0x...>,
    'label': 3
    }
    """
    for sample in test_dataset:
        label=sample['label']
        if label in target_classes:
            filtered_images.append(sample['img']) #PIL Image对象
            filtered_labels.append(label)

    print(f"筛选完成: 共{len(filtered_images)}张图片")
    print(f"  - bird: {filtered_labels.count(2)}张")
    print(f"  - cat:  {filtered_labels.count(3)}张")
    print(f"  - dog:  {filtered_labels.count(5)}张")

    return filtered_images, filtered_labels, class_names



