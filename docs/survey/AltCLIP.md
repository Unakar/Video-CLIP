# AltCLIP
https://arxiv.org/pdf/2211.06679.pdf
## Abstract
这项工作利用知识蒸馏的方法来训练一个强大的双语/多语言多模态表示模型。从CLIP开始，改变了其文本编码器，使用预训练的多语言文本编码器XLM-R，并通过两阶段训练方案（包括teacher learning和对比学习）对两种语言和图像表示进行对齐，扩展了clip的多语言能力

其实这项工作实现起来比较简单，可以作为大作业的基础

## Framework
### teacher learning
将clip的文本encoder作为teacher，XLM-R作为student，使用一个全连接层对齐。在训练时输入$(sent_1,sent_2)$，$sent_1$是一个[TOS]的embedding，$sent_2$是一个[CLS]的embedding，将他们分别放入teacher和student后，对得到的结果计算平方误差。（这里也是我们可以后期改进的点）
### 对比学习
这部分目的是通过对比学习增强多语言能力，使用clip的image encode，将上一步学习得到的student作为text encoder，使用交叉熵作为损失函数进行微调。

## Experiment
### Training Datasets
1. TSL2019 5M大小的中文到英文的翻译数据集
2. OPUS 一个开源平行语料库
3. Wudao MM 中文文本-图像数据集
4. LAION 5B 英文文本-图像数据集
5. LAION Multilingual 2B 多语言文本-图像数据集

## Metirc
1. 多语言图像分类任务：
在多语言图像分类任务上实现了高精度的识别能力，几乎在所有测试的语言上都保持了90%以上的准确率。
2. 跨模态检索任务：
在Flickr30k和MSCOCO的英文版和中文版（Flickr30kCN和MSCOCOCN）上，AltCLIP在文本到图像检索和图像到文本检索任务上的表现优于多个基线模型，包括原始的CLIP模型。
3. zero shot
在ImageNet及其变体数据集上，AltCLIP在零样本分类任务上的表现接近原始的CLIP模型。不过在中文版本的ImageNet上性能有所提高。

## idea
这篇文章在训练时使用的数据集较小，性能上提升其实不是很明显，可能还是文章发的比较早。不过利用teacher learning进行知识蒸馏这点可以借鉴。