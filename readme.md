# Adapting CLIP for Cross-modal Video Understanding

- 利用图像预训练模型CLIP，迁移搭建一个简单的视频理解模型。要求模型能处理视频（多个图像），完成文本到视频检索的pipline即可。（zero-shot）
- 针对具体的任务，改进搭建的视频模型，并在任务上微调，如，对时序做建模（VTR任务），利用CLIP的文本分支做prompt设计（VAR任务），两阶段提议生成+细粒度匹配（VMR任务）
- 更多内容的挖掘，如，CLIP迁移高效微调（adapter或prompt tuning），数据集噪声问题修正，引入音频等其他模态，构建多任务通用的CLIP-based视频模型

# Contributors
谢天，洪毓谦，陈鹤影，林洛昊，张又升（排名不分先后）

