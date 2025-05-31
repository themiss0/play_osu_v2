# V2版本

5.14
## 目标
- 基本能预测note和slider的位置和打击时间


## 实现方式和过程

通过CNN+RNN(GRU)实现图像特征提取和时间序列特征分析，依然是输出热图，通过heat判定点击和长按。

发现osrparser可以提取replay信息，可以提取autoplay的鼠标点击和坐标来制作数据集。

发现V1输出的heatmaps同时包含了点击概率和光标位置概率，两者耦合可能会混淆模型关注点，准备改成点击概率序列+位置heatmaps的形式，但是如何制作数据集和设计模型呢

数据集现在分离了，每个map一个文件夹，包含了训练需要的数据
meta.pkl:字典类型，包含map的cs和ar
rep.osr:auto模式的回放，通过osrparser提取光标轨迹和点击作为数据
pics.npy:三维nparray，[pic, h, w]，作为map的图像序列，每张图片也是下面对象的基准
times.npy:一维nparray，每张pic对应的time
heats.npy:三维nparray，[heat, h, w]，热图，由HeatGen生成，用于描述光标注意力焦点
clicks.npy:一维nparray，对应改时间需要点击的概率
