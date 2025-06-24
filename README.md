# Play Osu Like Neuro
正在实现一个能够玩osu的AI
目前只是一个课程作业级别的项目，<3


## 配置方法

1. 环境要求：Windows10及以上系统, python解释器为3.10版本
2. python依赖：请详见根目录下的requirements.txt
   **注意**
   torch_directml不是必须的，如果你已经配置好cuda并打算用cuda推理，不要安装这个依赖
   
3. 其他依赖：
- 最新的osu!lazer游戏，可以在[官网](osu.ppy.sh)上下载
- 最新的[tosu](github.com/tosuapp/tosu)插件

4. 构建流程：
   由于构建流程设计到数据集采集和模型训练，比较繁琐，故不细讲，只给出推理流程。

5. 推理流程：
   1. 将项目克隆到本地
   2. 打开游戏和tosu插件，等待tosu运行并弹出控制台webui，将tosu检测频率调到最小
   3. 在设置中代开游戏边界，设置mod为relex
   4. 运行box_detect，并进入游戏任意一个谱面，记录box_detect的输出
   5. 将稳定的数值修改setting中的play_field
   6. 退出游戏到选择界面
   7. 运行Main，等待加载完成后，进入游戏任意一张谱面
   8. 此时鼠标会自动移动，并有新窗口显示模型可视化输出
   9. 等待运行完成后Main自动关闭
   10. 可以运行HeatmapViewer查看上一次推理的回放

6. 其他杂项
   1. 大部分模块在头部有控制参数，可以调整模块功能
   2. 游戏状态检查代码比较简陋，如果被鼠标被硬控了出不来可以CTRL+ALT+DEL，程序会被关闭
   3. 项目使用的游戏的皮肤是经过定制的，皮肤包位于build_req中，准确率可能会下降。


7. 关于大作业查库：
   应该转到deployable分支来检查。main分支高速更新中，大概率跑不起来。