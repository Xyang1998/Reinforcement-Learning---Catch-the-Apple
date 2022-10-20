# Reinforcement-Learning---Catch-the-Apple
使用强化学习训练AI玩接苹果
reward函数有些问题，如果换成最低苹果的水平距离的话效果可能会更好。
AI不做跳跃动作，之前训练时发现ai会变成一个只会跳的憨憨（可能和reward函数有关）
由于torch检测不到显卡（torch版本1.12.1+cuda11.6），将原设计的vgg16改为小型CNN，使用cpu训练。
如果想重新训练AI，将game.py中Play设为False并移除两个模型文件即可。
