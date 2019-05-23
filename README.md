# deep_head_pose_test
modify the codes deep_head_pose of the paper: Fine-Grained Head Pose Estimation Without Keypoints


[论文笔记](https://blog.csdn.net/qq_42189368/article/details/84849638)  
[代码链接](https://github.com/natanielruiz/deep-head-pose)  

1.引言：  
本文提出了一种简洁和鲁棒的方式来确定姿态，通过训练一个multi-loss的卷积神经网络。
直接使用RGB结合分类和回归损失来预测Euler angles（yaw，pitch and roll）。

2.网络结构：  
![](https://ws1.sinaimg.cn/large/cdd040eely1g3b31srr0wj20qk0ao40b.jpg)  

本文提出使用3个分离的losses，为每一个角度。每个loss由两部分组成：a binned pose classification and a regression component 组成。  

最后为每一个欧拉角的损失为:  
![](https://ws1.sinaimg.cn/large/cdd040eely1g3b321bj0kj20cr02lq35.jpg)  

3.实现细节：  
1） 对欧拉角（Yaw，Pitch，Roll）按角度区间进行分类，比如3度，那么Yaw：-90-+90，可以分成180/3= 60个类别，Pitch和Roll同Yaw角类似。这样就可以进行分类任务了。  
2） 对分类的结果恢复成实际的角度，类别*3-90，在和实际的角度计算回归损失。  
3） 最后将回归损失和分类损失进行合并来得到最后的损失，回归损失的前面增加了一个权重系数α。  
