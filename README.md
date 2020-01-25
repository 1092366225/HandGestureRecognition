# HandGestureRecognition(Based on skin color detection using otsu)
1.利用摄像头实时采集视频<br>
2.根据肤色在YCrCb颜色空间的聚类性质，采用otsu自适应阈值法来对Cr通道进行划分肤色区域和背景。<br>
3.然后使用模板匹配的方法区分脸部和手部，得到手部区域的外接矩形。<br>
4.用手部外接矩形的左上角坐标来代表手的位置，跟踪手部运动轨迹<br>
5.采用16连通链码对坐标序列离散化，得到动态手势的特征向量。<br>
