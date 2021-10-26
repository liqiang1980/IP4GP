# Interactive Perception for Grasping Procedure (IP4GP)

## 文件说明
- create_point.py 由stl文件产生点云
- create_point_matlab.py 由点云文件产生matlab能够读取的点云数据
- fcl_python 为fcl碰撞检测库 读取.obj文件的接口
- func.py 为mujoco-py的部分封装的函数，如读取相对位姿、基于jacobian的手指运动学逆解等
- ur5_allegro.py为运行的主程序
- ur5_allegro_update.py为添加了EKF后验更新的主程序
- 3dparty 文件夹 为pykdl库
- pos_save文件夹为ur5_allegro.py运行产生的ground_truth和估计姿态的曲线，使用plot_pose.py即可读取
- read_point.py用来读取点云文件并可视化输出，用于裁切点云后的可视化  


