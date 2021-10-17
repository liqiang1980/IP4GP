# Tactile_EKF_Pose_Estimation

## 文件说明
- pcd_cut 用于裁切点云，根据需要的区域对点云进行裁切
- pcd2BVH 用于将点云转换为BVH模型需要的输入类型，输出的文件为.obj格式
- tactile 为姿态估计的python程序，直接运行里面的ur5_allegro.py即可，如果报库确实，使用pip install安装需要的库
- checkout pcl from:  https://github.com/PointCloudLibrary/pcl.git
  pcl/build/bin/pcl_example_nurbs_fitting_surface 使用B样条曲线拟合曲面,需要先编译文件。
- checkout TinyEKF from:  https://github.com/simondlevy/TinyEKF.git
  为准备的EKF程序

