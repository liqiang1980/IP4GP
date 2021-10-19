# Interactive Perception for Grasping Procedure (IP4GP)

## 文件说明
- pcd_cut 用于裁切点云，根据需要的区域对点云进行裁切. Please complile with following cmd
  ```bash
  cd pcd_cut/build
  ccmake ../
  make -j3
  ```
- pcd2BVH 用于将点云转换为BVH模型需要的输入类型，输出的文件为.obj格式. Please complile with following cmd
  ```bash
  cd pcd2BVH
  ccmake ../
  make -j3
  ```
- tactile 为姿态估计的python程序，直接运行里面的ur5_allegro.py即可，如果报库确实，使用pip install安装需要的库
- checkout pcl from:  https://github.com/PointCloudLibrary/pcl.git

  pcl/build/bin/pcl_example_nurbs_fitting_surface 使用B样条曲线拟合曲面,需要先编译文件。

- checkout TinyEKF from:  https://github.com/simondlevy/TinyEKF.git
  为准备的EKF程序

## 运行程序


Point cloud pre-processing
  ```bash
- 运行tactile/create_point.py 由stl文件产生点云
- 运行pcd_cut/build中 ./point_cut 裁切点云
- 运行pcd_cut/build/read_point.py 查看输出的点云
- pcd2BVH/build 中 ./greedy_project 生成BVH模型  
  ```

运行程序 in tactile folder
```bash
python3 ur5_allegro.py
```

查看输出 in tactile folder
```bash
python3 plot_pose.py
```

# install 3rdparty components

## install pykdl
- https://github.com/orocos/orocos_kinematics_dynamics
    - install orocos_kdl 
        
        read related INSTALL.md at folder  3dparty/orocos_kinematics_dynamics/orocos_kdl/INSTALL.md
    - install python_orocos_kdl
        
        read related INSTALL.md at folder  3dparty/orocos_kinematics_dynamics/python_orocos_kdl/INSTALL.md

        - Note: Using cmake with python3

            cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_VERSION=3 ..

## install pykdl_utils
dependence PyKDL

```bash
cd 3dparty/pykdl_utils
pip install -e .
```

## install urdf_parser_py
use load URDF to kdl tree 
```bash
cd 3dparty/urdf_parser_py
pip install -e .
```






