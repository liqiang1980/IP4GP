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

## hand structure and tactile sensor array
- links definition (refer ../UR/allegro_hand_right.pdf for structure)
  - first finger[FF] (link_0.0[proximity] to link_3.0_tip[tip])
  - middle finger[MF] (link_4.0[proximity] to link_7.0_tip[tip])
  - ring finger[RF] (link_8.0[proximity] to link_11.0_tip[tip])
  - thumb [TH] (link_12.0[proximity] to link_15.0_tip[tip])
  - sensor touch_x_y_z in xml file: 
    - x- taxels area
      - 111 plam; 
      - 2 ff proximity link; 1 ff middle link; 0 ff distal link; 
      - 9 mf proximity link; 8 mf middle link; 7 mf distal link; 
      - 13 rf proximity link; 12 rf middle link; 11 rf distal link; 
      - 16 th middle link; 15 th distal link; 
    - y-row; 
    - z-column

## tactile array ：
- the number of taxels on plam is: 
  ```bash
  15 * 4 + 13 + 6 * 4 + 5 + 4 + 7 = 113 taxels
  ```
- on fingertip 
  ```bash
  6*12 = 72 taxels
  ```
- on other finger link 
  ```bash
  6*6 = 36 taxels
  ```

