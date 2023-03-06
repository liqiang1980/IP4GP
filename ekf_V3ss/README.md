### 最外层PY文件：

​	fcl_python.py：fcl库相关函数

​	func.py & func2.py：用于运算的函数

​	storeQR.py：存储了调试过程中的QR参数

​	surfaceFitting.py：曲面拟合相关函数

​	test_Plot_plus.py：绘图函数		



### Folder：

​	UR5：模型配置

​	One_Fin_Program & Two_Fin_Program & In_hand_Program：3种不同动作配置的仿真程序。

#### 	Folder：

​		pos_save：存储了初始化位置参数

​		save_date & save_f & save_i：存档数据，以便绘图

#### 	模型文件：

​		cup_1.obj & fingertip_part.obj：仿真模型实体

#### 	参数文件：

​		err_oneFin_v3bi.txt：存储了iEKF的误差补偿参数

#### 	执行主文件：

​		one_fin_v3B.py & one_fin_v3bf.py & one_fin_v3bi.py：绘图主程序 & pior EKF only主程序 & posterior EKF主程序

#### 	其他文件：

​		output.npy：来自CJ的代码部分，似乎没有用

## program running

- add fcl_python fun fun2 module path
 ```bash
  export PYTHONPATH=to-your-ekf_V5_offline-folder:$PYTHONPATH
 ```
- run one_finger_contact
 ```bash
  cd One_Fin_Program
  python3 one_fin_v3bf.py
  python3 one_fin_v3bi.py
  ```
- run two_finger_contact
 ```bash
  cd Two_Fin_Program
  python3 mul_fin_v3bf.py
  python3 mul_fin_v3bi.py
  ```
- run in_hand_contact
 ```bash
  cd In_hand_Program
  python3 in_hand_v3bf.py
  python3 in_hand_v3bi.py
  ```
  
- run fingers configurable object pose estimation
 ```bash
  cd ekf_obj_pose
  python3 verification_main.py data.xml
  ```
- want to answer the following questions

  --how many fingers are needed in order to estimate the pose correctly
  -- some parameters in the estimation 
     1. initialized error (object pose error and contact error in the object frame)
     2. measurement noise contact position/normal(taxel)
     3. is force controller of fingers helpful for the estimation procedure?
  -- how to track the pose of movable object (pushed by fingers)
  -- how to control fingers in order to grasp object robustly.




