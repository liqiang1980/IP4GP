# tactile sim to real

## install pykdl
- https://github.com/orocos/orocos_kinematics_dynamics
    - install orocos_kdl 
        
        read related INSTALL.md at folder  3dparty/orocos_kinematics_dynamics/orocos_kdl/INSTALL.md
    - install python_orocos_kdl
        
        read related INSTALL.md at folder  3dparty/orocos_kinematics_dynamics/python_orocos_kdl/INSTALL.md

        - Note: Using cmake with python3

            cmake -DCMAKE_BUILD_TYPE=Release -DPYTHON_VERSION=3 ..

## install pykdl_utils
depandence PyKDL

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
