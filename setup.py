# from setuptools import setup
# from setuptools import Extension

# add_mat_module = Extension(
#     name='base',
#     sources=['base.cpp'],
#     include_dirs=[
#         r"D:\py\Python3\Lib\site-packages\eigen-3.4.0\eigen-3.4.0",
#         r"d:\py\python3\lib\site-packages\pybind11\include",
#         r"E:\BSP-OT-main",
#     ],
#     language="c++",
#     extra_compile_args=[
#         "/std:c++20",         # 关键：C++20
#         "/O2",
#         "/EHsc",
#         "/Zc:__cplusplus",    # 建议：让 __cplusplus 宏反映真实版本
#     ],
# )

# setup(ext_modules=[add_mat_module])


# #python setup.py build_ext --inplace
# #



import os
from setuptools import setup, Extension
import pybind11

# 获取当前 setup.py 所在的绝对路径
current_dir = os.path.dirname(os.path.abspath(__file__))

add_mat_module = Extension(
    name='base',
    sources=['base.cpp'],  # 确保这里和你的文件名一致
    include_dirs=[
        # 1. 自动获取当前环境下的 pybind11 路径
        pybind11.get_include(),
        
        # 2. 使用相对路径指向你放在根目录下的 BSP-OT 文件夹
        os.path.join(current_dir, 'BSP-OT-main'),
        
        # 3. Eigen 库路径 (建议让用户自己通过环境变量指定，或者给出默认相对路径)
        # 假设大部分人通过 conda 或标准路径安装，或者你可以把 eigen 也放进根目录
        os.environ.get("EIGEN_DIR", "/usr/include/eigen3") 
    ],
    language="c++",
    extra_compile_args=[
        "/std:c++20",         
        "/O2",
        "/EHsc",
        "/Zc:__cplusplus",    
    ],
)

setup(ext_modules=[add_mat_module])

# #python setup.py build_ext --inplace
