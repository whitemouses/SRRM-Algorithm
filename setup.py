from setuptools import setup
from setuptools import Extension

add_mat_module = Extension(
    name='base',
    sources=['base.cpp'],
    include_dirs=[
        r"D:\py\Python3\Lib\site-packages\eigen-3.4.0\eigen-3.4.0",
        r"d:\py\python3\lib\site-packages\pybind11\include",
        r"E:\HCP-main\HCP-main\BSP-OT-main",
    ],
    language="c++",
    extra_compile_args=[
        "/std:c++20",         # 关键：C++20
        "/O2",
        "/EHsc",
        "/Zc:__cplusplus",    # 建议：让 __cplusplus 宏反映真实版本
    ],
)

setup(ext_modules=[add_mat_module])


#python setup.py build_ext --inplace
#
