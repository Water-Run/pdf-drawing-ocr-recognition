from setuptools import setup, find_packages

# 读取 README.md 文件作为长描述
with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name='pdor',  # 包名
    version='0.2.0',  # 版本号
    install_requires=[
        'PyPDF2',
        'pdf2image',
        'opencv-python',  # cv2
        'numpy',
        'simpsave',
        'pandas',
        'openpyxl',
        'pyyaml',
        'toml',
        'requests',
    ],  # 根据代码需求列出的依赖库
    packages=find_packages(),  # 自动查找包
    author='WaterRun',  # 作者名
    author_email='2263633954@qq.com',  # 作者邮箱
    description='PDF OCR识别工具，用于自动文档提取和分析',  # 短描述
    long_description=long_description,  # 长描述
    long_description_content_type='text/markdown',  # 长描述类型
    url='https://github.com/Water-Run/pdor',  # 项目地址 (假设)
    classifiers=[
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.10',
        'License :: OSI Approved :: MIT License',  # 假设使用 MIT 协议
        'Operating System :: OS Independent',
        'Intended Audience :: Developers',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Topic :: Text Processing :: General',
        'Topic :: Scientific/Engineering :: Image Processing',
    ],
    python_requires='>=3.10',  # 根据代码中使用的现代语法限定 Python 版本
    include_package_data=True,  # 包含其他数据文件
    package_data={
        '': ['configs.ini'],  # 包含根目录下的 configs.ini 文件
    },
)
