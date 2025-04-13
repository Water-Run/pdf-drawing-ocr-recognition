r"""
PDOR工具

:author: WaterRun
:time: 2025-04-13
:file: pdor_utils.py
"""

import os
import platform


def check_env() -> list[bool, list[str]]:
    r"""
    检查当前环境是否支持Pdor运行.
    检查内容包括：
    1. 必要Python库是否已安装
    2. Tesseract OCR是否已安装(默认路径)
    3. 必要的语言包（中文简体和英语）是否已安装
    4. LLM是否可正常访问

    示例使用:
    status, msg = check_env()
    status == True => 检查通过
    msg => 缺失的组件名称字符串列表

    :return: 一个列表.第一项是一个布尔值,表示检查是否通过.第二项是一个列表,表示出现异常的信息.
    """
    required_libraries = (
        "os",
        "platform",
        "subprocess",
        "simpsave",
        "requests",
        "json",
        "csv",
        "yaml",
        "toml",
        "html",
        "xml",
        "gc",
        "cv2",
        "shutil",
        "inspect",
        "tempfile",
        "numpy",
        "PyPDF2",
        "pdf2image",
    )

    missing_components = []
    status = True

    # 检查平台
    if not platform.system().lower() == 'windows':
        missing_components.append('非Windows平台')

    # 检查Python库
    for lib in required_libraries:
        try:
            __import__(lib)  # 动态导入库
        except ImportError:
            missing_components.append(f"库缺失: {lib}")

    # 检查配置文件
    if not os.path.exists('configs.ini'):
        missing_components.append("缺失配置文件configs.ini")
    else:
        if "库缺失: simpsave" not in missing_components:
            import simpsave as ss
            try:
                ss.read('api', file='configs.ini')
            except KeyError:
                missing_components.append("配置文件缺失必备键`api`")

    if missing_components:
        status = False

    return [status, missing_components]


def get_config_path() -> str:
    r"""
    返回配置文件路径，保证在被其他目录导入时路径不变
    :return: 配置文件路径
    """
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_filename = "configs.ini"
    return os.path.join(current_dir, config_filename)
