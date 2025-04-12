r"""
PDOR工具

:author: WaterRun
:time: 2025-04-13
:file: pdor_utils.py
"""

import os
import platform

from pdor.pdor_llm import check_connection


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
        # 检查API访问
        if not check_connection():
            missing_components.append('LLM不可访问')

    if missing_components:
        status = False

    return [status, missing_components]


def switch_api(api: str) -> bool:
    r"""
    更改API,仅在通过自检时可以更改.
    :param api: 待修改的API
    :return: 写入情况
    """
    if not check_env()[0]:
        return False
    import simpsave as ss
    return ss.write('api', api, file=_get_config_path())


def _get_config_path() -> str:
    r"""
    返回配置文件路径
    :return: 配置文件路径
    """
