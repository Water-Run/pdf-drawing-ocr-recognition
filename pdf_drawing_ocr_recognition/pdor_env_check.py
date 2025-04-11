r"""
PDOR环境检查

:author: WaterRun
:time: 2025-04-11
:file: pdor_env_check.py
"""


def check_env() -> list[bool, list[str]]:
    r"""
    检查当前环境是否支持Pdor运行.
    示例使用:
    status, msg = check_env()
    status == True => 检查通过
    msg => 缺失的库名称字符串

    :return: 一个列表.第一项是一个布尔值,表示检查是否通过.第二项是一个列表,表示缺失的库(如果有).
    """
    required_libraries = ["simpsave", "PyPDF2"]
    missing_libraries = []
    status = True

    for lib in required_libraries:
        try:
            __import__(lib)  # 动态导入库
        except ImportError:
            missing_libraries.append(lib)

    if missing_libraries:
        status = False

    return [status, missing_libraries]
