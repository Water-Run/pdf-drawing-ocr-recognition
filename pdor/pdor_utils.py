r"""
PDOR工具

:author: WaterRun
:time: 2025-04-20
:file: pdor_utils.py
"""

import os
import ast
import platform
import simpsave as ss


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
    required_libraries = [
        'PyPDF2',
        'pdf2image',
        'opencv-python',
        'numpy',
        'simpsave',
        'pandas',
        'openpyxl',
        'pyyaml',
        'toml',
        'requests',
    ],

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
                api = ss.read('api', file='configs.ini')
                if 'sk-' not in api:
                    missing_components.append("配置文件api内容不合法")
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


def parse_llm_result(parse_llm: str) -> tuple[bool, dict]:
    r"""
    解析LLM OCR返回的结果
    :param parse_llm: LLM返回的OCR识别结果字符串
    :return: 包含两个元素的元组：[是否解析成功的布尔值, 解析后的Python字典]
    """

    if not parse_llm.count('{') == parse_llm.count('}'):
        # 如果左括号与右括号数量不相等，返回失败
        return False, {}

    # 查找第一个 '{' 和最后一个 '}'
    start_index = parse_llm.find('{')
    end_index = parse_llm.rfind('}')

    if start_index == -1 or end_index == -1 or start_index >= end_index:
        # 如果找不到有效的括号或起始索引大于等于结束索引，返回失败
        return False, {}

    try:
        # 截取字典部分，并使用 ast.literal_eval 解析为 Python 字典
        parsed_dict = ast.literal_eval(parse_llm[start_index:end_index + 1])
        if not isinstance(parsed_dict, dict):
            # 如果解析结果不是字典，返回失败
            return False, {}
        return True, parsed_dict
    except (SyntaxError, ValueError):
        # 捕获解析错误并返回失败
        return False, {}


def set_api_key(key: str) -> bool:
    r"""
    更改API KEY,仅在通过自检时可以更改.
    :param key: 待修改的API
    :return: 修改情况
    """
    if not check_env()[0]:
        return False
    if not isinstance(key, str):
        return False

    return ss.write('api key', key, file=get_config_path())


def set_api_url(url: str) -> bool:
    r"""
    更改API,仅在通过自检时可以更改.
    :param url: 待修改的API地址
    :return: 修改情况
    """
    if not check_env()[0]:
        return False
    if not isinstance(url, str):
        return False

    return ss.write('api url', url, file=get_config_path())


def set_llm_model(model: str) -> bool:
    r"""
    更改模型,仅在通过自检时可以更改.
    :param model: 使用的模型
    :return: 修改情况
    """
    if not check_env()[0]:
        return False
    if not isinstance(model, str):
        return False

    return ss.write('model', model, file=get_config_path())


def set_max_try(max_try: int) -> bool:
    r"""
    更改最大尝试次数(1-10),仅在通过自检时可以更改.
    :param max_try: 最大尝试次数
    :return: 修改情况
    """
    if not check_env()[0]:
        return False
    if not isinstance(max_try, int):
        return False
    if not 1 <= max_try <= 10:
        return False

    return ss.write('max try', max_try, file=get_config_path())


def get_api_url() -> str:
    r"""
    读取API地址
    :return: API地址
    """
    return ss.read('api url', file=get_config_path())


def get_api_key() -> str:
    r"""
    读取API KEY
    :return: API KEY
    """
    return ss.read('api key', file=get_config_path())


def get_llm_model() -> str:
    r"""
    读取当前使用的模型
    :return: 当前模型
    """
    return ss.read('model', file=get_config_path())


def get_max_try() -> int:
    r"""
    读取最大尝试次数
    :return: 最大尝试次数
    """
    return ss.read('max try', file=get_config_path())
