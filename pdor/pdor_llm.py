r"""
Pdor LLM交互
:author: WaterRun
:time: 2025-04-19
:file: pdor_llm.py
"""

import base64
import requests
import simpsave as ss

from pdor_utils import get_config_path, check_env


def get_img_result(prompt: str, img: str) -> str:
    r"""
    发送本地的图片链接和Prompt到API并返回结果

    :param prompt: 发送的Prompt
    :param img: 本地图片路径
    :return: API返回的结果字符串
    """
    api_url = get_api_url()

    try:
        with open(img, "rb") as img_file:
            image_data = img_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')

            # 构建消息格式，包含图像内容
            payload = {
                "model": get_llm_model(),  # 尝试使用视觉模型
                "messages": [
                    {
                        "role": "user",
                        "content": [
                            {"type": "text", "text": prompt},
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:image/jpeg;base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                "max_tokens": 8000
            }

            headers = {
                "Authorization": f"Bearer {ss.read('api key', file=get_config_path())}",
                "Content-Type": "application/json"
            }

            response = requests.post(api_url, json=payload, headers=headers)

            if response.status_code == 200:
                try:
                    result = response.json()
                    if "choices" in result and len(result["choices"]) > 0:
                        print(result["choices"][0]["message"]["content"])
                        return result["choices"][0]["message"]["content"]
                    else:
                        return f"Error: 响应中未找到有效结果: {response.text[:150]}..."
                except ValueError as json_error:
                    input(response.text)
                    return f"Error: JSON解析失败: {str(json_error)}, 原始响应: {response.text[:150]}..."
            else:
                return f"Error: 状态码 {response.status_code}, 响应: {response.text[:150]}..."
    except FileNotFoundError:
        return "Error: 图片文件未找到，检查路径"
    except Exception as e:
        return f"Error: 捕获其它异常 {str(e)}"


def check_connection() -> bool:
    r"""
    检查大模型是否可用

    :return: 大模型是否可用的布尔值
    """
    api_endpoints = [
        get_api_url(),
    ]

    api_key = get_api_key()

    for endpoint in api_endpoints:
        try:

            payload = {
                "model": get_llm_model(),
                "messages": [{"role": "user", "content": "我在测试我与你的链接.如果链接正常,请回复ok"}],
                "max_tokens": 100
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(endpoint, json=payload, headers=headers, timeout=5)

            if response.status_code == 200:

                if response.status_code == 200:

                    try:
                        result = response.json()
                        if "choices" in result and len(result["choices"]) > 0:
                            return 'ok' in result["choices"][0]["message"]["content"].lower()
                    except Exception as e:
                        return False

            return False

        except Exception as e:
            return False

    return False


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
