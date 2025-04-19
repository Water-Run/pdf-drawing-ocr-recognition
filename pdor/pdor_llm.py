r"""
Pdor LLM交互
:author: WaterRun
:time: 2025-04-20
:file: pdor_llm.py
"""

import base64
import requests
import simpsave as ss

from pdor_utils import get_config_path, get_api_url, get_api_key, get_llm_model


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
                        return result["choices"][0]["message"]["content"]
                    else:
                        return f"Error: 响应中未找到有效结果: {response.text[:150]}..."
                except ValueError as json_error:
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
