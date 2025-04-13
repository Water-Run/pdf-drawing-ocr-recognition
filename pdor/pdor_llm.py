r"""
Pdor LLM交互
:author: WaterRun
:time: 2025-04-13
:file: pdor_llm.py
"""

import base64
import requests
import simpsave as ss

from pdor_utils import get_config_path, check_env


def get_img_result(prompt: str, img: str) -> str:
    """
    发送本地的图片链接和Prompt到API并返回结果

    :param prompt: 发送的Prompt
    :param img: 本地图片路径
    :return: API返回的结果字符串
    """
    # API调用地址 - 使用已验证可用的端点
    api_url = "https://api.nuwaapi.com/v1/chat/completions"

    try:

        with open(img, "rb") as img_file:
            image_data = img_file.read()
            base64_image = base64.b64encode(image_data).decode('utf-8')

            # 构建消息格式，包含图像内容
            payload = {
                "model": "gpt-4-vision-preview",  # 尝试使用视觉模型
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
                "max_tokens": 500
            }

            # 添加必要的headers
            headers = {
                "Authorization": f"Bearer {ss.read('api', file=get_config_path())}",
                "Content-Type": "application/json"
            }

            # 发送POST请求
            response = requests.post(api_url, json=payload, headers=headers)

            # 检查响应是否成功
            if response.status_code == 200:
                # 解析JSON响应
                result = response.json()
                if "choices" in result and len(result["choices"]) > 0:
                    return result["choices"][0]["message"]["content"]
                else:
                    return "No result found in response."
            else:
                return f"Error: {response.status_code}, {response.text}"
    except FileNotFoundError:
        return "Error: 图片文件未找到，请检查路径是否正确。"
    except Exception as e:
        return f"Error: {str(e)}"


def check_connection() -> bool:
    """
    检查大模型是否可用，并打印诊断信息

    :return: 大模型是否可用的布尔值
    """
    # 测试不同的常见API端点
    api_endpoints = [
        "https://api.nuwaapi.com/v1/chat/completions",
        "https://api.openai.com/v1/chat/completions",
        "https://api.deepseek.com/v1/chat/completions"
    ]

    api_key = ss.read('api', file=get_config_path())

    for endpoint in api_endpoints:
        try:

            payload = {
                "model": "gpt-3.5-turbo",  # 使用常见模型名称
                "messages": [{"role": "user", "content": "测试"}],
                "max_tokens": 5
            }

            headers = {
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json"
            }

            response = requests.post(endpoint, json=payload, headers=headers, timeout=5)

            if response.status_code == 200:
                return True

        except Exception as e:
            return False

    return False


def switch_api(api: str) -> bool:
    r"""
    更改API,仅在通过自检时可以更改.
    :param api: 待修改的API
    :return: 写入情况
    """
    if not check_env()[0]:
        return False
    return ss.write('api', api, file=get_config_path())


if __name__ == "__main__":
    print(check_connection())
