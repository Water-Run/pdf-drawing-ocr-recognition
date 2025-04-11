r"""
PDOR环境检查

:author: WaterRun
:time: 2025-04-11
:file: pdor_env_check.py
"""

import os
import subprocess
import platform


def check_env() -> list[bool, list[str]]:
    r"""
    检查当前环境是否支持Pdor运行.
    检查内容包括：
    1. 必要Python库是否已安装
    2. Tesseract OCR是否已安装(默认路径)
    3. 必要的语言包（中文简体和英语）是否已安装

    示例使用:
    status, msg = check_env()
    status == True => 检查通过
    msg => 缺失的组件名称字符串列表

    :return: 一个列表.第一项是一个布尔值,表示检查是否通过.第二项是一个列表,表示出现异常的信息.
    """
    required_libraries = ["simpsave", "PyPDF2", "numpy", "pytesseract", "cv2", "pdf2image"]
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

    # 检查Tesseract OCR是否已安装
    tesseract_installed = False
    tesseract_path = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

    # 检查默认路径
    if os.path.isfile(tesseract_path):
        tesseract_installed = True
    else:
        # 尝试在PATH中查找
        try:
            result = subprocess.run(["tesseract", "--version"],
                                   stdout=subprocess.PIPE,
                                   stderr=subprocess.PIPE,
                                   text=True,
                                   shell=True,
                                   creationflags=subprocess.CREATE_NO_WINDOW)
            if result.returncode == 0:
                tesseract_installed = True
                tesseract_path = "tesseract"  # 在PATH中
        except (subprocess.SubprocessError, FileNotFoundError):
            pass

    if not tesseract_installed:
        missing_components.append("Tesseract OCR未安装或未添加到PATH")
    else:
        # 检查必要的语言包
        tessdata_dir = os.path.join(os.path.dirname(tesseract_path), "tessdata") if tesseract_path != "tesseract" else None

        # 如果在PATH中，尝试常见的tessdata位置
        if tessdata_dir is None:
            possible_dirs = [
                r"C:\Program Files\Tesseract-OCR\tessdata",
                r"C:\Program Files (x86)\Tesseract-OCR\tessdata",
                os.path.join(os.path.expanduser("~"), "AppData", "Local", "Tesseract-OCR", "tessdata")
            ]
            for dir_path in possible_dirs:
                if os.path.isdir(dir_path):
                    tessdata_dir = dir_path
                    break

        if tessdata_dir and os.path.isdir(tessdata_dir):
            required_language_files = ["eng.traineddata", "chi_sim.traineddata"]
            for lang_file in required_language_files:
                if not os.path.isfile(os.path.join(tessdata_dir, lang_file)):
                    missing_components.append(f"Tesseract语言包: {lang_file}")
        else:
            missing_components.append("无法找到Tesseract语言包目录")

    if missing_components:
        status = False

    return [status, missing_components]
