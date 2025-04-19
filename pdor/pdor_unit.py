r"""
PDOR单元
:author: WaterRun
:time: 2025-04-20
:file: pdor_unit.py
"""

import os
import gc
import cv2
import time
import shutil
import inspect
import tempfile
import numpy as np

from PyPDF2 import PdfReader
from pdf2image import convert_from_path

from pdor.pdor_pattern import PdorPattern, load
from pdor.pdor_llm import get_img_result, check_connection
from pdor.pdor_utils import parse_llm_result, get_max_try
from pdor.pdor_exception import *


class PdorUnit:
    r"""
    Pdor单元,构造后只读.
    构造需要对应PDF文件的路径,调用parse()方法执行解析.
    解析结果存储在result属性中,为一个字典.使用output()方法输出至simpsave文件.
    :param file: 用于构造的PDF文件名
    :param pattern: 用于构造的Pdor模式
    """

    def __init__(self, file: str, pattern: PdorPattern):
        self._file_name = file
        self._pattern = pattern
        self._pdf = None
        self._img = None
        self._time_cost = None
        self._result = None
        self._initialized = True

    def _is_internal_call(self):
        r"""
        判断当前调用是否来自类内部方法
        :return: 如果调用来自内部方法则返回True，否则返回False
        """
        frame = inspect.currentframe().f_back.f_back

        if frame is None:
            return False

        calling_self = frame.f_locals.get('self')

        is_internal = calling_self is self and frame.f_code.co_filename == __file__

        return is_internal

    def __setattr__(self, name, value):
        r"""
        属性设置拦截器，保证对象的只读特性
        :param name: 属性名
        :param value: 属性值
        :raise PdorAttributeModificationError: 如果在初始化后尝试修改受保护属性
        """
        if (not hasattr(self, '_initialized') or name == '_initialized' or name not in
                {"_file_name", "_pdf", "_img", "_result", "_time_cost", "_pattern"}):
            super().__setattr__(name, value)
        elif self._is_internal_call():
            super().__setattr__(name, value)
        else:
            raise PdorAttributeModificationError(
                message=name
            )

    def __delattr__(self, name):
        r"""
        属性删除拦截器，防止删除核心属性
        :param name: 要删除的属性名
        :raise PdorAttributeModificationError: 如果尝试删除受保护属性
        """
        if name in {"_file_name", "_pdf", "_img", "_result", "_time_cost", "_pattern"}:
            raise PdorAttributeModificationError(
                message=name
            )
        super().__delattr__(name)

    def _load(self, print_repr: bool):
        r"""
        载入PDF文件
        :param print_repr: 打印回显
        :raise PdorPDFNotExistError: 如果PDF文件不存在
        :raise PdorPDFReadError: 如果PDF读取异常
        :return: None
        """
        if not os.path.isfile(self._file_name):
            raise PdorPDFNotExistError(
                message=self._file_name
            )
        try:
            reader = PdfReader(self._file_name)
            self._pdf = [page.extract_text() for page in reader.pages]
            if print_repr:
                print('- PDF已读取并载入')
        except Exception as error:
            raise PdorPDFReadError(
                message=str(error)
            )

    def _imagify(self, print_repr: bool):
        r"""
        将读出的PDF转为图片
        :param print_repr: 打印回显
        :raise PdorImagifyError: 如果图片转换时出现异常
        :return: None
        """
        if self._pdf is None:
            raise PdorImagifyError(
                message="无可用的PDF实例"
            )

        try:
            if print_repr:
                print(f'- DPI: {self._pattern.dpi}')
            with tempfile.TemporaryDirectory() as temp_dir:
                start_time = time.time()
                images = convert_from_path(
                    self._file_name,
                    dpi=self._pattern.dpi,
                    thread_count=4,
                    use_cropbox=True,
                    output_folder=temp_dir,
                    fmt="jpeg",
                    jpegopt={"quality": 90, "optimize": True, "progressive": True}
                )

                convert_time = time.time() - start_time
                if print_repr:
                    print(f"- PDF转换耗时: {convert_time: .2f} s")
                    print(f"- 总页数: {len(images)}")

                self._img = []

                for i, image in enumerate(images):
                    if print_repr:
                        print(f"\t- 开始处理第 {i + 1} 页")
                    img_size = f"{image.width}x{image.height}"
                    if print_repr:
                        print(f"\t- 图像尺寸: {img_size}")

                    img_array = np.array(image)

                    self._img.append(img_array)

                    image.close()
                    gc.collect()

            if not self._img:
                raise PdorImagifyError(
                    message="无法从PDF中提取图像"
                )

        except Exception as error:
            raise PdorImagifyError(
                message=f"PDF图片化失败: {str(error)}"
            )

    def _ocr(self, print_repr: bool):
        r"""
        使用Pattern的子图定义切分图片并依次进行OCR,获取结果
        :param print_repr: 打印回显
        :return:
        """
        cache_dir = "__pdor_cache__"
        os.makedirs(cache_dir, exist_ok=True)
        if print_repr:
            print(f'- 已构建缓存目录 {cache_dir}')

        sub_imgs = self._pattern.sub_imgs
        sub_img_paths = []

        try:

            """子图切分"""

            for page_idx, img_array in enumerate(self._img):
                page_height, page_width, _ = img_array.shape

                original_path = f"{cache_dir}/page_{page_idx}_original.jpg"
                cv2.imwrite(original_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))

                if print_repr:
                    print(f"\t- 保存原始图片: {original_path}")
                    print(f"\t- 载入模式, 处理子图划分({len(sub_imgs)}张)")

                for sub_idx, (top, bottom, left, right) in enumerate(sub_imgs):

                    y1 = max(0, min(page_height, int(page_height * (top / 100))))
                    y2 = max(0, min(page_height, int(page_height * (bottom / 100))))
                    x1 = max(0, min(page_width, int(page_width * (left / 100))))
                    x2 = max(0, min(page_width, int(page_width * (right / 100))))

                    sub_img = img_array[y1:y2, x1:x2]

                    sub_img_path = f"{cache_dir}/sub_{page_idx}_{sub_idx}.jpg"
                    cv2.imwrite(sub_img_path, cv2.cvtColor(sub_img, cv2.COLOR_RGB2GRAY))
                    sub_img_paths.append((sub_idx, sub_img_path))
                    if print_repr:
                        print(f"\t- 保存子图({sub_idx + 1}/{len(sub_imgs)}): {sub_img_path}")

            """LLM OCR"""

            if print_repr:
                print(f"- LLM OCR请求")
                print(f"\t- 检查LLM可用性")

            if not check_connection():
                if print_repr:
                    print(f"\t- 检查不通过, 重试")
                if not check_connection():
                    raise PdorLLMError('LLM连接检查未通过, 检查连接')

            results = []

            for sub_idx, sub_img_path in sub_img_paths:
                MAX_RETRIES = get_max_try()

                for retry_count in range(1, MAX_RETRIES + 1):

                    if print_repr:
                        print(
                            f'\t- (尝试 {retry_count}/{MAX_RETRIES}) 识别子图 #{sub_idx}: {os.path.basename(sub_img_path)}')

                    try:
                        llm_result = get_img_result(self._pattern.prompt, sub_img_path)

                        if llm_result.startswith("Error:"):
                            if print_repr:
                                print(f'\t\t- API错误: {llm_result}. 进行重试')
                            continue

                        success, result_dict = parse_llm_result(llm_result)

                        if success:
                            if print_repr:
                                print(f'\t\t- LLM OCR结果成功解析')
                            results.append((sub_idx, result_dict))
                            break
                        else:
                            if print_repr:
                                print(f'\t\t- 解析失败: {result_dict.get("error", "未知错误")}. 重试中...')

                    except Exception as e:
                        if print_repr:
                            print(f'\t\t- 识别出错: {str(e)}. 重试中...')
                else:
                    if print_repr:
                        print(f'\t\t- 所有重试失败, 终止识别')
                        break

            if len(results) == 0:

                if print_repr:
                    print(f'- LLM OCR失败')
                raise PdorLLMError(
                    message='LLM OCR失败'
                )

            else:
                merged_dict = {}
                for sub_idx, result_dict in results:
                    prefix = f"sub_{sub_idx}"

                    if not result_dict:
                        continue

                    if len(result_dict) == 1 and "text" in result_dict:
                        merged_dict[prefix] = result_dict["text"]
                    else:
                        for key, value in result_dict.items():
                            merged_dict[f"{prefix}_{key}"] = value
                self._result = merged_dict

        except PdorLLMError as e:
            raise e
        except Exception as e:
            raise PdorLLMError(
                message=f'OCR处理过程接受异常: {str(e)}'
            )

        finally:
            if os.path.exists(cache_dir):
                shutil.rmtree(cache_dir)
                if print_repr:
                    print(f'- 已删除缓存目录 {cache_dir}')

    def parse(self, *, print_repr: bool = True) -> None:
        r"""
        执行解析
        :param print_repr: 是否启用回显
        """
        if self._result is not None:
            raise PdorParsedError(
                message='无法再次解析'
            )

        start = time.time()
        task_info_flow = (
            (lambda x: None, f'Pdor单元解析: {self._file_name}'),
            (self._load, '载入PDF...'),
            (self._imagify, 'PDF图片化...'),
            (self._ocr, 'OCR识别...'),
            (lambda x: None, f'解析完成: 访问result属性获取结果, 打印本单元获取信息, 调用PdorOut输出'),
        )

        for task, info in task_info_flow:
            if print_repr:
                print(info)
            task(print_repr)
        self._time_cost = time.time() - start

    def is_parsed(self) -> bool:
        r"""
        返回是否已经解析
        :return: 是否已经解析
        """
        return self._result is not None

    @property
    def file(self) -> str:
        r"""
        返回构造Pdor单元的PDF的文件名
        :return: PDF文件名
        """
        return self._file_name

    @property
    def result(self) -> dict:
        r"""
        返回Pdor结果
        :return: Pdor结果字典
        :raise PdorUnparsedError: 如果未解析
        """
        if self._result is None:
            raise PdorUnparsedError(
                message='无法访问属性`result`'
            )
        return self._result

    @property
    def pattern(self) -> PdorPattern:
        r"""
        返回Pdor模式
        :return: Pdor模式
        """
        return self._pattern

    @property
    def time_cost(self) -> float:
        r"""
        返回Pdor解析用时
        :return: 解析用时(s)
        :raise PdorUnparsedError: 如果未解析
        """
        if self._time_cost is None:
            raise PdorUnparsedError(
                message='无法访问属性`time_cost`'
            )
        return self._time_cost

    def __repr__(self) -> str:
        r"""
        返回 Pdor 单元信息
        :return: Pdor 单元信息
        """

        def format_dict(data, indent=0):
            """
            递归格式化字典内容为字符串，支持嵌套字典的结构化输出。

            :param data: 要格式化的对象，通常是字典
            :param indent: 当前缩进层级
            :return: 格式化后的字符串
            """
            formatted_str = ""
            spaces = " " * indent
            if isinstance(data, dict):
                for key, value in data.items():
                    formatted_str += f"{spaces}{key}: "
                    if isinstance(value, dict):
                        formatted_str += "\n" + format_dict(value, indent + 4)
                    elif isinstance(value, list):
                        formatted_str += "[\n"
                        for item in value:
                            if isinstance(item, dict):
                                formatted_str += format_dict(item, indent + 4)
                            else:
                                formatted_str += f"{' ' * (indent + 4)}{item}, \n"
                        formatted_str += f"{spaces}]\n"
                    else:
                        formatted_str += f"{value}\n"
            else:
                formatted_str += f"{spaces}{data}\n"
            return formatted_str

        base_info = (f"===Pdor单元===\n"
                     f"[构造信息]\n"
                     f"文件名: {self._file_name}\n"
                     f"模式: \n"
                     f"{self._pattern}\n"
                     f"[状态信息]\n"
                     f"PDF: {'已读取' if self._pdf else '未读取'}\n"
                     f"图片化: {'已转换' if self._img else '未转换'}\n"
                     f"LLM OCR: {'已处理' if self._result else '未处理'}\n"
                     f"耗时: {f'{self._time_cost: .2f} s' if hasattr(self, '_time_cost') and self._time_cost else '未解析'}")

        if self._result is not None:
            result_info = "\n[提取的表格数据]\n"
            if isinstance(self._result, dict) and self._result:
                result_info += format_dict(self._result, indent=4)
            else:
                result_info += "无表格数据\n"
            return base_info + result_info

        return base_info
