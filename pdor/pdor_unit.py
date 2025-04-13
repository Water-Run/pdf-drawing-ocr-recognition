r"""
PDOR单元
:author: WaterRun
:time: 2025-04-12
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
from pdor_llm import get_img_result, check_connection
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
        """
        使用Pattern的子图定义切分图片并保存到_temp目录
        :param print_repr: 打印回显
        :return:
        """
        debug_dir = "__pdor_cache__"
        os.makedirs(debug_dir, exist_ok=True)
        if print_repr:
            print(f'- 已构建缓存目录 {debug_dir}')

        sub_imgs = self._pattern.sub_imgs
        sub_img_paths = []

        try:
            # 处理图片切分
            for page_idx, img_array in enumerate(self._img):
                page_height, page_width, _ = img_array.shape

                original_path = f"{debug_dir}/page_{page_idx}_original.jpg"
                cv2.imwrite(original_path, cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY))
                if print_repr:
                    print(f"\t- 保存原始图片: {original_path}")

                for sub_idx, (top, bottom, left, right) in enumerate(sub_imgs):
                    y1 = max(0, min(page_height, int(page_height * (top / 100))))
                    y2 = max(0, min(page_height, int(page_height * (bottom / 100))))
                    x1 = max(0, min(page_width, int(page_width * (left / 100))))
                    x2 = max(0, min(page_width, int(page_width * (right / 100))))

                    sub_img = img_array[y1:y2, x1:x2]

                    sub_img_path = f"{debug_dir}/sub_{page_idx}_{sub_idx}.jpg"
                    cv2.imwrite(sub_img_path, cv2.cvtColor(sub_img, cv2.COLOR_RGB2GRAY))
                    sub_img_paths.append((sub_idx, sub_img_path))  # 保存子图序号和路径
                    if print_repr:
                        print(f"\t- 保存子图({sub_idx + 1}/{len(sub_imgs)}): {sub_img_path}")

            # 检查LLM连接
            if print_repr:
                print(f"- LLM OCR请求")
                print(f"\t- 检查LLM可用性")
            if not check_connection():
                if print_repr:
                    print(f"\t- 检查不通过, 重试")
                if not check_connection():
                    raise PdorLLMError('LLM连接检查未通过，请检查网络连接')

            # 解析LLM结果的函数
            def _parse_llm_result(llm_result: str) -> tuple[bool, dict]:
                """
                解析LLM OCR返回的结果
                :param llm_result: LLM返回的OCR识别结果字符串
                :return: 包含两个元素的元组：[是否解析成功的布尔值, 解析后的Python字典]
                """
                try:
                    # 检查返回内容是否表明有错误
                    if llm_result.startswith("Error:"):
                        return False, {"error": llm_result}

                    # 先尝试直接解析为字典（可能模型已经返回了格式化的JSON字符串）
                    import json
                    try:
                        # 检查是否已经是JSON格式
                        if llm_result.strip().startswith('{') and llm_result.strip().endswith('}'):
                            result_dict = json.loads(llm_result)
                            if isinstance(result_dict, dict):
                                return True, result_dict
                    except:
                        pass  # 如果不是JSON格式，继续其他解析方法

                    # 尝试提取字典部分
                    dict_content = llm_result

                    # 如果结果包含代码块，提取代码块内容
                    if "```python" in llm_result and "```" in llm_result:
                        dict_content = llm_result.split("```python")[1].split("```")[0].strip()
                    elif "```json" in llm_result and "```" in llm_result:
                        dict_content = llm_result.split("```json")[1].split("```")[0].strip()
                    elif "```" in llm_result and "```" in llm_result:
                        dict_content = llm_result.split("```")[1].split("```")[0].strip()

                    # 确保字典内容开始和结束是大括号
                    dict_content = dict_content.strip()
                    if not (dict_content.startswith('{') and dict_content.endswith('}')):
                        # 如果不是字典格式，尝试创建一个简单的文本字典
                        return True, {"text": llm_result.strip()}

                    # 使用ast.literal_eval安全地将字符串转换为字典
                    import ast
                    result_dict = ast.literal_eval(dict_content)

                    # 验证结果是否是字典类型
                    if not isinstance(result_dict, dict):
                        return True, {"text": llm_result.strip()}

                    return True, result_dict

                except Exception as e:
                    # 如果解析过程中出现任何异常，尝试返回纯文本结果
                    return True, {"text": llm_result.strip()}

            # 识别所有子图
            results = []
            model_tried = ["gpt-4-vision-preview"]  # 已尝试的模型列表

            # for sub_idx, sub_img_path in sub_img_paths:
            #     MAX_RETRIES = 3
            #     for retry_count in range(1, MAX_RETRIES + 1):
            #         if print_repr:
            #             print(
            #                 f'\t- ({retry_count}/{MAX_RETRIES}) 尝试识别子图 #{sub_idx}: {os.path.basename(sub_img_path)}')
            #
            #         try:
            #             # 获取OCR结果
            #             llm_result = get_img_result(self._pattern.prompt, sub_img_path)
            #
            #             # 检查是否为模型不可用错误
            #             if "无可用渠道" in llm_result and retry_count == MAX_RETRIES:
            #                 # 最后一次重试尝试不同的模型名称
            #                 if print_repr:
            #                     print(f'\t\t- 模型不可用，尝试其他模型...')
            #
            #                 # 修改get_img_result中的模型名称并重试
            #                 import re
            #
            #                 # 读取函数内容并修改模型名称
            #                 get_img_result_src = inspect.getsource(get_img_result)
            #                 for model in ["gpt-4-vision", "gpt-4", "claude-3-opus-vision"]:
            #                     if model not in model_tried:
            #                         model_tried.append(model)
            #                         if print_repr:
            #                             print(f'\t\t- 尝试模型: {model}')
            #
            #                         # 临时修改函数中的模型名称
            #                         temp_get_img_result = re.sub(
            #                             r'"model": "([^"]+)"',
            #                             f'"model": "{model}"',
            #                             get_img_result_src
            #                         )
            #
            #                         # 执行修改后的函数
            #                         temp_func_namespace = {}
            #                         exec(temp_get_img_result, globals(), temp_func_namespace)
            #                         temp_get_img_result_func = temp_func_namespace["get_img_result"]
            #
            #                         # 使用临时函数获取结果
            #                         llm_result = temp_get_img_result_func(self._pattern.prompt, sub_img_path)
            #
            #                         # 如果没有错误，跳出循环
            #                         if not llm_result.startswith("Error:"):
            #                             break
            #
            #             # 检查是否为API错误
            #             if llm_result.startswith("Error:"):
            #                 if print_repr:
            #                     print(f'\t\t- API错误: {llm_result}. 重试中...')
            #                 continue
            #
            #             # 解析结果
            #             success, result_dict = _parse_llm_result(llm_result)
            #
            #             if success:
            #                 if print_repr:
            #                     print(f'\t\t- 识别成功')
            #                 results.append((sub_idx, result_dict))
            #                 break
            #             else:
            #                 if print_repr:
            #                     print(f'\t\t- 解析失败: {result_dict.get("error", "未知错误")}. 重试中...')
            #
            #         except Exception as e:
            #             if print_repr:
            #                 print(f'\t\t- 识别出错: {str(e)}. 重试中...')
            #     else:
            #         # 超出最大重试次数
            #         if print_repr:
            #             print(f'\t\t- 所有重试失败，使用原始文本作为结果')
            #         # 如果所有重试都失败，使用原始文本作为结果
            #         results.append((sub_idx, {"text": f"识别失败: 已尝试模型 {', '.join(model_tried)}"}))

            # 将子字典列表合并为一个大字典
            merged_dict = {}
            for sub_idx, result_dict in results:
                # 使用子图序号作为键名前缀
                prefix = f"sub_{sub_idx}"

                # 如果结果字典为空，跳过
                if not result_dict:
                    continue

                # 如果字典中只有一个值且键名为"text"，直接使用前缀作为键名
                if len(result_dict) == 1 and "text" in result_dict:
                    merged_dict[prefix] = result_dict["text"]
                else:
                    # 对于多个键值的情况，为每个键添加前缀
                    for key, value in result_dict.items():
                        merged_dict[f"{prefix}_{key}"] = value

            self._result = merged_dict

        except PdorLLMError as e:
            # 直接向上抛出自定义错误
            raise e
        except Exception as e:
            # 捕获所有其他异常，包装为PdorLLMError并提供详细信息
            raise PdorLLMError(f'OCR处理过程出错: {str(e)}')

        finally:
            # 无论成功与否，都清理临时目录
            if os.path.exists(debug_dir):
                shutil.rmtree(debug_dir)
                if print_repr:
                    print(f'- 已删除缓存目录 {debug_dir}')

    def parse(self, *, print_repr: bool = False) -> None:
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
        return self._result is None

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
        返回Pdor单元信息
        :return: Pdor单元信息
        """
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
            tables_info = "\n[提取的表格数据]\n"

            for page_key, page_tables in self._result.get('tables', {}).items():
                tables_info += f"\n=== {page_key} ===\n"

                for table_idx, table_data in enumerate(page_tables):
                    tables_info += f"\n  表格 #{table_idx + 1}: \n"

                    for row_id in sorted(table_data.keys(),
                                         key=lambda x: int(x.split('_')[1]) if x.startswith('Row_') and x.split('_')[
                                             1].isdigit() else float('inf')):
                        tables_info += f"\n    {row_id}: \n"
                        row_data = table_data[row_id]

                        if not row_data:
                            tables_info += "      (空行)\n"
                            continue

                        for col in sorted(row_data.keys(),
                                          key=lambda x: int(x.split('_')[1]) if x.startswith('Col_') and x.split('_')[
                                              1].isdigit() else 0):
                            cell_value = row_data[col]
                            tables_info += f"      {col}: '{cell_value}'\n"

            return base_info + tables_info

        return base_info


if __name__ == '__main__':
    # 示例使用
    # unit_1 = PdorUnit('../tests/700501-8615-72-12 750kV 第四串测控柜A+1端子排图左.PDF', load('端子排图左', 'pdor_patterns.ini'))
    # unit_1.parse(print_repr=True)
    # print(unit_1.result)

    # unit_3 = PdorUnit('../tests/700501-8615-73-04 第四串W4Q1断路器LCP柜接线图二.PDF',
    # load('700501-8615-73-04 第四串W4Q1断路器LCP柜接线图二'))
    # unit_3.parse()

    unit_3 = PdorUnit('../tests/duanzipai.pdf', load('duanzipai'))
    unit_3.parse(print_repr=True)
    print(unit_3)
