r"""
PDOR单元
:author: WaterRun
:time: 2025-04-11
:file: pdor_unit.py
"""

import os
import time
import simpsave as ss
import numpy as np
import inspect
import pytesseract
import cv2

from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from collections import defaultdict

from pdor_pattern import PdorPattern
from pdor_exception import *


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

    def _load(self):
        r"""
        载入PDF文件
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
        except Exception as error:
            raise PdorPDFReadError(
                message=str(error)
            )

    def _imagify(self):
        r"""
        将读出的PDF转为图片
        :raise PdorImagifyError: 如果图片转换时出现异常
        :return: None
        """
        if self._pdf is None:
            raise PdorImagifyError(
                message="无可用的PDF实例"
            )

        self._img = []
        for index, image in enumerate(convert_from_path(self._file_name)):
            img_array = np.array(image)
            self._img.append(img_array)

        if not self._img:
            raise PdorImagifyError(
                message="无法从PDF中提取图像"
            )

    def _ocr(self):
        r"""
        进行OCR图像识别，识别端子排表格内容
        :return: None
        :raise PdorOCRError: 如果OCR识别失败
        """
        # 设置Tesseract路径 - 添加这一行
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        if not self._img:
            raise PdorOCRError(message="没有可用的图像进行OCR")

        try:
            # 结果存储
            result_dict = {}

            for page_idx, img_array in enumerate(self._img):
                # 1. 预处理图像以增强OCR效果
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]

                # 2. 检测表格结构
                # 使用形态学操作和轮廓检测来识别表格
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
                dilate = cv2.dilate(thresh, kernel, iterations=3)
                contours, _ = cv2.findContours(dilate, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 筛选可能的表格轮廓
                table_contours = []
                for c in contours:
                    area = cv2.contourArea(c)
                    if area > 10000:  # 根据面积筛选可能的表格
                        table_contours.append(c)

                page_results = []

                # 3. 对每个可能的表格区域进行处理
                for table_idx, contour in enumerate(table_contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    table_img = img_array[y:y + h, x:x + w]

                    # 4. 使用pytesseract进行OCR识别
                    custom_config = r'--oem 3 --psm 6'
                    text = pytesseract.image_to_string(table_img, lang='chi_sim+eng', config=custom_config)

                    # 5. 使用pytesseract的表格识别功能
                    data = pytesseract.image_to_data(table_img, lang='chi_sim+eng', output_type=pytesseract.Output.DICT)

                    # 6. 根据识别结果构建结构化数据
                    # 创建行列结构
                    rows = defaultdict(dict)

                    # 处理识别结果
                    for i in range(len(data['text'])):
                        if data['text'][i].strip():
                            # 根据y坐标估计行号
                            row_idx = data['top'][i] // 20  # 假设每行高度约为20像素
                            # 根据x坐标估计列号
                            col_idx = data['left'][i] // 100  # 假设每列宽度约为100像素

                            # 存储文本
                            if col_idx not in rows[row_idx]:
                                rows[row_idx][col_idx] = data['text'][i]
                            else:
                                rows[row_idx][col_idx] += ' ' + data['text'][i]

                    # 7. 将行列数据转换为结构化字典
                    structured_table = {}
                    for row_idx, cols in rows.items():
                        # 第一列通常是标识符或端子号
                        key = cols.get(0, f"Row_{row_idx}")
                        # 其他列作为值
                        values = {f"Col_{col_idx}": value for col_idx, value in cols.items() if col_idx > 0}
                        structured_table[key] = values

                    page_results.append(structured_table)

                # 将当前页的结果添加到总结果中
                result_dict[f"Page_{page_idx + 1}"] = page_results

            # 更新结果
            self._result = {
                "file_name": os.path.basename(self._file_name),
                "total_pages": len(self._img),
                "tables": result_dict,
            }

        except Exception as error:
            raise PdorOCRError(message=f"OCR处理失败: {str(error)}")

    def parse(self, *, print_repr: bool = False) -> None:
        r"""
        执行解析.
        :param print_repr: 是否启用回显
        """
        if self._result is not None:
            raise PdorParsedError(
                message='无法再次解析'
            )

        start = time.time()
        task_info_flow = (
            (lambda: None, f'Pdor单元解析: {self._file_name}'),
            (self._load, '载入PDF...'),
            (self._imagify, 'PDF图片化...'),
            (self._ocr, 'OCR识别...'),
            (lambda: None, f'解析完成: 访问result属性获取结果, 打印本单元获取信息, 调用output()方法输出'),
        )

        for task, info in task_info_flow:
            task()
            if print_repr:
                print(info)
        self._time_cost = time.time() - start

    def output(self, *, print_repr: bool = True) -> None:
        r"""
        输出结果至simpsave .ini文件.
        键为`Pdor Result`,
        文件为和输入PDF同路径同文件名.ini文件
        :param print_repr: 是否启用回显
        """
        if self._result is None:
            raise PdorUnparsedError(
                message='无法输出至simpsave'
            )

        base_name = self._file_name
        if base_name.lower().endswith('.pdf'):
            dot_pos = base_name.rfind('.')
            base_name = base_name[:dot_pos]

        output_file = f"{base_name}.ini"

        ss.write("Pdor Result", self._result, file=output_file)

        if print_repr:
            print(f'{self._file_name}的结果输出至{output_file}的键`Pdor Result`.\n'
                  f'读取代码示例: \n'
                  f'import simpsave as ss\n'
                  f'ss.read("Pdor Result", file="{output_file}")')

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
                     f"模式: {self._pattern}\n"
                     f"[状态信息]\n"
                     f"PDF: {'已读取' if self._pdf else '未读取'}\n"
                     f"图片化: {'已转换' if self._img else '未转换'}\n"
                     f"耗时: {f'{self._time_cost: .2f} s' if hasattr(self, '_time_cost') and self._time_cost else '未解析'}")

        # 当且仅当self._result不为None时，添加表格数据输出
        if self._result is not None:
            tables_info = "\n[提取的表格数据]\n"

            # 遍历每一页数据
            for page_key, page_tables in self._result.get('tables', {}).items():
                tables_info += f"\n=== {page_key} ===\n"

                # 遍历该页的每个表格
                for table_idx, table_data in enumerate(page_tables):
                    tables_info += f"\n  表格 #{table_idx + 1}:\n"

                    # 调整显示方式：将行作为主要结构，每个单元格单独一行显示
                    for row_id in sorted(table_data.keys(),
                                         key=lambda x: int(x.split('_')[1]) if x.startswith('Row_') and x.split('_')[
                                             1].isdigit() else float('inf')):
                        tables_info += f"\n    {row_id}:\n"
                        row_data = table_data[row_id]

                        if not row_data:  # 跳过空行
                            tables_info += "      (空行)\n"
                            continue

                        # 按列顺序排序
                        for col in sorted(row_data.keys(),
                                          key=lambda x: int(x.split('_')[1]) if x.startswith('Col_') and x.split('_')[
                                              1].isdigit() else 0):
                            cell_value = row_data[col]
                            tables_info += f"      {col}: '{cell_value}'\n"

            return base_info + tables_info

        return base_info

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

