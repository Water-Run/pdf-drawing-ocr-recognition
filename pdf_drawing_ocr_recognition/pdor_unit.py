r"""
PDOR单元
:author: WaterRun
:time: 2025-04-11
:file: pdor_unit.py
"""

import os
import simpsave as ss
import numpy as np
import inspect

from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from PIL import Image

from pdor_exception import *


class PdorUnit:

    r"""
    Pdor单元,构造后只读.
    构造需要对应PDF文件的路径,调用parse()方法执行解析.
    解析结果存储在result属性中,为一个字典.使用output()方法输出至simpsave文件.
    :param file: 用于构造的PDF文件名
    """

    def __init__(self, file: str):
        self._file_name = file
        self._pdf = None
        self._img = None
        self._result = None
        # 添加一个内部标志，用于区分初始化阶段和后续操作
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
        进行OCR图像识别
        :return:
        """

    def parse(self) -> None:
        r"""
        执行解析.
        """
        self._load()
        self._imagify()

    def output(self) -> None:
        r"""
        输出结果至simpsave .ini文件.
        键为`Pdor Result`,
        文件为和输入PDF同路径同文件名.ini文件
        """
        if self._result is None:
            raise PdorUnparsedError()

        base_name = self._file_name
        if base_name.lower().endswith('.pdf'):
            dot_pos = base_name.rfind('.')
            base_name = base_name[:dot_pos]

        output_file = f"{base_name}.ini"

        ss.write("Pdor Result", self._result, file=output_file)

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
            raise PdorUnparsedError()
        return self._result

    def __repr__(self) -> str:
        r"""
        返回Pdor单元信息
        :return: Pdor单元信息
        """
        return (f"Pdor单元: \n"
                f"文件名: {self._file_name}\n"
                f"PDF: {'未读取' if not self._pdf else '已读取'}\n"
                f"图片化: {'未转换' if not self._img else '已转换'}\n")

    def __setattr__(self, name, value):
        r"""
        属性设置拦截器，保证对象的只读特性
        :param name: 属性名
        :param value: 属性值
        :raise PdorAttributeModificationError: 如果在初始化后尝试修改受保护属性
        """
        if not hasattr(self, '_initialized') or name == '_initialized' or name not in {"_file_name", "_pdf", "_img", "_result"}:
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
        if name in {"_file_name", "_pdf", "_img", "_result"}:
            raise PdorAttributeModificationError(
                message=name
            )
        super().__delattr__(name)


"""test"""
if __name__ == '__main__':
    from pdor_env_check import check_env
    print(check_env())
    pdor = PdorUnit("../tests/700501-8615-72-12 750kV 第四串测控柜A+1端子排图左.PDF")
    pdor.parse()
    print(pdor)
    pdor._result = "123"