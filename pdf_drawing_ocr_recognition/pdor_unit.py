"""
PDOR单元
:author: WaterRun
:time: 2025-04-11
:file: pdor_unit.py
"""

import os
import simpsave as ss

from PyPDF2 import PdfReader
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

    def _load(self):
        r"""
        载入PDF文件
        :raise PdorPDFNotExistError: 如果PDF文件不存在
        :raise PdorPDFReadError: 如果PDF读取异常
        :return: None
        """
        if not os.path.isfile(self._file_name):
            raise PdorPDFNotExistError(self._file_name)
        try:
            reader = PdfReader(self._file_name)
            self._pdf = [page.extract_text() for page in reader.pages]  # 提取每一页的文本内容
        except Exception as error:
            raise PdorPDFReadError(str(error))

    def _imagify(self):
        r"""
        将读出的PDF转为图片
        :return: None
        """

    def parse(self) -> None:
        r"""
        执行解析.
        """
        self._load()

    def output(self, file_name: str) -> None:
        r"""
        输出结果至simpsave .ini文件, 键为PDF文件名
        :param file_name: simpsave .ini文件名(注意`.ini`后缀)
        """
        if self._result is None:
            raise PdorUnparsedError("单元未解析")
        ss.write(self._file_name, self._result, file=file_name)

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
            raise PdorUnparsedError("单元未解析")
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
        if name in {"_file_name", "_pdf", "_img", "_result"} and hasattr(self, name):
            raise PdorAttributeModificationError(name)
        super().__setattr__(name, value)

    def __delattr__(self, name):
        if name in {"_file_name", "_pdf", "_img", "_result"}:
            raise PdorAttributeModificationError(name)
        super().__delattr__(name)
