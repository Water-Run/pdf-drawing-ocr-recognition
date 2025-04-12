r"""
PDOR输出
:author: WaterRun
:time: 2025-04-12
:file: pdor_out.py
"""

import simpsave as ss

from enum import Enum

from pdor_unit import PdorUnit
from pdor_exception import *


class PdorOut:
    r"""
    Pdor输出
    """

    class TYPE(Enum):
        r"""
        输出类型枚举
        """
        PLAIN_TEXT = 'plaintext'
        MARKDOWN = 'markdown'
        SIMPSAVE = 'simpsave'
        JSON = 'json'
        YAML = 'yaml'
        XML = 'xml'
        TOML = 'toml'
        CSV = 'csv'
        XLSX = 'xlsx'
        HTML = 'html'
        PYTHON = 'python'

    @staticmethod
    def out(pdor: PdorUnit, out_type: TYPE, *, print_repr: bool = False) -> None:
        r"""
        输出Pdor单元. 输出的文件名称和构造的PDF保持一致.
        :param pdor: 待输出的Pdor单元
        :param out_type: 输出的类型
        :param print_repr: 回显功能开关
        :return: None
        """
        if not pdor.is_parsed():
            raise PdorUnparsedError(
                message='无法进行输出'
            )

        base_name = pdor.file
        if base_name.lower().endswith('.pdf'):
            dot_pos = base_name.rfind('.')
            base_name = base_name[:dot_pos]

        match out_type:
            case PdorOut.TYPE.SIMPSAVE:
                output_file = f"{base_name}.ini"
                ss.write("Pdor Result", pdor.result, file=output_file)
                if print_repr:
                    print(f'{pdor.file}的结果输出至{output_file}的键`Pdor Result`.\n'
                          f'读取代码示例: \n'
                          f'import simpsave as ss\n'
                          f'ss.read("Pdor Result", file="{output_file}")')
