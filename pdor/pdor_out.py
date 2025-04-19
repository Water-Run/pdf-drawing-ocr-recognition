r"""
PDOR输出
:author: WaterRun
:time: 2025-04-20
:file: pdor_out.py
"""

import os
import json
import yaml
import toml
import xml.dom.minidom
import simpsave as ss

from enum import Enum
from typing import Dict, Any

from .pdor_unit import PdorUnit
from .pdor_exception import *


class PdorOut:

    r"""
    Pdor输出静态类
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
        PYTHON = 'python'

    @staticmethod
    def out(pdor: PdorUnit, out_type: TYPE = TYPE.SIMPSAVE, *, print_repr: bool = True) -> None:
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

        result = pdor.result

        match out_type:
            case PdorOut.TYPE.SIMPSAVE:
                output_file = f"{base_name}.ini"
                ss.write("Pdor Result", result, file=output_file)
                if print_repr:
                    print(f'{pdor.file}的结果输出至{output_file}的键`Pdor Result`.\n'
                          f'读取代码示例: \n'
                          f'import simpsave as ss\n'
                          f'ss.read("Pdor Result", file="{output_file}")')

            case PdorOut.TYPE.JSON:
                output_file = f"{base_name}.json"
                with open(output_file, 'w', encoding='utf-8') as f:
                    json.dump(result, f, ensure_ascii=False, indent=2)
                if print_repr:
                    print(f'{pdor.file}的结果输出至{output_file}.\n'
                          f'读取代码示例: \n'
                          f'import json\n'
                          f'with open("{output_file}", "r", encoding="utf-8") as f:\n'
                          f'    data = json.load(f)')

            case PdorOut.TYPE.YAML:
                output_file = f"{base_name}.yaml"
                with open(output_file, 'w', encoding='utf-8') as f:
                    yaml.dump(result, f, allow_unicode=True, default_flow_style=False)
                if print_repr:
                    print(f'{pdor.file}的结果输出至{output_file}.\n'
                          f'读取代码示例: \n'
                          f'import yaml\n'
                          f'with open("{output_file}", "r", encoding="utf-8") as f:\n'
                          f'    data = yaml.safe_load(f)')

            case PdorOut.TYPE.XML:
                output_file = f"{base_name}.xml"
                PdorOut._write_xml(result, output_file)
                if print_repr:
                    print(f'{pdor.file}的结果输出至{output_file}.\n'
                          f'读取代码示例: \n'
                          f'import xml.etree.ElementTree as ET\n'
                          f'tree = ET.parse("{output_file}")\n'
                          f'root = tree.getroot()')

            case PdorOut.TYPE.TOML:
                output_file = f"{base_name}.toml"
                with open(output_file, 'w', encoding='utf-8') as f:
                    toml.dump(result, f)
                if print_repr:
                    print(f'{pdor.file}的结果输出至{output_file}.\n'
                          f'读取代码示例: \n'
                          f'import toml\n'
                          f'with open("{output_file}", "r", encoding="utf-8") as f:\n'
                          f'    data = toml.load(f)')

            case PdorOut.TYPE.PYTHON:
                output_file = f"{base_name}.py"
                with open(output_file, 'w', encoding='utf-8') as f:
                    f.write(f"# 由PDOR生成的Python数据文件\n\n")
                    f.write(f"data = {repr(result)}\n")
                if print_repr:
                    print(f'{pdor.file}的结果输出至{output_file}.\n'
                          f'读取代码示例: \n'
                          f'import {os.path.basename(base_name)}\n'
                          f'data = {os.path.basename(base_name)}.data')

            case PdorOut.TYPE.MARKDOWN:
                output_file = f"{base_name}.md"
                PdorOut._write_markdown(result, output_file)
                if print_repr:
                    print(f'{pdor.file}的结果输出至{output_file}.\n'
                          f'可以使用任何Markdown查看器打开该文件查看.')

            case PdorOut.TYPE.PLAIN_TEXT:
                output_file = f"{base_name}.txt"
                PdorOut._write_plaintext(result, output_file)
                if print_repr:
                    print(f'{pdor.file}的结果输出至{output_file}.\n'
                          f'可以使用任何文本编辑器打开该文件查看.')

            case _:
                raise PdorOutUnsupportedTypeError(
                    message=f'不支持的输出类型: {out_type}'
                )

    @staticmethod
    def _format_value(value: Any, indent: int = 0) -> str:
        r"""
        格式化输出值，对嵌套字典进行美观处理
        :param value: 要格式化的值
        :param indent: 缩进级别
        :return: 格式化后的字符串
        """
        indent_str = "  " * indent

        if isinstance(value, dict):
            if not value:
                return "{}"

            result = "{\n"
            for k, v in value.items():
                result += f"{indent_str}  {k}: {PdorOut._format_value(v, indent + 1)},\n"
            result += f"{indent_str}}}"
            return result
        elif isinstance(value, list):
            if not value:
                return "[]"

            result = "[\n"
            for item in value:
                result += f"{indent_str}  {PdorOut._format_value(item, indent + 1)},\n"
            result += f"{indent_str}]"
            return result
        elif isinstance(value, str):
            return f'"{value}"'
        else:
            return str(value)

    @staticmethod
    def _write_xml(data: Dict[str, Any], filename: str) -> None:
        r"""
        将数据以XML格式写入文件，支持嵌套字典
        """
        doc = xml.dom.minidom.getDOMImplementation().createDocument(None, "pdor_result", None)
        root = doc.documentElement

        def add_dict_to_element(element, data_dict):
            for key, value in data_dict.items():
                if isinstance(value, dict):
                    child = doc.createElement(str(key))
                    element.appendChild(child)
                    add_dict_to_element(child, value)
                elif isinstance(value, list):
                    child = doc.createElement(str(key))
                    element.appendChild(child)
                    for i, item in enumerate(value):
                        if isinstance(item, dict):
                            item_elem = doc.createElement(f"item_{i}")
                            child.appendChild(item_elem)
                            add_dict_to_element(item_elem, item)
                        else:
                            item_elem = doc.createElement(f"item_{i}")
                            item_elem.appendChild(doc.createTextNode(str(item)))
                            child.appendChild(item_elem)
                else:
                    child = doc.createElement(str(key))
                    child.appendChild(doc.createTextNode(str(value)))
                    element.appendChild(child)

        add_dict_to_element(root, data)

        with open(filename, 'w', encoding='utf-8') as f:
            f.write(doc.toprettyxml(indent="  "))

    @staticmethod
    def _flatten_dict(data: Dict[str, Any], prefix: str = "") -> Dict[str, Any]:
        r"""
        将嵌套字典展平为扁平结构
        :param data: 嵌套字典
        :param prefix: 前缀（用于递归）
        :return: 展平后的字典
        """
        items = {}
        for k, v in data.items():
            key = f"{prefix}.{k}" if prefix else k

            if isinstance(v, dict):
                items.update(PdorOut._flatten_dict(v, key))
            else:
                items[key] = v

        return items

    @staticmethod
    def _dict_to_markdown(data: Dict[str, Any], level: int = 0) -> str:
        r"""
        将字典转换为Markdown格式，支持嵌套
        """
        indent = "  " * level
        md_content = ""

        for key, value in data.items():
            md_content += f"{indent}- **{key}**: "

            if isinstance(value, dict):
                md_content += "\n" + PdorOut._dict_to_markdown(value, level + 1)
            elif isinstance(value, list):
                md_content += "\n"
                for item in value:
                    if isinstance(item, dict):
                        md_content += f"{indent}  - 项目:\n{PdorOut._dict_to_markdown(item, level + 2)}"
                    else:
                        md_content += f"{indent}  - {item}\n"
            else:
                md_content += f"{value}\n"

        return md_content

    @staticmethod
    def _write_markdown(data: Dict[str, Any], filename: str) -> None:
        r"""
        将数据以Markdown格式写入文件，支持嵌套字典
        """
        md_content = "# PDOR识别结果\n\n"
        md_content += PdorOut._dict_to_markdown(data)

        # 写入文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(md_content)

    @staticmethod
    def _dict_to_plaintext(data: Dict[str, Any], level: int = 0) -> str:
        r"""
        将字典转换为纯文本格式，支持嵌套
        """
        indent = "  " * level
        text_content = ""

        for key, value in data.items():
            text_content += f"{indent}{key}: "

            if isinstance(value, dict):
                text_content += "\n" + PdorOut._dict_to_plaintext(value, level + 1)
            elif isinstance(value, list):
                if not value:
                    text_content += "[]\n"
                else:
                    text_content += "\n"
                    for index, item in enumerate(value):
                        if isinstance(item, dict):
                            text_content += f"{indent}  [{index}]:\n{PdorOut._dict_to_plaintext(item, level + 2)}"
                        else:
                            text_content += f"{indent}  [{index}]: {item}\n"
            else:
                text_content += f"{value}\n"

        return text_content

    @staticmethod
    def _write_plaintext(data: Dict[str, Any], filename: str) -> None:
        r"""
        将数据以纯文本格式写入文件，支持嵌套字典
        """
        text_content = "PDOR识别结果\n"
        text_content += "=" * 50 + "\n\n"
        text_content += PdorOut._dict_to_plaintext(data)

        # 写入文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text_content)
