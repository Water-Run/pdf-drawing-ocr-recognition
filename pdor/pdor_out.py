r"""
PDOR输出
:author: WaterRun
:time: 2025-04-13
:file: pdor_out.py
"""

import os
import json
import csv
import yaml
import toml
import html
import xml.dom.minidom
import simpsave as ss
import pandas as pd

from enum import Enum
from typing import Dict, Any

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
    def out(pdor: PdorUnit, out_type: 'PdorOut.TYPE', *, print_repr: bool = False) -> None:
        r"""
        输出Pdor单元. 输出的文件名称和构造的PDF保持一致.
        :param pdor: 待输出的Pdor单元
        :param out_type: 输出的类型
        :param print_repr: 回显功能开关
        :return: None
        """
        if pdor.is_parsed():
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

            case PdorOut.TYPE.CSV:
                output_file = f"{base_name}.csv"
                PdorOut._write_csv(result, output_file)
                if print_repr:
                    print(f'{pdor.file}的结果输出至{output_file}.\n'
                          f'读取代码示例: \n'
                          f'import csv\n'
                          f'with open("{output_file}", "r", encoding="utf-8") as f:\n'
                          f'    reader = csv.DictReader(f)\n'
                          f'    data = [row for row in reader]')

            case PdorOut.TYPE.XLSX:
                output_file = f"{base_name}.xlsx"
                PdorOut._write_xlsx(result, output_file)
                if print_repr:
                    print(f'{pdor.file}的结果输出至{output_file}.\n'
                          f'读取代码示例: \n'
                          f'import pandas as pd\n'
                          f'data = pd.read_excel("{output_file}", sheet_name=None)')

            case PdorOut.TYPE.HTML:
                output_file = f"{base_name}.html"
                PdorOut._write_html(result, output_file)
                if print_repr:
                    print(f'{pdor.file}的结果输出至{output_file}.\n'
                          f'可以使用任何浏览器打开该文件查看.')

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
    def _write_xml(data: Dict[str, Any], filename: str) -> None:
        """
        将数据以XML格式写入文件
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
    def _write_csv(data: Dict[str, Any], filename: str) -> None:
        """
        将数据以CSV格式写入文件
        注意：由于CSV是扁平结构，这里会将嵌套结构展平
        """
        # 提取所有表格数据
        all_rows = []

        for page_key, page_tables in data.get('tables', {}).items():
            for table_idx, table_data in enumerate(page_tables):
                for row_id, row_data in table_data.items():
                    # 创建一个包含页面和表格信息的行
                    flat_row = {
                        'page': page_key,
                        'table': f"Table_{table_idx}",
                        'row_id': row_id
                    }
                    # 添加行数据
                    flat_row.update(row_data)
                    all_rows.append(flat_row)

        if all_rows:
            # 获取所有可能的列名
            all_fields = set()
            for row in all_rows:
                all_fields.update(row.keys())

            # 确保关键列在前面
            fieldnames = ['page', 'table', 'row_id']
            for field in sorted(all_fields):
                if field not in fieldnames:
                    fieldnames.append(field)

            # 写入CSV
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(all_rows)
        else:
            # 处理没有表格数据的情况
            with open(filename, 'w', newline='', encoding='utf-8') as csvfile:
                writer = csv.writer(csvfile)
                writer.writerow(['file_name', 'total_pages'])
                writer.writerow([data.get('file_name', ''), data.get('total_pages', 0)])

    @staticmethod
    def _write_xlsx(data: Dict[str, Any], filename: str) -> None:
        """
        将数据以XLSX格式写入文件
        会为每个表格创建一个工作表
        """
        with pd.ExcelWriter(filename, engine='openpyxl') as writer:
            # 创建概览工作表
            overview_data = {
                'file_name': [data.get('file_name', '')],
                'total_pages': [data.get('total_pages', 0)]
            }
            pd.DataFrame(overview_data).to_excel(writer, sheet_name='概览', index=False)

            # 为每个页面和表格创建工作表
            for page_key, page_tables in data.get('tables', {}).items():
                for table_idx, table_data in enumerate(page_tables):
                    # 转换表格数据为DataFrame
                    table_rows = []
                    for row_id, row_data in table_data.items():
                        row = {'row_id': row_id}
                        row.update(row_data)
                        table_rows.append(row)

                    if table_rows:
                        df = pd.DataFrame(table_rows)
                        sheet_name = f"{page_key}_Table{table_idx}"
                        # Excel限制工作表名称长度为31个字符
                        if len(sheet_name) > 31:
                            sheet_name = sheet_name[:31]
                        df.to_excel(writer, sheet_name=sheet_name, index=False)

    @staticmethod
    def _write_html(data: Dict[str, Any], filename: str) -> None:
        """
        将数据以HTML格式写入文件
        创建一个带有CSS样式的HTML文件
        """
        # 基本HTML结构和CSS样式
        html_head = f'''<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>PDOR识别结果: {html.escape(data.get('file_name', ''))}</title>
    <style>
        body {{
            font-family: Arial, sans-serif;
            line-height: 1.6;
            color: #333;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
        }}
        h1, h2, h3 {{
            color: #2c3e50;
        }}
        .overview {{
            background-color: #f8f9fa;
            padding: 15px;
            border-radius: 5px;
            margin-bottom: 20px;
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin-bottom: 20px;
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 8px;
            text-align: left;
        }}
        th {{
            background-color: #4CAF50;
            color: white;
        }}
        tr:nth-child(even) {{
            background-color: #f2f2f2;
        }}
        .page {{
            margin-bottom: 30px;
            border: 1px solid #ddd;
            padding: 15px;
            border-radius: 5px;
        }}
        .table-container {{
            margin-bottom: 20px;
        }}
    </style>
</head>
<body>
    <h1>PDOR识别结果</h1>
    <div class="overview">
        <p><strong>文件名:</strong> {html.escape(data.get('file_name', ''))}</p>
        <p><strong>总页数:</strong> {data.get('total_pages', 0)}</p>
    </div>
'''

        # 添加表格数据
        html_body = ''
        for page_key, page_tables in data.get('tables', {}).items():
            html_body += f'<div class="page">\n<h2>{html.escape(page_key)}</h2>\n'

            for table_idx, table_data in enumerate(page_tables):
                html_body += f'<div class="table-container">\n<h3>表格 {table_idx + 1}</h3>\n'

                if table_data:
                    # 获取所有列名
                    all_columns = set()
                    for row_data in table_data.values():
                        all_columns.update(row_data.keys())

                    # 创建表格
                    html_body += '<table>\n<thead>\n<tr>\n<th>行ID</th>\n'
                    for col in sorted(all_columns):
                        html_body += f'<th>{html.escape(col)}</th>\n'
                    html_body += '</tr>\n</thead>\n<tbody>\n'

                    # 添加行数据
                    for row_id, row_data in table_data.items():
                        html_body += f'<tr>\n<td>{html.escape(str(row_id))}</td>\n'
                        for col in sorted(all_columns):
                            cell_value = row_data.get(col, '')
                            html_body += f'<td>{html.escape(str(cell_value))}</td>\n'
                        html_body += '</tr>\n'

                    html_body += '</tbody>\n</table>\n'
                else:
                    html_body += '<p>此表格无数据</p>\n'

                html_body += '</div>\n'

            html_body += '</div>\n'

        # 关闭HTML标签
        html_footer = '''
</body>
</html>
'''

        # 写入完整HTML
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(html_head + html_body + html_footer)

    @staticmethod
    def _write_markdown(data: Dict[str, Any], filename: str) -> None:
        """
        将数据以Markdown格式写入文件
        """
        md_content = f"# PDOR识别结果\n\n"
        md_content += f"**文件名:** {data.get('file_name', '')}\n\n"
        md_content += f"**总页数:** {data.get('total_pages', 0)}\n\n"

        for page_key, page_tables in data.get('tables', {}).items():
            md_content += f"## {page_key}\n\n"

            for table_idx, table_data in enumerate(page_tables):
                md_content += f"### 表格 {table_idx + 1}\n\n"

                if table_data:
                    # 获取所有列名
                    all_columns = set()
                    for row_data in table_data.values():
                        all_columns.update(row_data.keys())

                    # 创建表头
                    md_content += "| 行ID |"
                    for col in sorted(all_columns):
                        md_content += f" {col} |"
                    md_content += "\n"

                    # 添加分隔行
                    md_content += "| --- |"
                    for _ in all_columns:
                        md_content += " --- |"
                    md_content += "\n"

                    # 添加行数据
                    for row_id, row_data in table_data.items():
                        md_content += f"| {row_id} |"
                        for col in sorted(all_columns):
                            cell_value = row_data.get(col, '')
                            md_content += f" {cell_value} |"
                        md_content += "\n"

                    md_content += "\n"
                else:
                    md_content += "此表格无数据\n\n"

        # 写入Markdown文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(md_content)

    @staticmethod
    def _write_plaintext(data: Dict[str, Any], filename: str) -> None:
        """
        将数据以纯文本格式写入文件
        """
        text_content = f"PDOR识别结果\n"
        text_content += f"=" * 50 + "\n\n"
        text_content += f"文件名: {data.get('file_name', '')}\n"
        text_content += f"总页数: {data.get('total_pages', 0)}\n\n"

        for page_key, page_tables in data.get('tables', {}).items():
            text_content += f"{page_key}\n"
            text_content += "-" * 50 + "\n\n"

            for table_idx, table_data in enumerate(page_tables):
                text_content += f"表格 {table_idx + 1}\n"
                text_content += "~" * 30 + "\n\n"

                if table_data:
                    for row_id, row_data in table_data.items():
                        text_content += f"行ID: {row_id}\n"
                        for col, value in row_data.items():
                            text_content += f"  {col}: {value}\n"
                        text_content += "\n"
                else:
                    text_content += "此表格无数据\n\n"

        # 写入文本文件
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(text_content)