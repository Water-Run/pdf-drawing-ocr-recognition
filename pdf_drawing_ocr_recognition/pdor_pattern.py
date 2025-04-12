r"""
PDOR模式
:author: WaterRun
:time: 2025-04-12
:file: pdor_pattern.py
"""

import re
import simpsave as ss

from pdor_exception import *


def build_pattern_config(
        dpi: int,
        table_headers: list,
        key_column: int,
        min_rows: int,
        min_columns: int,
        header_row: int,
        data_start_row: int,
        threshold_method: str,
        contrast_adjust: float,
        denoise: bool,
        border_removal: int,
        deskew: bool,
        lang: str,
        psm: int,
        oem: int,
        whitelist: str,
        trim_whitespace: bool,
        merge_adjacent_cells: bool,
        pattern_corrections: list,
        column_types: dict,
        column_patterns: dict,
        min_area: int,
        aspect_ratio_range: list,
        line_detection_method: str
) -> dict:
    r"""
    构造模式配置
    :param dpi: 解析PDF的图片DPI
    :param table_headers: 表头字段列表
    :param key_column: 关键列索引（作为字典键的列）
    :param min_rows: 最小行数
    :param min_columns: 最小列数
    :param header_row: 表头行号
    :param data_start_row: 数据起始行
    :param threshold_method: 阈值处理方法，可选 'otsu' 或 'adaptive'
    :param contrast_adjust: 对比度调整系数
    :param denoise: 是否去噪
    :param border_removal: 移除边框像素数
    :param deskew: 是否纠正倾斜
    :param lang: OCR语言
    :param psm: 页面分割模式
    :param oem: OCR引擎模式
    :param whitelist: 允许的字符集
    :param trim_whitespace: 是否去除首尾空白
    :param merge_adjacent_cells: 是否合并相邻单元格
    :param pattern_corrections: 正则表达式替换规则列表，每项为 (pattern, replacement) 元组
    :param column_types: 列数据类型字典，键为列索引，值为类型名称
    :param column_patterns: 列正则表达式匹配模式字典，键为列索引，值为正则表达式
    :param min_area: 最小表格面积
    :param aspect_ratio_range: 表格宽高比范围 [min_ratio, max_ratio]
    :param line_detection_method: 线检测方法，可选 'hough' 或 'contour'
    :return: 配置的模式字典
    :raise PdorBuildPatternInvalidParamError: 如果参数验证失败
    """

    # 检查dpi
    if not isinstance(dpi, int) or not 72 < dpi < 1350:
        raise PdorBuildPatternInvalidParamError(
            message="dpi必须是72到1350内的整数"
        )

    # 检查表头和关键列
    if not isinstance(table_headers, list) or not all(isinstance(h, str) for h in table_headers):
        raise PdorBuildPatternInvalidParamError(
            message="表头必须是字符串列表"
        )

    if not isinstance(key_column, int):
        raise PdorBuildPatternInvalidParamError(
            message="关键列索引必须是整数"
        )
    if key_column < 0 or (table_headers and key_column >= len(table_headers)):
        raise PdorBuildPatternInvalidParamError(
            message=f"关键列索引 {key_column} 超出表头范围 [0, {len(table_headers) - 1}]"
        )

    # 检查表格结构参数
    if not isinstance(min_rows, int) or min_rows < 1:
        raise PdorBuildPatternInvalidParamError(
            message="最小行数必须是大于0的整数"
        )
    if not isinstance(min_columns, int) or min_columns < 1:
        raise PdorBuildPatternInvalidParamError(
            message="最小列数必须是大于0的整数"
        )
    if not isinstance(header_row, int) or header_row < 0:
        raise PdorBuildPatternInvalidParamError(
            message="表头行号必须是非负整数"
        )
    if not isinstance(data_start_row, int) or data_start_row <= header_row:
        raise PdorBuildPatternInvalidParamError(
            message="数据起始行必须大于表头行号"
        )

    # 检查图像处理参数
    if threshold_method not in ['otsu', 'adaptive']:
        raise PdorBuildPatternInvalidParamError(
            message="阈值处理方法必须是 'otsu' 或 'adaptive'"
        )
    if not isinstance(contrast_adjust, (int, float)) or contrast_adjust <= 0:
        raise PdorBuildPatternInvalidParamError(
            message="对比度调整系数必须是正数"
        )
    if not isinstance(denoise, bool):
        raise PdorBuildPatternInvalidParamError(
            message="去噪标志必须是布尔值"
        )
    if not isinstance(border_removal, int) or border_removal < 0:
        raise PdorBuildPatternInvalidParamError(
            message="边框移除像素数必须是非负整数"
        )
    if not isinstance(deskew, bool):
        raise PdorBuildPatternInvalidParamError(
            message="倾斜纠正标志必须是布尔值"
        )

    # 检查OCR参数
    if not isinstance(lang, str) or not lang:
        raise PdorBuildPatternInvalidParamError(
            message="OCR语言必须是非空字符串"
        )
    if not isinstance(psm, int) or psm < 0 or psm > 13:
        raise PdorBuildPatternInvalidParamError(
            message="页面分割模式必须在0到13之间"
        )
    if not isinstance(oem, int) or oem < 0 or oem > 3:
        raise PdorBuildPatternInvalidParamError(
            message="OCR引擎模式必须在0到3之间"
        )
    if not isinstance(whitelist, str):
        raise PdorBuildPatternInvalidParamError(
            message="允许字符集必须是字符串"
        )

    # 检查后处理参数
    if not isinstance(trim_whitespace, bool):
        raise PdorBuildPatternInvalidParamError(
            message="去除空白标志必须是布尔值"
        )
    if not isinstance(merge_adjacent_cells, bool):
        raise PdorBuildPatternInvalidParamError(
            message="合并单元格标志必须是布尔值"
        )

    if not isinstance(pattern_corrections, list):
        raise PdorBuildPatternInvalidParamError(
            message="替换规则必须是列表"
        )
    for i, rule in enumerate(pattern_corrections):
        if not isinstance(rule, tuple) or len(rule) != 2 or not all(isinstance(item, str) for item in rule):
            raise PdorBuildPatternInvalidParamError(
                message=f"替换规则 #{i + 1} 必须是 (pattern, replacement) 格式的字符串元组"
            )

    # 检查列类型和模式
    if not isinstance(column_types, dict):
        raise PdorBuildPatternInvalidParamError(
            message="列类型必须是字典"
        )
    for col, type_name in column_types.items():
        if not isinstance(col, int) or col < 0:
            raise PdorBuildPatternInvalidParamError(
                message=f"列索引 {col} 必须是非负整数"
            )
        if not isinstance(type_name, str) or not type_name:
            raise PdorBuildPatternInvalidParamError(
                message=f"列 {col} 的类型名称必须是非空字符串"
            )

    if not isinstance(column_patterns, dict):
        raise PdorBuildPatternInvalidParamError(
            message="列模式必须是字典"
        )
    for col, pattern in column_patterns.items():
        if not isinstance(col, int) or col < 0:
            raise PdorBuildPatternInvalidParamError(
                message=f"列索引 {col} 必须是非负整数"
            )
        if not isinstance(pattern, str) or not pattern:
            raise PdorBuildPatternInvalidParamError(
                message=f"列 {col} 的模式必须是非空字符串"
            )
        try:
            re.compile(pattern)
        except re.error:
            raise PdorBuildPatternInvalidParamError(
                message=f"列 {col} 的模式 '{pattern}' 不是有效的正则表达式"
            )

    # 检查表格检测参数
    if not isinstance(min_area, int) or min_area <= 0:
        raise PdorBuildPatternInvalidParamError(
            message="最小表格面积必须是正整数"
        )

    if not isinstance(aspect_ratio_range, list) or len(aspect_ratio_range) != 2:
        aspect_ratio_range = [0.5, 5.0]  # 默认值
    else:
        if not all(isinstance(ratio, (int, float)) and ratio > 0 for ratio in aspect_ratio_range):
            raise PdorBuildPatternInvalidParamError(
                message="表格宽高比必须是正数"
            )
        if aspect_ratio_range[0] >= aspect_ratio_range[1]:
            raise PdorBuildPatternInvalidParamError(
                message="表格宽高比范围的最小值必须小于最大值"
            )

    if line_detection_method not in ['hough', 'contour']:
        raise PdorBuildPatternInvalidParamError(
            message="线检测方法必须是 'hough' 或 'contour'"
        )

    return {
        'dpi': dpi,
        'table_headers': table_headers,
        'key_column': key_column,
        'table_structure': {
            'min_rows': min_rows,
            'min_columns': min_columns,
            'header_row': header_row,
            'data_start_row': data_start_row,
        },
        'image_processing': {
            'threshold_method': threshold_method,
            'contrast_adjust': contrast_adjust,
            'denoise': denoise,
            'border_removal': border_removal,
            'deskew': deskew,
        },
        'ocr_config': {
            'lang': lang,
            'psm': psm,
            'oem': oem,
            'whitelist': whitelist,
        },
        'post_processing': {
            'trim_whitespace': trim_whitespace,
            'merge_adjacent_cells': merge_adjacent_cells,
            'pattern_corrections': pattern_corrections,
        },
        'column_types': column_types,
        'column_patterns': column_patterns,
        'table_detection': {
            'min_area': min_area,
            'aspect_ratio_range': aspect_ratio_range,
            'line_detection_method': line_detection_method,
        }
    }


class PdorPattern:
    r"""
    Pdor模式单元
    :param name: 模式名称
    :param config: 构造的模式字典
    """

    def __init__(self, name: str, config: dict):
        try:
            if not isinstance(name, str) or not name:
                raise PdorBuildPatternInvalidParamError(
                    message="模式名称必须是非空字符串"
                )
            if not isinstance(config, dict):
                raise PdorBuildPatternInvalidParamError(
                    message="配置必须是字典"
                )

            self.name = name
            self.description = config.get('description', f"{name}模式")

            self.table_headers = config.get('table_headers', [])
            self.key_column = config.get('key_column', 0)

            table_structure = config.get('table_structure')
            if not table_structure or not isinstance(table_structure, dict):
                raise PdorBuildPatternInvalidParamError(
                    message="缺少表格结构配置或格式错误"
                )
            self.table_structure = {
                'min_rows': table_structure.get('min_rows', 3),
                'min_columns': table_structure.get('min_columns', 3),
                'header_row': table_structure.get('header_row', 0),
                'data_start_row': table_structure.get('data_start_row', 1),
            }

            image_processing = config.get('image_processing')
            if not image_processing or not isinstance(image_processing, dict):
                raise PdorBuildPatternInvalidParamError(
                    message="缺少图像处理配置或格式错误"
                )
            self.image_processing = {
                'threshold_method': image_processing.get('threshold_method', 'otsu'),
                'contrast_adjust': image_processing.get('contrast_adjust', 1.0),
                'denoise': image_processing.get('denoise', False),
                'border_removal': image_processing.get('border_removal', 0),
                'deskew': image_processing.get('deskew', False),
            }

            ocr_config = config.get('ocr_config')
            if not ocr_config or not isinstance(ocr_config, dict):
                raise PdorBuildPatternInvalidParamError(
                    message="缺少OCR配置或格式错误"
                )
            self.ocr_config = {
                'lang': ocr_config.get('lang', 'chi_sim+eng'),
                'psm': ocr_config.get('psm', 6),
                'oem': ocr_config.get('oem', 3),
                'whitelist': ocr_config.get('whitelist'),
            }

            post_processing = config.get('post_processing')
            if not post_processing or not isinstance(post_processing, dict):
                raise PdorBuildPatternInvalidParamError(
                    message="缺少后处理配置或格式错误"
                )
            self.post_processing = {
                'trim_whitespace': post_processing.get('trim_whitespace', True),
                'merge_adjacent_cells': post_processing.get('merge_adjacent_cells', False),
                'pattern_corrections': post_processing.get('pattern_corrections', []),
            }

            self.column_types = config.get('column_types', {})
            self.column_patterns = config.get('column_patterns', {})

            table_detection = config.get('table_detection')
            if not table_detection or not isinstance(table_detection, dict):
                raise PdorBuildPatternInvalidParamError(
                    message="缺少表格检测配置或格式错误"
                )
            self.table_detection = {
                'min_area': table_detection.get('min_area', 10000),
                'aspect_ratio_range': table_detection.get('aspect_ratio_range', [0.5, 5.0]),
                'line_detection_method': table_detection.get('line_detection_method', 'hough'),
            }
            self.config = config

        except KeyError as e:
            raise PdorBuildPatternInvalidParamError(
                message=f"缺少必要的配置键: {str(e)}"
            )
        except TypeError as e:
            raise PdorBuildPatternInvalidParamError(
                message=f"配置类型错误: {str(e)}"
            )
        except Exception as e:
            raise PdorBuildPatternInvalidParamError(
                message=f"配置构造失败: {str(e)}"
            )

    def __repr__(self) -> str:
        """
        返回Pdor模式的字符串表示
        """
        info = [
            f"===PdorPattern: {self.name}===",
            f"描述: {self.description}",
            "",
            f"表头: {', '.join(self.table_headers) if self.table_headers else '未定义'}",
            f"关键列: {self.key_column} ({self.table_headers[self.key_column] if self.table_headers and self.key_column < len(self.table_headers) else '未命名'})",
            "",
            "表格结构配置:",
        ]

        for key, value in self.table_structure.items():
            info.append(f"  {key}: {value}")

        info.append("")
        info.append("图像处理配置:")
        for key, value in self.image_processing.items():
            info.append(f"  {key}: {value}")

        info.append("")
        info.append("OCR配置:")
        for key, value in self.ocr_config.items():
            if key == 'whitelist' and value and len(value) > 20:
                info.append(f"  {key}: {value[:20]}...")
            else:
                info.append(f"  {key}: {value}")

        info.append("")
        info.append("后处理配置:")
        for key, value in self.post_processing.items():
            if key == 'pattern_corrections':
                info.append(f"  替换规则数量: {len(value)}")
                for i, (pattern, replacement) in enumerate(value[:3]):
                    info.append(f"    规则{i + 1}: '{pattern}' -> '{replacement}'")
                if len(value) > 3:
                    info.append(f"    ... 及其他{len(value) - 3}条规则")
            else:
                info.append(f"  {key}: {value}")

        if self.column_patterns:
            info.append("")
            info.append("列模式配置:")
            for col, pattern in self.column_patterns.items():
                col_name = self.table_headers[col] if self.table_headers and col < len(
                    self.table_headers) else f"列{col}"
                if len(pattern) > 30:
                    info.append(f"  {col_name}: {pattern[:30]}...")
                else:
                    info.append(f"  {col_name}: {pattern}")

        return "\n".join(info)


def save(pattern: PdorPattern, file: str) -> None:
    r"""
    保存PdorPattern.
    保存的键名和PdorPattern单元名一致.
    :param pattern: 待保存的PdorPattern
    :param file: 保存的simpsave文件名
    :return: None
    """
    ss.write(pattern.name, pattern.config, file=file)


def load(name: str, file: str) -> PdorPattern:
    r"""
    读取PdorPattern.
    :param name: 待读取的PdorPattern名称
    :param file: 读取的simpsave文件名
    :return: 根据读取内容构造的PdorPattern
    """
    return PdorPattern(name, ss.read(name, file=file))
