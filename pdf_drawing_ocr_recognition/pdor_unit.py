r"""
PDOR单元
:author: WaterRun
:time: 2025-04-12
:file: pdor_unit.py
"""

import os
import re
import time
import tempfile
import simpsave as ss
import numpy as np
import inspect
import pytesseract
import cv2

from PyPDF2 import PdfReader
from pdf2image import convert_from_path
from collections import defaultdict

from pdor_pattern import PdorPattern, PDOR_PATTERNS
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
        将读出的PDF转为图片，使用较高DPI处理大型PDF文件
        :raise PdorImagifyError: 如果图片转换时出现异常
        :return: None
        """
        if self._pdf is None:
            raise PdorImagifyError(
                message="无可用的PDF实例"
            )

        try:
            # 从模式中获取DPI设置，默认为300
            dpi = 300
            if hasattr(self._pattern, 'image_processing') and isinstance(self._pattern.image_processing, dict):
                dpi = self._pattern.image_processing.get('dpi', 1200)

            # 处理大型PDF的参数配置
            # thread_count: 使用多线程加速处理
            # first_page/last_page: 允许分批处理页面以减少内存消耗
            # use_cropbox: 使用裁剪框可以减少内存消耗
            # output_folder: 临时保存转换的图片到磁盘，减轻内存压力

            # 创建临时目录
            with tempfile.TemporaryDirectory() as temp_dir:
                # 转换时使用较高DPI但不超出内存限制
                self._img = []
                images = convert_from_path(
                    self._file_name,
                    dpi=dpi,  # 较高的DPI以提高OCR质量
                    thread_count=4,  # 使用4个线程加速处理
                    use_cropbox=True,  # 使用裁剪框减少内存消耗
                    output_folder=temp_dir,  # 临时保存到磁盘
                    fmt="jpeg",  # 使用JPEG格式节省空间
                    jpegopt={"quality": 90, "optimize": True, "progressive": True}  # JPEG优化选项
                )

                for image in images:
                    # 优化图像处理，避免直接将所有图像加载到内存
                    img_array = np.array(image)

                    # 对于特别大的图像，可以考虑调整大小以减少内存消耗
                    h, w = img_array.shape[:2]
                    max_dimension = 3000  # 设置最大尺寸限制

                    if max(h, w) > max_dimension:
                        # 等比例缩小
                        scale = max_dimension / max(h, w)
                        new_h, new_w = int(h * scale), int(w * scale)
                        img_array = cv2.resize(img_array, (new_w, new_h), interpolation=cv2.INTER_AREA)

                    self._img.append(img_array)

                    # 主动释放PIL图像对象以节省内存
                    image.close()

            if not self._img:
                raise PdorImagifyError(
                    message="无法从PDF中提取图像"
                )

        except Exception as error:
            raise PdorImagifyError(
                message=f"PDF图片化失败: {str(error)}"
            )

    def _ocr(self):
        # 设置Tesseract路径
        pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

        if not self._img:
            raise PdorOCRError(message="没有可用的图像进行OCR")

        try:
            # 获取模式中的图像处理配置
            img_proc_config = self._pattern.image_processing

            # 结果存储
            result_dict = {}

            # 创建调试目录
            debug_dir = "debug_images"
            os.makedirs(debug_dir, exist_ok=True)

            for page_idx, img_array in enumerate(self._img):
                # 保存原始图像用于调试
                cv2.imwrite(f"{debug_dir}/page_{page_idx}_original.jpg", cv2.cvtColor(img_array, cv2.COLOR_RGB2BGR))

                # 1. 预处理图像
                # 转为灰度图
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)

                # 调整对比度
                if img_proc_config['contrast_adjust'] != 1.0:
                    alpha = img_proc_config['contrast_adjust']
                    gray = cv2.convertScaleAbs(gray, alpha=alpha, beta=0)

                # 去噪处理
                if img_proc_config['denoise']:
                    gray = cv2.fastNlMeansDenoising(gray, None, 10, 7, 21)

                # 根据配置选择阈值处理方法
                if img_proc_config['threshold_method'] == 'adaptive':
                    binary = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                   cv2.THRESH_BINARY_INV, 11, 2)
                else:  # otsu
                    _, binary = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)

                # 倾斜校正
                if img_proc_config['deskew']:
                    # 找到所有可能的线段
                    edges = cv2.Canny(gray, 50, 150, apertureSize=3)
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

                    if lines is not None:
                        angles = []
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            if x2 - x1 != 0:  # 避免除零错误
                                angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
                                # 只考虑接近水平的线
                                if abs(angle) < 45:
                                    angles.append(angle)

                        if angles:
                            # 计算角度中位数作为校正角度
                            median_angle = np.median(angles)
                            (h, w) = gray.shape[:2]
                            center = (w // 2, h // 2)
                            M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                            gray = cv2.warpAffine(gray, M, (w, h),
                                                  flags=cv2.INTER_CUBIC,
                                                  borderMode=cv2.BORDER_REPLICATE)
                            binary = cv2.warpAffine(binary, M, (w, h),
                                                    flags=cv2.INTER_CUBIC,
                                                    borderMode=cv2.BORDER_REPLICATE)
                            img_array = cv2.warpAffine(img_array, M, (w, h),
                                                       flags=cv2.INTER_CUBIC,
                                                       borderMode=cv2.BORDER_REPLICATE)

                # 保存预处理后的图像用于调试
                cv2.imwrite(f"{debug_dir}/page_{page_idx}_binary.jpg", binary)

                # 2. 检测表格结构 - 使用配置中指定的检测方法
                if self._pattern.table_detection['line_detection_method'] == 'hough':
                    # 使用霍夫变换检测表格线条
                    # 首先膨胀二值图像使线条更明显
                    kernel = np.ones((3, 3), np.uint8)
                    dilated = cv2.dilate(binary, kernel, iterations=2)

                    # 检测边缘
                    edges = cv2.Canny(dilated, 50, 150, apertureSize=3)
                    cv2.imwrite(f"{debug_dir}/page_{page_idx}_edges.jpg", edges)

                    # 检测水平和垂直线
                    lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 50, minLineLength=50, maxLineGap=10)

                    # 使用检测到的线绘制表格掩码
                    h, w = gray.shape[:2]
                    table_mask = np.zeros((h, w), dtype=np.uint8)

                    if lines is not None:
                        for line in lines:
                            x1, y1, x2, y2 = line[0]
                            cv2.line(table_mask, (x1, y1), (x2, y2), 255, 2)

                    # 再次膨胀掩码以连接线条
                    table_mask = cv2.dilate(table_mask, kernel, iterations=3)
                    cv2.imwrite(f"{debug_dir}/page_{page_idx}_table_mask.jpg", table_mask)

                    # 寻找表格轮廓
                    contours, _ = cv2.findContours(table_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                else:  # 'contour'方法
                    # 直接通过轮廓检测查找表格区域
                    dilated = cv2.dilate(binary, np.ones((5, 5), np.uint8), iterations=3)
                    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 筛选可能的表格轮廓
                table_contours = []
                for c in contours:
                    area = cv2.contourArea(c)
                    x, y, w, h = cv2.boundingRect(c)
                    aspect_ratio = float(w) / h if h > 0 else 0

                    # 根据配置的面积和宽高比范围筛选表格
                    min_area = self._pattern.table_detection['min_area']
                    aspect_ratio_range = self._pattern.table_detection['aspect_ratio_range']

                    if (area > min_area and
                            aspect_ratio_range[0] <= aspect_ratio <= aspect_ratio_range[1]):
                        table_contours.append(c)

                # 将表格轮廓绘制到原图上以便调试
                debug_img = img_array.copy()
                cv2.drawContours(debug_img, table_contours, -1, (0, 255, 0), 3)
                cv2.imwrite(f"{debug_dir}/page_{page_idx}_detected_tables.jpg",
                            cv2.cvtColor(debug_img, cv2.COLOR_RGB2BGR))

                page_results = []

                # 3. 对每个检测到的表格进行处理
                for table_idx, contour in enumerate(table_contours):
                    x, y, w, h = cv2.boundingRect(contour)
                    table_img = img_array[y:y + h, x:x + w]

                    # 保存提取的表格图像用于调试
                    cv2.imwrite(f"{debug_dir}/page_{page_idx}_table_{table_idx}.jpg",
                                cv2.cvtColor(table_img, cv2.COLOR_RGB2BGR))

                    # 4. 使用pytesseract设置带有配置的参数
                    ocr_config = self._pattern.ocr_config
                    custom_config = f'--oem {ocr_config["oem"]} --psm {ocr_config["psm"]}'

                    if ocr_config['whitelist']:
                        custom_config += f' -c tessedit_char_whitelist="{ocr_config["whitelist"]}"'

                    # 5. 进行OCR识别
                    # 首先增强表格图像
                    table_gray = cv2.cvtColor(table_img, cv2.COLOR_RGB2GRAY)

                    if img_proc_config['threshold_method'] == 'adaptive':
                        table_thresh = cv2.adaptiveThreshold(table_gray, 255,
                                                             cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                             cv2.THRESH_BINARY, 11, 2)
                    else:
                        _, table_thresh = cv2.threshold(table_gray, 0, 255,
                                                        cv2.THRESH_BINARY + cv2.THRESH_OTSU)

                    # 保存表格二值图像用于调试
                    cv2.imwrite(f"{debug_dir}/page_{page_idx}_table_{table_idx}_binary.jpg", table_thresh)

                    # 使用pytesseract的表格识别功能
                    try:
                        # 先尝试直接识别表格结构
                        data = pytesseract.image_to_data(table_thresh, lang=ocr_config['lang'],
                                                         output_type=pytesseract.Output.DICT,
                                                         config=custom_config)

                        # 创建行列结构
                        rows = defaultdict(dict)

                        # 处理识别结果
                        for i in range(len(data['text'])):
                            if data['text'][i].strip():
                                # 应用文本修正规则
                                corrected_text = data['text'][i].strip()
                                for pattern, replacement in self._pattern.post_processing['pattern_corrections']:
                                    corrected_text = re.sub(pattern, replacement, corrected_text)

                                # 根据y坐标估计行号
                                row_idx = data['top'][i] // 30  # 调整行高估计
                                # 根据x坐标估计列号
                                col_idx = data['left'][i] // 150  # 调整列宽估计

                                # 存储文本
                                if col_idx not in rows[row_idx]:
                                    rows[row_idx][col_idx] = corrected_text
                                else:
                                    rows[row_idx][col_idx] += ' ' + corrected_text

                        # 7. 将行列数据转换为结构化字典
                        structured_table = {}

                        # 确定表头行
                        header_row = self._pattern.table_structure['header_row']
                        data_start_row = self._pattern.table_structure['data_start_row']

                        # 获取表头
                        headers = self._pattern.table_headers

                        # 如果没有表头或行数不足，使用默认表头
                        if not headers or len(rows) <= header_row:
                            # 从识别结果中提取表头
                            if header_row in rows:
                                headers = [rows[header_row].get(i, f"Col_{i}")
                                           for i in range(max(rows[header_row].keys()) + 1
                                                          if rows[header_row] else 0)]
                            else:
                                # 使用默认表头
                                headers = [f"Col_{i}" for i in range(10)]  # 假设最多10列

                        # 提取数据行
                        for row_idx in sorted(rows.keys()):
                            if row_idx >= data_start_row:  # 跳过表头
                                # 构建行数据
                                row_data = {}

                                for col_idx, value in rows[row_idx].items():
                                    # 如果有对应的表头则使用，否则使用列索引
                                    col_name = headers[col_idx] if col_idx < len(headers) else f"Col_{col_idx}"
                                    row_data[col_name] = value

                                # 确定行的唯一键
                                key_column = self._pattern.key_column
                                if key_column < len(headers) and headers[key_column] in row_data:
                                    key = row_data[headers[key_column]]
                                else:
                                    # 如果指定的键列不存在，使用行索引作为键
                                    key = f"Row_{row_idx}"

                                structured_table[key] = row_data

                        page_results.append(structured_table)

                    except Exception as e:
                        print(f"表格 {table_idx} OCR失败: {str(e)}")
                        # 添加空表格作为占位符
                        page_results.append({})

                # 如果没有检测到表格，尝试整页OCR
                if not table_contours:
                    print(f"页面 {page_idx} 未检测到表格，尝试整页OCR...")

                    # 整页OCR
                    ocr_config = self._pattern.ocr_config
                    custom_config = f'--oem {ocr_config["oem"]} --psm 6'

                    text = pytesseract.image_to_string(gray, lang=ocr_config['lang'], config=custom_config)

                    # 简单按行分割作为最基本的结构
                    lines = text.split('\n')
                    structured_data = {}

                    for i, line in enumerate(lines):
                        if line.strip():
                            structured_data[f"Row_{i}"] = {"Text": line.strip()}

                    page_results.append(structured_data)

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

    def output(self, *, print_repr: bool = False) -> None:
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
                     f"模式: \n"
                     f"{self._pattern}\n"
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


if __name__ == '__main__':
    def debug_image_processing(pdf_file, output_dir="debug_output", dpi=1200):
        """
        对PDF文件进行图像处理调试，支持高分辨率和大文件处理
        :param pdf_file: PDF文件路径
        :param output_dir: 输出目录
        :param dpi: 转换时使用的DPI值，默认300
        """
        # 创建输出目录
        os.makedirs(output_dir, exist_ok=True)

        print(f"开始处理PDF文件: {pdf_file}")
        print(f"使用DPI: {dpi}")

        # 创建临时目录用于存储中间文件
        with tempfile.TemporaryDirectory() as temp_dir:
            # 转换PDF到图像，使用更高DPI和多线程
            start_time = time.time()

            # 转换参数优化，提高大文件处理能力
            images = convert_from_path(
                pdf_file,
                dpi=dpi,
                thread_count=4,
                use_cropbox=True,
                output_folder=temp_dir,
                fmt="jpeg",
                jpegopt={"quality": 90, "optimize": True, "progressive": True}
            )

            convert_time = time.time() - start_time
            print(f"PDF转换耗时: {convert_time:.2f}秒")
            print(f"总页数: {len(images)}")

            # 处理每一页图像，优化内存使用
            for i, image in enumerate(images):
                print(f"处理第 {i + 1} 页...")
                img_size = f"{image.width}x{image.height}"
                print(f"图像尺寸: {img_size}")

                # 保存原始图像
                img_path = f"{output_dir}/page_{i}_original.jpg"
                image.save(img_path, "JPEG", quality=90, optimize=True)
                file_size = os.path.getsize(img_path) / 1024
                print(f"原始图像文件大小: {file_size:.2f} KB")

                # 转换为OpenCV格式处理
                img_array = np.array(image)

                # 记录内存使用
                import psutil
                process = psutil.Process()
                memory_info = process.memory_info()
                print(f"当前内存使用: {memory_info.rss / (1024 * 1024):.2f} MB")

                # 灰度处理
                gray = cv2.cvtColor(img_array, cv2.COLOR_RGB2GRAY)
                cv2.imwrite(f"{output_dir}/page_{i}_gray.jpg", gray, [cv2.IMWRITE_JPEG_QUALITY, 90])

                # 基本图像处理
                # 自适应阈值
                adaptive = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 11, 2)
                cv2.imwrite(f"{output_dir}/page_{i}_adaptive.jpg", adaptive, [cv2.IMWRITE_JPEG_QUALITY, 90])

                # OTSU阈值
                _, otsu = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                cv2.imwrite(f"{output_dir}/page_{i}_otsu.jpg", otsu, [cv2.IMWRITE_JPEG_QUALITY, 90])

                # 边缘检测
                edges = cv2.Canny(gray, 50, 150)
                cv2.imwrite(f"{output_dir}/page_{i}_edges.jpg", edges, [cv2.IMWRITE_JPEG_QUALITY, 90])

                # 轮廓检测 - 直接在二值图上检测
                contours1, _ = cv2.findContours(adaptive, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 过滤小轮廓以提高效率
                large_contours = [c for c in contours1 if cv2.contourArea(c) > 1000]
                print(f"检测到 {len(large_contours)} 个主要轮廓")

                # 仅绘制大轮廓
                contour_img = img_array.copy()
                cv2.drawContours(contour_img, large_contours, -1, (0, 255, 0), 3)
                cv2.imwrite(f"{output_dir}/page_{i}_contours.jpg",
                            cv2.cvtColor(contour_img, cv2.COLOR_RGB2BGR),
                            [cv2.IMWRITE_JPEG_QUALITY, 85])

                # 膨胀轮廓的测试
                kernel = np.ones((5, 5), np.uint8)
                dilated = cv2.dilate(adaptive, kernel, iterations=3)
                contours2, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # 过滤小轮廓
                large_contours2 = [c for c in contours2 if cv2.contourArea(c) > 5000]

                # 为每个检测到的表格区域创建单独图像
                for j, contour in enumerate(large_contours2):
                    x, y, w, h = cv2.boundingRect(contour)
                    # 扩大边界以确保完整捕获表格
                    padding = 20
                    x = max(0, x - padding)
                    y = max(0, y - padding)
                    w = min(img_array.shape[1] - x, w + 2 * padding)
                    h = min(img_array.shape[0] - y, h + 2 * padding)

                    # 提取区域
                    region = img_array[y:y + h, x:x + w]
                    if region.size > 0:  # 确保区域有效
                        cv2.imwrite(f"{output_dir}/page_{i}_region_{j}.jpg",
                                    cv2.cvtColor(region, cv2.COLOR_RGB2BGR),
                                    [cv2.IMWRITE_JPEG_QUALITY, 95])

                # 释放大型图像以节省内存
                del img_array, gray, adaptive, otsu, edges, contour_img, dilated
                image.close()  # 关闭PIL图像

                # 强制垃圾回收
                import gc
                gc.collect()

                print(f"完成第 {i + 1} 页处理\n")

        return f"调试图像已保存到 {output_dir} 目录，使用DPI={dpi}"


    debug_image_processing('../tests/700501-8615-72-12 750kV 第四串测控柜A+1端子排图左.PDF', output_dir='../_debug/')