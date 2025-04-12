# pdf-drawing-ocr-recognition: Pdor  

定制项目,一个Python库,实现将PDF输入的指定模式的表格进行数据化读取.  

## 使用前准备  

### 环境  

- Windows
- Python 3.10+  

## 依赖库  

使用`pip`安装以下依赖库:  

- simpsave - 轻量级配置文件存储工具
- PyPDF2 - PDF操作库
- pdf2image - PDF转图像工具
- numpy - 科学计算库
- pytesseract - Tesseract OCR接口
- opencv-python (cv2) - 图像处理库

## `Tesseract OCR`安装  

1. 下载[Tesseract安装器](https://github.com/tesseract-ocr/tesseract/releases/download/5.5.0/tesseract-ocr-w64-setup-5.5.0.20241111.exe)  
2. 执行安装,安装在默认路径`C:\Program Files\Tesseract-OCR`  
3. 安装语言包`eng.traineddata`和`chi_sim.traineddata`:[下载](https://github.com/tesseract-ocr/tessdata/tree/main)并拷贝至`C:\Program Files\Tesseract-OCR\tessdata`  

## 快速指南

### 安装  

项目已经发布在PyPi上,使用`pip`进行安装:  

```cmd
pip install pdor
```

库的名称简写为`pdor`.  

然后,在你的项目中导入pdor:  
```python
import pdor
```

### 环境检查

提供了一个`check_env()`方法检测当前是否可用.  

***示例***:  
```python
import pdor

status, msg = pdor.check_env()
if not status: # 如果检查不通过
    for info in msg: # 逐项打印错误
        print(info)
```

- `status`将是一个布尔值.如果为真,则环境可用.  
- `msg`是一个字符串列表,包括错误信息.  

### Pdor模式  

Pdor模式是表格识别的核心配置，定义了如何解析和提取表格数据。模式定义了表格的结构、OCR参数、图像处理方法和识别后的文本处理规则。

#### 预设模式读取  

#### 自行构建  

对于更复杂或特定的表格，可以使用`build_pattern_config()`函数自定义模式配置，然后创建`PdorPattern`实例：

```python
from pdor import PdorPattern, build_pattern_config

custom_config = build_pattern_config(
    table_headers=["自定义列1", "自定义列2", "自定义列3"],
    key_column=0,
    min_rows=3,
    min_columns=3,
    header_row=0,
    data_start_row=1,
    threshold_method='adaptive',
    contrast_adjust=1.8,
    denoise=True,
    border_removal=5,
    deskew=True,
    lang='chi_sim+eng',
    psm=6,
    oem=3,
    whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz-+. ',
    trim_whitespace=True,
    merge_adjacent_cells=True,
    pattern_corrections=[],
    column_types={},
    column_patterns={},
    min_area=10000,
    aspect_ratio_range=[0.5, 5.0],
    line_detection_method='hough'
)

custom_pattern = PdorPattern("自定义表格模式", custom_config)
```

`build_pattern_config()`函数需要以下参数:

| 参数名                     | 类型    | 说明                                                                       |
|-------------------------|-------|--------------------------------------------------------------------------|
| `dpi`                   | int   | 图片化的DPI                                                                  |
| `table_headers`         | list  | 表头字段列表，如 ["功能", "位置", "器件"]                                              |
| `key_column`            | int   | 关键列索引（作为字典键的列）                                                           |
| `min_rows`              | int   | 最小行数                                                                     |
| `min_columns`           | int   | 最小列数                                                                     |
| `header_row`            | int   | 表头行号                                                                     |
| `data_start_row`        | int   | 数据起始行                                                                    |
| `threshold_method`      | str   | 阈值处理方法，可选 'otsu' 或 'adaptive'                                            |
| `contrast_adjust`       | float | 对比度调整系数                                                                  |
| `denoise`               | bool  | 是否去噪                                                                     |
| `border_removal`        | int   | 移除边框像素数                                                                  |
| `deskew`                | bool  | 是否纠正倾斜                                                                   |
| `lang`                  | str   | OCR语言，如'chi_sim+eng'表示简体中文和英文                                            |
| `psm`                   | int   | 页面分割模式，6表示假定为单个文本块                                                       |
| `oem`                   | int   | OCR引擎模式，3表示只使用LSTM引擎                                                     |
| `whitelist`             | str   | 允许的字符集，如'0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz' |
| `trim_whitespace`       | bool  | 是否去除首尾空白                                                                 |
| `merge_adjacent_cells`  | bool  | 是否合并相邻单元格                                                                |
| `pattern_corrections`   | list  | 正则表达式替换规则列表，每项为 (pattern, replacement) 元组                                |
| `column_types`          | dict  | 列数据类型字典，键为列索引，值为类型名称                                                     |
| `column_patterns`       | dict  | 列正则表达式匹配模式字典，键为列索引，值为正则表达式                                               |
| `min_area`              | int   | 最小表格面积                                                                   |
| `aspect_ratio_range`    | list  | 表格宽高比范围 [min_ratio, max_ratio]                                           |
| `line_detection_method` | str   | 线检测方法，可选 'hough' 或 'contour'                                             |

创建好的`PdorPattern`实例可以通过`__repr__`方法查看详细配置.  

### Pdor单元  

Pdor的核心功能在提供的类`PdorUnit`中. 通过构造`PdorUnit`实例并调用`parse()`方法实现OCR识别过程.

`PdorUnit`是一个只读对象，构造后不允许修改其核心属性. 构造需要提供要进行识别的PDF文件名和匹配的`PdorPattern`. 文件名存储在`file`属性中, 模式存储在`pattern`属性中.

***主要属性***:
- `file`: PDF文件名 (只读)
- `pattern`: 使用的识别模式 (只读)
- `result`: 识别结果，解析完成后可访问 (只读)
- `time_cost`: 解析耗时，单位为秒 (只读)

***主要方法***:
- `parse()`: 执行OCR识别解析，支持关键字参数`print_repr=True`实时打印进度
- `output()`: 将结果导出到`simpsave`文件，`print_repr=True`打印回显信息
- `__repr__()`: 打印单元详细信息，包括构造信息、状态信息及识别的表格数据

***示例***:  

```python
import simpsave as ss
from pdor import PdorUnit, PDOR_PATTERNS

# 创建一个Pdor单元实例，使用端子排表格模式
pdor_unit = PdorUnit('terminal.pdf', PDOR_PATTERNS.TERMINAL_BLOCK_1)

# 执行解析(打印回显)
pdor_unit.parse(print_repr=True)

# 打印单元的详细信息
print(pdor_unit)

# 获取结果
result = pdor_unit.result

# 查看解析耗时
print(f"解析用时: {pdor_unit.time_cost:.2f}秒")

# 持久化结果:输出至terminal.ini(打印回显)
pdor_unit.output(print_repr=True)

# 后续可以通过simpsave读取结果
same_result = ss.read('Pdor Result', file='terminal.ini')
```

执行解析后，`result`字典包含以下结构：
- `file_name`: 原PDF文件名
- `total_pages`: 总页数
- `tables`: 识别出的表格数据，按页码和表格编号组织

通过`PdorUnit`类，Pdor能够将PDF中的表格数据自动提取并转换为结构化的字典数据.  