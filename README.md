# pdf-drawing-ocr-recognition: Pdor  

定制项目,一个Python库,实现将PDF输入的指定模式的表格进行数据化读取.  

## 使用前准备  

### 环境  

- Windows
- Python 3.10+  

## 依赖库  

使用`pip`安装以下依赖库:  

- simpsave
- PyPDF2
- pdf2image
- numpy
- pytesseract
- opencv-python (cv2)
- pandas
- toml

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

### Pdor单元  

### 结果表和输出  
