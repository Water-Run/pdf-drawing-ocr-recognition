# pdf-drawing-ocr-recognition: Pdor  

定制项目,一个Python库,实现将PDF输入的指定模式的表格进行数据化读取.  

## 使用前准备  

### 环境  

- Windows
- Python 3.10+  

## 依赖库  

使用`pip`安装以下依赖库:  

- simpsave  

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

Pdor的核心功能在提供的类`Pdor`中.通过构造`Pdor`实例并调用`parse()`方法实现OCR识别过程.

`Pdor`实例的构造需要进行识别的PDF文件名,和匹配的`PdorPattern`.文件名存储在`file`属性中,模式存储在`pattern`属性中.  
`Pdor`单元实现的`__repr__`方法,可以打印出单元信息.  
调用`parse()`方法,进行OCR识别,将关键字参数`print_repr`赋值为`True`实时打印回显.识别后的结果存储于`result`属性中,为一个字典.  
解析用时将存储在`time_cost`属性中(秒).  
调用`output()`方法将数据导出到`simpsave`文件中,也支持关键字参数`print_repr`打印回显,存储在和PDF文件同路径同名`.ini`文件.  

***示例***:  

```python
import simpsave as ss
from pdor import Pdor, PDOR_PATTERNS

first_pdor = Pdor('111.PDF') # 假设我们需要读取111.PDF
first_pdor.parse(print_repr=True) # 执行解析(打印回显)
print(first_pdor) # 打印first_pdor

result = first_pdor.result # 获取结果
first_pdor.output(print_repr=True) # 持久化结果:输出至111.ini(打印回显)

same_result = ss.read('Pdor Result', file='111.ini') # 读取simpsave结果,和result一样
```

### 结果字典  
