# pdf-drawing-ocr-recognition: Pdor  

一个开源的定制项目,一个Python库,实现将PDF输入的指定模式的表格进行数据化读取.  

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

`pdor`的使用依赖一些Python库.提供了一个`check_env()`方法检测当前是否可用.  

```python
import pdor
status, msg = pdor.check_env()
```
`status`将是一个布尔值.如果为真,则环境可用.  
如果环境不可用,`msg`是一个字符串列表,包括缺失的库,安装后再次检查.  

### Pdor单元  

Pdor的核心功能在提供的类`Pdor`中.通过构造`Pdor`实例并调用`parse()`方法实现OCR识别过程.  

#### 构造`Pdor`实例  

`Pdor`实例的构造需要一个参数: 需要进行识别的PDF文件名.文件名存储在`file`属性中.  
`Pdor`单元实现的`__repr__`方法,可以打印出单元信息.  
调用`parse()`方法,进行OCR识别.识别后的结果存储于`result`属性中,为一个字典.
调用`output()`方法将数据导出到`simpsave`文件中,存储在和PDF文件同路径同名`.ini`文件.  

***示例***:  

```python
from pdor import Pdor
first_pdor = Pdor('111.PDF') # 假设我们需要读取111.PDF
```

## 实践示例  

