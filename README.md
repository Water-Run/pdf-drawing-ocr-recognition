# pdf-drawing-ocr-recognition  

一个开源的定制项目,一个Python库,实现将PDF输入的指定模式的表格进行数据化读取.  

## 使用教程  

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
