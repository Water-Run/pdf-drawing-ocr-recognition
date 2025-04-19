# pdf-drawing-ocr-recognition: Pdor  

定制项目,一个Python库,实现将PDF输入的指定模式的表格进行结构化读取(调用LLM OCR).  

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
- opencv-python (cv2)
- pandas
- toml
- pyyaml
- requests

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

提供了一个`check_env()`方法检测`pdor`是否可用.  

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

### 配置  

Pdor的运行包含以下配置项:  

- `max_try`: LLM的最大重试次数  
- `api_url`: LLM的API地址  
- `api_key`: LLM的API KEY  
- `llm_model`: 使用的LLM视觉模型  

对于每个配置项, 提供`getter`和`setter`进行配置.  

> 仅当通过自检时可修改配置  

### Pdor模式  

Pdor模式定义要识别的表格结构.每个模式都是一个`PdorPattern`实例.   

#### 读取模式  

使用`pattern.load`方法读取预设的模式.接受一个参数,即待读取的模式名称.  
目前预设了三种模式:  
- `700501-8615-72-12 750kV 第四串测控柜A+1端子排图左`  
- `700501-8615-73-04 第四串W4Q1断路器LCP柜接线图二`  
- `duanzipai`  

***示例:***  
```python
import pdor
pattern = pdor.pattern.load('700501-8615-73-04 第四串W4Q1断路器LCP柜接线图二')
```

> 可以打印Pdor模式  

#### 自行构建和保存模式  

可以自行实例`PdorPattern`创建模式.需要以下参数:  
- `name`: `str` 模式的名称
- `prompt`: `str` 发送大模型的Prompt  
- `dpi`: `int` PDF转换为图片时的DPI
- `sub_imgs`: `list[list[float, float, float, float]]` 子图列表.每个子图应该是一个表格.列表每一项都是一个由四个浮点数构成的子列表,分别对应子表位于原图的上,下,左,右百分比位置.如果子图列表为空,会自动生成一个包含整张图片的默认列表.  

使用`save`方法进行存储.接受一个参数,即要写入的`PdorPattern`,名称和模式名称保持一致.  

> Pdor模式是只读的  

### Pdor单元  

创建`Pdor`单元的实例以进行OCR解析.  
实例化`Pdor`需要两个参数:  

- 构造的PDF文件路径  
- 构造的Pdor模式  

可以打印Pdor单元格式化显示信息.Pdor单元是只读的.  
调用`parse()`方法执行Pdor单元解析.可以通过开关`print_repr`关键字参数控制回显.  
调用`is_parsed()`方法,可以判断是否已经解析.  
在成功解析后,访问`result`属性即可获取结果.  
其它可访问的属性包括:  

- `file`: 构造的PDF文件路径  
- `pattern`: 构造的Pdor模式  
- `time_cost`: 解析的用时  

***示例***:  

```python
import pdor
from pdor import Pdor

# 构造单元
unit = Pdor("示例PDF.pdf", pdor.pattern.load("示例模式"))

# 打印单元信息
print(unit)

# 执行解析
unit.parse()

# 访问结果和解析耗时
print(f'结果: {unit.result}, 耗时: {unit.time_cost}')
```

### 结果  

Pdor的结果存储在Pdor实例的`result`属性中.  
仅当解析完毕后才可访问该属性.  

#### 输出结果  

Pdor封装了常用的输出方式在`Out`类中.  

支持的输出类型包括:  

- PLAIN_TEXT  
- MARKDOWN  
- SIMPSAVE  
- JSON  
- YAML  
- XML  
- TOML
- PYTHON  

> 默认输出为`SIMPSAVE`  

`Out`是一个静态类,其中的静态方法`out()`自行输出.输出在构造单元的PDF同路径同名文件下.接受三个参数:  

- `pdor`: 待输出的Pdor单元  
- `out_type`: 输出的类型.封装在`Out.TYPE`枚举中  
- `print_repr`: 开关回显(关键字参数, 缺省值为`True`)  

***示例***
```python
import pdor

... # 假设我们前面获得了一个Pdor单元unit

Pdor.Out.out(unit, Pdor.Out.TYPE.SIMPSAVE) # 输出至simpsave ini
Pdor.Out.out(unit, Pdor.Out.TYPE.HTML, print_repr=False) # 输出至网页,并关闭回显  
```
