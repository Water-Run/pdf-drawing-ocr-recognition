r"""
:author: WaterRun
:date: 2025-04-20
:file: __init__.py
:description: Pdor初始化
"""
from pdor.pdor_utils import (check_env,
                             set_llm_model, set_max_try, set_api_url, set_api_key,
                             get_llm_model, get_max_try, get_api_url, get_api_key)
from pdor.pdor_unit import PdorUnit as Pdor
import pdor.pdor_pattern as pattern
from pdor.pdor_out import PdorOut as Out

__all__ = [check_env, Pdor, pattern, Out]
__version__ = "0.1"
__author__ = "WaterRun"

if __name__ == '__main__':
    # 示例使用
    # unit_1 = PdorUnit('../tests/700501-8615-72-12 750kV 第四串测控柜A+1端子排图左.PDF', load('700501-8615-72-12 750kV 第四串测控柜A+1端子排图左'))
    # unit_1.parse()
    # print(unit_1.result)

    # unit_3 = PdorUnit('../tests/700501-8615-73-04 第四串W4Q1断路器LCP柜接线图二.PDF',
    # load('700501-8615-73-04 第四串W4Q1断路器LCP柜接线图二'))
    # unit_3.parse()

    unit_3 = Pdor('../tests/duanzipai.pdf', pattern.load('duanzipai'))
    unit_3.parse()
    print(unit_3)
    for output_type in pdor.Out.TYPE:
        print(f"正在测试输出类型: {output_type.name}")
        try:
            pdor.Out.out(unit_3, output_type)  # 回显开启
        except Exception as e:
            print(f"输出类型 {output_type.name} 失败: {e}")
