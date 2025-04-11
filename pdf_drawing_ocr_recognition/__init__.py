r"""
:author: WaterRun
:date: 2025-04-11
:file: __init__.py
:description: Pdor初始化
"""
from pdor_env_check import check_env
from pdor_unit import PdorUnit as Pdor
from pdor_pattern import PdorPattern, PDOR_PATTERNS

__all__ = [check_env, Pdor, PdorPattern, PDOR_PATTERNS]
__version__ = "0.1"
__author__ = "WaterRun"

"""test"""
if __name__ == '__main__':
    unit = Pdor('../tests/700501-8615-73-04 第四串W4Q1断路器LCP柜接线图二.PDF')
    unit.parse(print_repr=True)
    print(unit)
