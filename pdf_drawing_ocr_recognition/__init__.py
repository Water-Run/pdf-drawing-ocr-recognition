r"""
:author: WaterRun
:date: 2025-04-12
:file: __init__.py
:description: Pdor初始化
"""
from pdor_env_check import check_env
from pdor_unit import PdorUnit as Pdor
from pdor_pattern import PdorPattern, build_pattern_config, load, save
from pdor_out import PdorOut

__all__ = [check_env, Pdor, PdorPattern, build_pattern_config, load, save, PdorOut]
__version__ = "0.1"
__author__ = "WaterRun"

"""test"""
if __name__ == '__main__':
    ...
