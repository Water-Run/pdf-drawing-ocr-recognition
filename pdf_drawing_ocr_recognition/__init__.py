r"""
:author: WaterRun
:date: 2025-04-11
:file: __init__.py
:description: Pdor初始化
"""
from pdor_env_check import check_env
from pdor_unit import PdorUnit as Pdor

__all__ = [check_env, Pdor]
__version__ = "0.1"
__author__ = "WaterRun"
