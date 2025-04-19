r"""
PDOR模式
:author: WaterRun
:time: 2025-04-19
:file: pdor_pattern.py
"""

import inspect
import simpsave as ss

from .pdor_exception import *
from .pdor_utils import get_config_path


class PdorPattern:
    r"""
    Pdor模式单元
    :param name: 模式名称
    :param prompt: llm的prompt
    :param sub_imgs: 子图定义，每一项为四个整数，表示子图在原图的上下左右切分位置百分比。
                     如 [[10, 20, 30, 40]] 表示有一张子图,且子图占原图从顶部 10% 到底部 20%，左侧 30% 到右侧 40% 的区域。
    :param dpi: 转换图片的DPI
    """

    def __init__(self, name: str, prompt: str, dpi: int, sub_imgs: list[list[float, float, float, float]]):

        if (not isinstance(name, str)) or len(name) == 0:
            raise PdorInvalidPatternError(
                message='name (非空字符串)'
            )
        self._name = name

        if not isinstance(prompt, str):
            raise PdorInvalidPatternError(
                message='prompt (字符串)'
            )
        self._prompt = prompt

        if (not isinstance(dpi, int)) and 72 <= dpi <= 1400:
            raise PdorInvalidPatternError(
                message='dpi (72-1400的整数)'
            )
        self._dpi = dpi

        if not isinstance(sub_imgs, list):
            raise PdorInvalidPatternError(
                message='sub_imgs (列表)'
            )
        for sub_img in sub_imgs:

            if not len(sub_img) == 4:
                raise PdorInvalidPatternError(
                    message='sub_imgs (子列表长度为四)'
                )

            if not all(isinstance(percentage, float) for percentage in sub_img):
                raise PdorInvalidPatternError(
                    message='sub_imgs (子列表元素为浮点数)'
                )

            if not all(0 <= x <= 100 for x in sub_img):
                raise PdorInvalidPatternError(
                    message='sub_imgs (百分比需要在0-100之间)'
                )

            y1, y2, x1, x2 = sub_img

            if y1 >= y2 or x1 >= x2:
                raise PdorInvalidPatternError(
                    message='sub_imgs (范围无效)'
                )

        self._sub_imgs = sub_imgs or [[0, 100, 0, 100]]

    @property
    def name(self) -> str:
        r"""
        返回模式名称
        :return: 模式名称
        """
        return self._name

    @property
    def prompt(self) -> str:
        r"""
        返回模式Prompt
        :return: 模式Prompt
        """
        return self._prompt

    @property
    def dpi(self) -> int:
        r"""
        返回模式DPI
        :return: 模式DPI
        """
        return self._dpi

    @property
    def sub_imgs(self) -> list[list[float, float, float, float]]:
        r"""
        返回模式子图定义
        :return: 模式子图定义列表
        """
        return self._sub_imgs

    def _is_internal_call(self):
        r"""
        判断当前调用是否来自类内部方法
        :return: 如果调用来自内部方法则返回True，否则返回False
        """
        frame = inspect.currentframe().f_back.f_back

        if frame is None:
            return False

        calling_self = frame.f_locals.get('self')

        is_internal = calling_self is self and frame.f_code.co_filename == __file__

        return is_internal

    def __setattr__(self, name, value):
        r"""
        属性设置拦截器，保证对象的只读特性
        :param name: 属性名
        :param value: 属性值
        :raise PdorAttributeModificationError: 如果在初始化后尝试修改受保护属性
        """
        if (not hasattr(self, '_initialized') or name == '_initialized' or name not in
                {"_name", "_prompt", "_dpi", "_sub_imgs"}):
            super().__setattr__(name, value)
        elif self._is_internal_call():
            super().__setattr__(name, value)
        else:
            raise PdorAttributeModificationError(
                message=name
            )

    def __delattr__(self, name):
        r"""
        属性删除拦截器，防止删除核心属性
        :param name: 要删除的属性名
        :raise PdorAttributeModificationError: 如果尝试删除受保护属性
        """
        if name in {"_name", "_prompt", "_dpi", "_sub_imgs"}:
            raise PdorAttributeModificationError(
                message=name
            )
        super().__delattr__(name)

    def __repr__(self) -> str:
        r"""
        返回Pdor模式的字符串表示
        :return: 模式的字符串显示
        """
        result = (f"[Pdor模式]\n"
                  f"名称: {self.name}\n"
                  f"DPI: {self.dpi}\n"
                  f"Prompt: \n"
                  f"{self.prompt}\n"
                  f"子图: \n")
        for index, sub_img in enumerate(self.sub_imgs):
            result += f"{index}: {sub_img[0]} %, {sub_img[1]} %, {sub_img[2]} %, {sub_img[3]} %\n"
        return result


def save(pdor_pattern: PdorPattern) -> None:
    r"""
    保存PdorPattern至配置文件.
    保存的键名和PdorPattern单元名一致.
    :param pdor_pattern: 待保存的PdorPattern
    :return: None
    """
    ss.write(pdor_pattern.name,
             {'prompt': pdor_pattern.prompt, 'dpi': pdor_pattern.dpi, 'sub imgs': pdor_pattern.sub_imgs},
             file=get_config_path())


def load(name: str) -> PdorPattern:
    r"""
    从配置文件中读取PdorPattern.
    :param name: 待读取的PdorPattern名称
    :return: 根据读取内容构造的PdorPattern
    """
    pattern_config = ss.read(name, file=get_config_path())
    return PdorPattern(name, pattern_config['prompt'], pattern_config['dpi'], pattern_config['sub imgs'])


if __name__ == '__main__':
    """不暴露方法"""
    def write_preset_pattern() -> None:
        r"""
        写入预设模式
        :return: None
        """
        if __name__ == '__main__':

            """写入预设模式"""
            patterns = [
                PdorPattern(
                    name="700501-8615-73-04 第四串W4Q1断路器LCP柜接线图二",
                    prompt=(
                        "你是一位擅长图像结构识别的模型，请对发送给你的图片中的表格进行 OCR 并提取结构化信息。\n\n"
                        "强调:我不需要你利用Python代码并运行进行OCR,而是利用你的视觉能力直接OCR.你的视觉OCR能力更加强大.\n"
                        "### 任务说明：\n"
                        "表格分为多个主类，每类下包含若干行连接信息。\n"
                        "每一行包含以下字段：\n"
                        "- 功能字段（function）：如 'CB' 或其他描述\n"
                        "- 位置字段（position）：如 '121', '122' 等\n"
                        "- 器件字段（component）：如 'YB2(C)', '47T2' 等\n"
                        "- 端子号字段（terminal）：如 'CB(A) X1', 'CB(B) X2' 等\n\n"
                        "### 输出要求：\n"
                        "- 返回一个合法的 Python 字典对象，且**仅返回字典内容**（即整个返回只包含一个 { 和一个 }）。\n"
                        "- 顶层键为 'connections'，值为一个以功能字段为键的字典。\n"
                        "- 每个功能字段对应一个连接列表，每项为一个字典，包含 position、component 和 terminal 字段。\n\n"
                        "### 注意事项：\n"
                        "- 请严格返回一个合法的 Python 字典对象，不添加解释或额外输出。\n"
                        "- 字符串字段使用单引号，数字字段使用整数格式。\n"
                        "- 如果某字段为空，则忽略该字段，不输出 None 或空值。\n\n"
                        "### 示例：\n"
                        "{\n"
                        " 'CB': [\n"
                        "     { 'position': '121', 'component': 'YB2(C)', 'terminal': 'CB(A) X1' },\n"
                        "     { 'position': '122', 'component': '47T2', 'terminal': 'CB(B) X2' }\n"
                        " ]\n"
                        "}"
                    ),
                    dpi=1390,
                    sub_imgs=[
                        [34.45, 54.57, 7.44, 12.09],  # 子图1
                        [34.45, 67.89, 16.44, 21.48],  # 子图2
                        [34.45, 58.30, 25.41, 30.48],  # 子图3
                        [34.45, 67.89, 34.71, 39.72],  # 子图4
                        [34.45, 67.89, 43.71, 48.72],  # 子图5
                        [34.45, 64.16, 52.58, 57.62],  # 子图6
                        [34.45, 64.16, 61.58, 66.62],  # 子图7
                        [34.45, 58.84, 80.65, 85.43],  # 子图8
                        [34.45, 60.30, 79.58, 86.62],  # 子图9
                        [34.45, 44.64, 88.68, 93.64],  # 子图10
                        [47.73, 53.67, 88.68, 93.64],  # 子图11
                    ],
                ),
                PdorPattern(
                    name="duanzipai",
                    prompt=(
                        "你是一位擅长图像结构识别的模型，请对发送给你的图片中的表格进行 OCR，并提取结构化信息。\n\n"
                        "强调:我不需要你利用Python代码并运行进行OCR,而是利用你的视觉能力直接OCR.你的视觉OCR能力更加强大.\n"
                        "### 任务说明：\n"
                        "表格分为多个主类（如 '2-4C2D'、'2-4BS'、'2-4WD'、'51D'、'52D' 等），每类下包含若干行连接信息。\n"
                        "每一行包含以下字段：\n"
                        "- 起点字段（from）：（可选）段字牌子编号，如 '2-4YLP1:1'、'51n:01:03' 等\n"
                        "- 编号字段（index）：数字序号，表示在该主类中的顺序\n"
                        "- 终点字段（to）：（可选）另一端段字牌子编号，如 '52D:3'。若无终点则忽略。\n\n"
                        "### 输出要求：\n"
                        "- 返回一个合法的 Python 字典对象，且**仅返回字典内容**（即整个返回只包含一个 { 和一个 }）。\n"
                        "- 顶层键为 'duanzipai'，值为一个以主类名称为键的字典。\n"
                        "- 每个主类对应一个连接列表，每项为一个字典，包含 from、index 和可选的 to 字段。\n\n"
                        "### 注意事项：\n"
                        "- 请严格返回一个合法的 Python 字典对象，不添加解释或额外输出。\n"
                        "- 字符串字段使用单引号，数字字段使用整数格式。\n"
                        "- 如果某字段为空，则忽略该字段，不输出 None 或空值。\n\n"
                        "### 示例：\n"
                        "{\n"
                        " 'duanzipai': {\n"
                        "     '2-4C2D': [\n"
                        "         { 'from': '2-4YLP1:1', 'index': 1, 'to': '52D:3' },\n"
                        "         { 'from': '2-4n-10:18', 'index': 3, 'to': '52D:9' }\n"
                        "     ],\n"
                        "     '2-4BS': [\n"
                        "         { 'index': 1 },\n"
                        "         { 'from': '2-4n-11:24', 'index': 2 }\n"
                        "     ]\n"
                        " }\n"
                        "}"
                    ),
                    dpi=450,
                    sub_imgs=[
                        [5.60, 45.20, 47.52, 64.93],  # 第一张子图
                        [5.60, 93.90, 74.45, 91.76],  # 第二张子图
                    ],
                ),
                PdorPattern(
                    name="700501-8615-72-12 750kV 第四串测控柜A+1端子排图左",
                    prompt=(
                        "你是一位擅长图像结构识别的模型，请对发送给你的图片中的表格进行 OCR 并提取结构化信息。\n\n"
                        "强调:我不需要你利用Python代码并运行进行OCR,而是利用你的视觉能力直接OCR.你的视觉OCR能力更加强大.\n"
                        "### 任务说明：\n"
                        "表格分为多个端子组，每组包含若干端子信息。\n"
                        "每一行包含以下字段：\n"
                        "- 起点字段（from）：如 '-X105:1', '-X108:1' 等\n"
                        "- 编号字段（index）：如 '1', '2', '3' 等\n"
                        "- 终点字段（to）：如 '-H1.6.X1:1', '-H1.6.X1:3' 等\n\n"
                        "### 输出要求：\n"
                        "- 返回一个合法的 Python 字典对象，且**仅返回字典内容**（即整个返回只包含一个 { 和一个 }）。\n"
                        "- 顶层键为 'terminals'，值为一个以端子组名称为键的字典。\n"
                        "- 每个端子组对应一个连接列表，每项为一个字典，包含 from、index 和 to 字段。\n\n"
                        "### 注意事项：\n"
                        "- 请严格返回一个合法的 Python 字典对象，不添加解释或额外输出。\n"
                        "- 字符串字段使用单引号，数字字段使用整数格式。\n"
                        "- 如果某字段为空，则忽略该字段，不输出 None 或空值。\n\n"
                        "### 示例：\n"
                        "{\n"
                        " 'terminals': {\n"
                        "     'X106': [\n"
                        "         { 'from': '-X105:1', 'index': 1, 'to': '-H1.6.X1:1' },\n"
                        "         { 'from': '-X108:1', 'index': 2, 'to': '-H1.6.X1:3' }\n"
                        "     ]\n"
                        " }\n"
                        "}"
                    ),
                    dpi=1200,
                    sub_imgs=[
                        [6.85, 81.44, 45.64, 48.94],  # 图1
                        [6.85, 86.81, 53.06, 56.39],  # 图2
                    ],
                ),
            ]

            for pattern in patterns:
                save(pattern)
