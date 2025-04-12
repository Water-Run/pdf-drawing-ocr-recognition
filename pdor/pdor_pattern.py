r"""
PDOR模式
:author: WaterRun
:time: 2025-04-13
:file: pdor_pattern.py
"""

import simpsave as ss


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
        self.name = name
        self.prompt = prompt
        self.dpi = dpi
        self.sub_imgs = sub_imgs or [[0, 100, 0, 100]]

    def __repr__(self) -> str:
        """
        返回Pdor模式的字符串表示
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


def save(pattern: PdorPattern, file: str) -> None:
    r"""
    保存PdorPattern.
    保存的键名和PdorPattern单元名一致.
    :param pattern: 待保存的PdorPattern
    :param file: 保存的simpsave文件名
    :return: None
    """
    ss.write(pattern.name, {'prompt': pattern.prompt, 'dpi': pattern.dpi, 'sub imgs': pattern.sub_imgs}, file=file)


def load(name: str, file: str) -> PdorPattern:
    r"""
    读取PdorPattern.
    :param name: 待读取的PdorPattern名称
    :param file: 读取的simpsave文件名
    :return: 根据读取内容构造的PdorPattern
    """
    pattern_config = ss.read(name, file=file)
    return PdorPattern(name, pattern_config['prompt'], pattern_config['dpi'], pattern_config['sub imgs'])


if __name__ == '__main__':

    """写入预设模式"""
    patterns = [
        PdorPattern(
            name="700501-8615-73-04 第四串W4Q1断路器LCP柜接线图二",
            prompt="",
            dpi=1390,
            sub_imgs=[
                [34.45, 54.57, 7.44, 12.09],  # 子图1
                [34.45, 67.89, 16.44, 21.48],  # 子图2
                [34.45, 58.30, 25.41, 30.48],  # 子图3
                [34.45, 67.89, 34.71, 39.72],  # 子图4
                [34.45, 67.89, 43.71, 48.72],  # 子图5
                [34.45, 64.16, 52.58, 57.62],  # 子图6
                [34.45, 64.16, 61.58, 66.62],  # 子图7
                [34.45, 58.84, 80.65, 85.43],  # 子图8 (修正)
                [34.45, 60.30, 79.58, 86.62],  # 子图9
                [34.45, 44.64, 88.68, 93.64],  # 子图10
                [47.73, 53.67, 88.68, 93.64],  # 子图11
            ]
        ),
        PdorPattern(
            name="duanzipai",
            prompt="",
            dpi=450,
            sub_imgs=[
                [5.60, 45.20, 47.52, 64.93],  # 第一张子图
                [5.60, 93.90, 74.45, 91.76],  # 第二张子图
            ]
        ),
        PdorPattern(
            name="700501-8615-72-12 750kV 第四串测控柜A+1端子排图左",
            prompt="",
            dpi=1200,
            sub_imgs=[
                [6.85, 81.44, 45.64, 48.94],  # 图1
                [6.85, 86.81, 53.06, 56.39],  # 图2
            ]
        ),
    ]

    for pattern in patterns:
        save(pattern, 'configs.ini')
