r"""
PDOR异常

:author: WaterRun
:time: 2025-04-11
:file: pdor_exception.py
"""


class PdorException(Exception):
    r"""
    PDOR异常基类
    所有PDOR项目中的异常都应继承自此类
    """

    def __init__(self, message: str = '') -> None:
        r"""
        初始化PDOR异常
        :param message: 异常信息参数，可在异常消息中显示
        """
        self.message = message
        super().__init__(self.__str__())

    def __str__(self) -> str:
        r"""
        返回异常的字符串表示
        :return: 格式化的异常消息字符串
        """
        if self.message:
            return f"{self.__class__.__name__}: {self.message}"
        return f"{self.__class__.__name__}"


class PdorPDFNotExistError(PdorException):
    r"""
    PDF文件不存在异常
    当尝试访问不存在的PDF文件时抛出
    """

    def __str__(self) -> str:
        r"""
        返回PDF不存在异常的字符串表示
        :return: 格式化的PDF不存在异常消息，包含文件路径
        """
        return f"{self.__class__.__name__}: 文件 `{self.message}` 不存在"


class PdorPDFReadError(PdorException):
    r"""
    PDF文件读取异常
    当读取PDF出现异常时抛出
    """

    def __str__(self) -> str:
        r"""
        返回读取PDF异常的字符串表示
        :return: 格式化的PDF读取异常消息
        """
        return f"{self.__class__.__name__}: PDF读取异常 `{self.message}`"


class PdorImagifyError(PdorException):
    r"""
    PDF图片转换异常
    当将读取的PDF转换为图片时出现异常时抛出
    """

    def __str__(self) -> str:
        r"""
        返回PDF图片转换异常的字符串表示
        :return: 格式化的PDF图片转换异常消息
        """
        return f"{self.__class__.__name__}: PDF图片转换异常 `{self.message}`"


class PdorUnparsedError(PdorException):
    r"""
    Pdor未解析异常
    当尝试访问未解析的Pdor单元的结果时抛出此异常
    """

    def __str__(self) -> str:
        r"""
        返回未解析异常的字符串表示
        :return: 格式化的未解析异常消息
        """
        return f"{self.__class__.__name__}: 单元为解析"


class PdorAttributeModificationError(PdorException):
    r"""
    Pdor属性修改异常
    当尝试修改或删除Pdor单元的受保护属性时抛出此异常
    """

    def __str__(self) -> str:
        r"""
        返回属性修改异常的字符串表示
        :return: 格式化的属性修改异常消息
        """
        return f"{self.__class__.__name__}: Pdor单元是只读的, 不可修改属性`{self.message}`"
