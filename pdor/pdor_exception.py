r"""
PDOR异常

:author: WaterRun
:time: 2025-04-13
:file: pdor_exception.py
"""


class PdorException(Exception):
    r"""
    PDOR异常基类
    所有PDOR项目中的异常都应继承自此类
    """

    def __init__(self, message: str) -> None:
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
        return f"{self.__class__.__name__}: 单元未解析, {self.message}"


class PdorParsedError(PdorException):
    r"""
    Pdor已解析异常
    当尝试访问对已解析的Pdor单元再次解析时抛出此异常
    """

    def __str__(self) -> str:
        r"""
        返回未解析异常的字符串表示
        :return: 格式化的未解析异常消息
        """
        return f"{self.__class__.__name__}: 单元已解析, {self.message}"
    

class PdorOutUnsupportedTypeError(PdorException):
    r"""
    Pdor不支持的输出类型异常
    当在尝试输出一个不支持的类型时时抛出此异常
    """

    def __str__(self) -> str:
        r"""
        返回不支持输出的类型的字符串表示
        :return: 格式化的不支持输出的类型异常消息
        """
        return f"{self.__class__.__name__}: {self.message}"


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
        return f"{self.__class__.__name__}: 实例只读, 不可修改属性`{self.message}`"


class PdorMissingConfigError(PdorException):
    r"""
    Pdor找不到配置文件修改异常
    当找不到配置文件时抛出此异常
    """

    def __str__(self) -> str:
        r"""
        返回找不到配置文件异常的字符串表示
        :return: 格式化的找不到配置文件异常消息
        """
        return f"{self.__class__.__name__}: 配置文件丢失`{self.message}`"


class PdorInvalidPatternError(PdorException):
    r"""
    Pdor模式非法异常
    当构造模式参数非法时抛出此异常
    """

    def __str__(self) -> str:
        r"""
        返回模式非法异常的字符串表示
        :return: 格式化模式非法异常异常消息
        """
        return f"{self.__class__.__name__}: 非法模式参数 `{self.message}`"


class PdorLLMError(PdorException):
    r"""
    Pdor大模型异常
    当使用大模型出错时抛出此异常
    """

    def __str__(self) -> str:
        r"""
        返回LLM异常的字符串表示
        :return: 格式化LLM异常消息
        """
        return f"{self.__class__.__name__}: LLM错误 `{self.message}`"
