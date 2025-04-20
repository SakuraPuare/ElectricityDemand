import sys
import logging
import os
from datetime import datetime
from loguru import logger

# 定义 InterceptHandler 以重定向标准 logging 到 Loguru
class InterceptHandler(logging.Handler):
    """
    将标准库 logging 的日志记录转发给 Loguru。
    """
    def emit(self, record):
        # 尝试获取相应的 Loguru 级别
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        # 回溯调用栈以找到正确的日志发起位置
        frame, depth = logging.currentframe(), 2
        while frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1

        logger.opt(depth=depth, exception=record.exc_info).log(level, record.getMessage())

def setup_logger(log_file_prefix: str = "app", logs_dir: str = "logs", level: str = "INFO"):
    """
    配置 Loguru 日志记录器。

    支持文件输出、彩色控制台输出，并拦截标准库 logging。

    Args:
        log_file_prefix (str): 日志文件名前缀 (e.g., 'download_data', 'data_analysis').
        logs_dir (str): 存放日志文件的目录。
        level (str): 最低日志记录级别 (e.g., "DEBUG", "INFO", "WARNING").
    """
    logger.remove() # 移除默认处理器，确保从干净状态开始配置

    # 确保日志目录存在
    os.makedirs(logs_dir, exist_ok=True)

    # 构建日志文件路径
    log_file_name = f"{log_file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = os.path.join(logs_dir, log_file_name)

    # 定义通用的日志格式 (控制台会自动应用颜色标签)
    log_format = (
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}</cyan>:<cyan>{function}</cyan>:<cyan>{line}</cyan> - "
        "<level>{message}</level>"
    )

    # 添加文件日志处理器 (无颜色)
    # 启用队列 (enqueue=True) 以提高异步/多进程环境下的性能和安全性
    logger.add(
        log_file_path,
        rotation="10 MB",  # 日志文件达到 10MB 时轮转
        retention="7 days", # 保留最近 7 天的日志
        level=level.upper(), # 使用传入的级别
        enqueue=True,
        encoding="utf-8", # 明确编码
        colorize=False, # 文件日志不需要颜色
        format=log_format # 使用通用格式 (颜色标签在文件中会被忽略)
    )

    # 添加控制台 (stderr) 日志处理器 (有颜色)
    logger.add(
        sys.stderr,
        level=level.upper(), # 使用传入的级别
        colorize=True, # 控制台日志使用颜色
        format=log_format # 使用带颜色的格式
    )

    # 配置标准库 logging 使用 InterceptHandler
    # level=0 确保捕获所有级别的标准日志，然后由 loguru 处理器根据其配置的 level 过滤
    logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)

    # 使用配置的级别记录初始化信息
    # 由调用方记录初始化信息更合适
    # logger.log(level.upper(), f"Loguru 初始化完成。日志级别设置为: {level.upper()}")
    # logger.info(f"文件日志将写入: {log_file_path}")

    # Loguru logger 是全局单例，通常不需要返回 logger 实例
    # return logger

if __name__ == '__main__':
    # 测试日志设置 (使用 DEBUG 级别以便看到所有日志)
    setup_logger("test_log", level="DEBUG")
    # 初始化完成后，可以在调用方记录
    logger.info("Loguru 配置完成，开始测试日志记录...")

    logger.debug("这是一条来自 Loguru 的 DEBUG 信息。")
    logger.info("这是一条来自 Loguru 的 INFO 信息。")
    logger.warning("这是一条来自 Loguru 的 WARNING 信息。")
    logger.error("这是一条来自 Loguru 的 ERROR 信息。")
    try:
        1 / 0
    except ZeroDivisionError:
        logger.exception("Loguru 捕获到一个异常。")

    print("-" * 30 + " 测试标准库 logging 拦截 " + "-" * 30)
    # 测试标准库 logging 拦截
    std_logger = logging.getLogger("MyStdLogger")
    # 不需要为 std_logger 单独设置级别，basicConfig 的 level=0 会捕获
    # Loguru 的处理器会根据其配置的 level ("DEBUG") 决定是否显示
    std_logger.debug("这是来自标准库 logging 的 DEBUG 信息。")
    std_logger.info("这是来自标准库 logging 的 INFO 信息。")
    std_logger.warning("这是来自标准库 logging 的 WARNING 信息。")
    std_logger.error("这是来自标准库 logging 的 ERROR 信息。")
    try:
        1 / 0
    except ZeroDivisionError:
        # 标准库 logging 记录异常的方式
        std_logger.exception("标准库 logging 捕获到一个异常。") 