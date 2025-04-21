import sys
import logging
import os
import stat  # 用于设置文件权限
from datetime import datetime, UTC
from loguru import logger
from tqdm import tqdm  # <--- 导入 tqdm

# --- 全局异常处理钩子 ---


def handle_exception(exc_type, exc_value, exc_traceback):
    """
    全局异常钩子，使用 Loguru 记录未捕获的异常。
    """
    if issubclass(exc_type, KeyboardInterrupt):
        # 用户中断时不视为错误
        sys.__excepthook__(exc_type, exc_value, exc_traceback)
        return

    # 当从 excepthook 调用时，logger.exception 会自动获取当前的异常信息
    # 无需显式传递 exc_info=(exc_type, exc_value, exc_traceback)
    # 这可以避免传递不可序列化的 traceback 对象给 enqueue=True 的处理器
    logger.exception("未捕获的异常")

# --- 标准 logging 拦截器 ---


class InterceptHandler(logging.Handler):
    """
    将标准库 logging 的日志记录转发给 Loguru。
    """

    def emit(self, record):
        try:
            level = logger.level(record.levelname).name
        except ValueError:
            level = record.levelno

        frame, depth = logging.currentframe(), 2
        while frame is not None and frame.f_code.co_filename == logging.__file__:
            frame = frame.f_back
            depth += 1
        if frame is None:
            # 在某些特殊情况（如 Cython 或嵌入式 Python）下 frame 可能为 None
            # 提供一个回退深度
            depth = 3  # 增加一点深度尝试补偿

        # 使用 opt(depth=...) 确保 Loguru 能找到正确的调用栈信息
        # exception=record.exc_info 确保异常信息被传递
        logger.opt(depth=depth, exception=record.exc_info).log(
            level, record.getMessage())

# --- 安全文件打开器 ---


def secure_opener(file, flags):
    """
    自定义文件打开器，设置权限为 0o600 (仅所有者可读写)。
    """
    # os.O_CREAT | os.O_WRONLY | os.O_APPEND 是 loguru 默认使用的 flags
    # 模式 0o600: 所有者读写，其他用户无权限
    return os.open(file, flags, mode=stat.S_IRUSR | stat.S_IWUSR)


# --- 日志设置函数 ---
def setup_logger(
    log_file_prefix: str = "app",
    logs_dir: str = "logs",
    level: str = "INFO",
    console_level: str | None = None,  # 允许为控制台设置不同级别
    file_level: str | None = None,    # 允许为文件设置不同级别
    json_log_path: str | None = None,  # 可选：JSON 日志文件路径
    secure_permissions: bool = False,  # 可选：是否设置安全文件权限
    intercept_standard_logging: bool = True,  # 是否拦截标准 logging
    enable_global_exception_hook: bool = True,  # 是否启用全局异常捕获
    diagnose_backtrace: bool = True  # 控制 diagnose 和 backtrace (生产环境建议 False)
):
    """
    配置 Loguru 日志记录器，集成多种高级特性。

    Args:
        log_file_prefix (str): 日志文件名前缀。
        logs_dir (str): 存放日志文件的目录。
        level (str): 默认的最低日志记录级别。
        console_level (str | None): 控制台的最低级别，默认为 'level'.
        file_level (str | None): 文本日志文件的最低级别，默认为 'level'.
        json_log_path (str | None): 如果提供，则启用 JSON 格式日志到指定文件。
        secure_permissions (bool): 如果为 True, 设置文本日志文件权限为 0o600.
        intercept_standard_logging (bool): 是否拦截标准库 logging 的日志。
        enable_global_exception_hook (bool): 是否设置 sys.excepthook 来捕获未处理异常。
        diagnose_backtrace (bool): 是否为所有处理器启用 diagnose 和 backtrace.
                                    警告：在生产环境中 diagnose=True 可能泄露敏感数据！
    """
    logger.remove()  # 清理现有配置

    # 确定各 sink 的日志级别
    effective_console_level = console_level.upper() if console_level else level.upper()
    effective_file_level = file_level.upper() if file_level else level.upper()

    # --- 配置控制台日志 (使用 tqdm.write) ---
    log_format_console = (  # 稍微简化控制台格式
        "<green>{time:YYYY-MM-DD HH:mm:ss.SSS}</green> | "
        "<level>{level: <8}</level> | "
        "<cyan>{name}:{function}:{line}</cyan> - <level>{message}</level>"
    )
    # 使用 lambda 将消息传递给 tqdm.write，并确保消息末尾没有多余换行符
    # colorize=True 可以在 tqdm.write 中保持颜色
    logger.add(
        lambda msg: tqdm.write(msg, end=""),  # <--- 使用 tqdm.write
        level=effective_console_level,
        colorize=True,  # <--- 保持颜色
        format=log_format_console,
        backtrace=diagnose_backtrace,
        diagnose=diagnose_backtrace
    )

    # --- 配置文件日志 (文本) ---
    os.makedirs(logs_dir, exist_ok=True)
    log_file_name = f"{log_file_prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
    log_file_path = os.path.join(logs_dir, log_file_name)
    log_format_file = (  # 文件格式不需要颜色，但保留结构
        "{time:YYYY-MM-DD HH:mm:ss.SSS} | "
        "{level: <8} | "
        "{name}:{function}:{line} - {message}"  # 移除 <level> 标签避免重复
        # Loguru 会自动附加异常信息，无需显式 {exception}
    )
    file_opener = secure_opener if secure_permissions else None

    logger.add(
        log_file_path,
        level=effective_file_level,
        format=log_format_file,
        encoding="utf-8",
        rotation="10 MB",
        retention="7 days",
        enqueue=True,  # 启用队列以提高性能，特别是在多进程/线程环境中
        backtrace=diagnose_backtrace,
        diagnose=diagnose_backtrace,
        opener=file_opener  # 使用自定义 opener (如果需要)
    )
    logger.info(f"文本日志将写入：{log_file_path}")
    if secure_permissions:
        logger.info(f"已为文本日志文件设置安全权限 (0o600)。")

    # --- 配置文件日志 (JSON, 可选) ---
    if json_log_path:
        os.makedirs(os.path.dirname(json_log_path), exist_ok=True)
        logger.add(
            json_log_path,
            level=effective_file_level,  # 通常与文本文件级别相同
            serialize=True,  # <--- 启用 JSON 序列化
            encoding="utf-8",
            rotation="50 MB",  # JSON 可能更大，增加轮转大小
            retention="14 days",
            enqueue=True,
            backtrace=diagnose_backtrace,  # JSON 中也包含堆栈信息
            diagnose=diagnose_backtrace
        )
        logger.info(f"JSON 结构化日志将写入：{json_log_path}")

    # --- 添加自定义级别示例 (TRACE) ---
    # level no 5 is between DEBUG (10) and NOTSET (0)
    try:
        logger.level("TRACE", no=5, color="<fg #666666>", icon="✏️")  # 使用灰色和图标
        logger.debug("添加了自定义日志级别 'TRACE' (no=5)")
    except ValueError:
        # 级别已存在，忽略错误，但记录一下信息
        logger.trace("自定义日志级别 'TRACE' 已存在。")  # 使用 trace 级别，因为它可能在 INFO 之下

    # 确保 trace 函数总是绑定到 logger 实例，无论级别是新建还是已存在
    if not hasattr(logger, 'trace'):
        def trace(message, *args, **kwargs):
            logger.log("TRACE", message, *args, **kwargs)
        logger.trace = trace

    # --- 拦截标准库 logging ---
    if intercept_standard_logging:
        logging.basicConfig(handlers=[InterceptHandler()], level=0, force=True)
        logger.info("已拦截标准库 logging。")

    # --- 设置全局异常钩子 ---
    if enable_global_exception_hook:
        sys.excepthook = handle_exception
        logger.info("已设置全局异常处理钩子 (sys.excepthook)。")

    logger.log(level.upper(
    ), f"Loguru 初始化完成。默认级别：{level.upper()}. 控制台：{effective_console_level}. 文件：{effective_file_level}.")
    if diagnose_backtrace:
        logger.warning(
            "Diagnose 和 Backtrace 已启用。注意：在生产环境中 'diagnose=True' 可能泄露敏感数据！")


# --- 示例和测试 ---
if __name__ == '__main__':
    import time  # <--- 导入 time 用于测试

    # 配置日志记录器，启用 JSON 和安全权限
    setup_logger(
        log_file_prefix="test_advanced",
        logs_dir="logs",
        level="TRACE",  # 设置默认级别为我们最低的 TRACE
        console_level="INFO",  # 但控制台只显示 INFO 及以上
        json_log_path="logs/structured_log.json",
        secure_permissions=True,
        diagnose_backtrace=True  # 开发时设为 True
    )

    logger.info("--- 开始高级 Loguru 功能测试 ---")

    # 1. 测试不同级别 (包括自定义级别)
    logger.trace("这是一条 TRACE 级别的消息，文件和 JSON 可见，控制台不可见。")
    logger.debug("这是一条 DEBUG 级别的消息，文件和 JSON 可见，控制台不可见。")
    logger.info("这是一条 INFO 级别的消息，所有 sink 都可见。")
    logger.success("这是一条 SUCCESS 消息。")  # Loguru 内建级别
    logger.warning("这是一条 WARNING 消息。")
    logger.error("这是一条 ERROR 消息。")
    try:
        result = 1 / 0
    except ZeroDivisionError:
        logger.exception("记录一个捕获的异常。")

    # 2. 测试标准库 logging 拦截
    print("-" * 20 + " 测试标准 logging 拦截 " + "-" * 20)
    std_logger = logging.getLogger("MyStandardLogger")
    std_logger.setLevel(logging.DEBUG)  # 标准 logger 需要设置自己的级别
    std_logger.debug("标准库 DEBUG (文件/JSON 可见)")
    std_logger.info("标准库 INFO (所有 sink 可见)")
    std_logger.warning("标准库 WARNING")
    std_logger.error("标准库 ERROR")

    # 3. 测试上下文绑定 (bind)
    print("-" * 20 + " 测试 bind() " + "-" * 20)
    context_logger = logger.bind(ip="192.168.1.100", user_id="user123")
    context_logger.info("这条日志带有 IP 和 user_id 上下文。")
    context_logger.bind(request_id="req-abc").warning("可以链式绑定更多上下文。")

    # 4. 测试临时上下文 (contextualize)
    print("-" * 20 + " 测试 contextualize() " + "-" * 20)
    logger.info("进入任务上下文之前。")
    with logger.contextualize(task_name="DataProcessing", step=1):
        logger.info("在任务上下文内部，第一步。")
        # 在此 'with' 块内的所有日志都会自动包含 task_name 和 step
        logger.warning("任务处理中发生警告。")
    logger.info("离开任务上下文之后。")

    # 5. 测试动态补丁 (patch) - 例如添加 UTC 时间戳
    print("-" * 20 + " 测试 patch() " + "-" * 20)
    # 创建一个临时的 logger 实例来 patch，避免影响全局
    patched_logger = logger.patch(lambda record: record["extra"].update(
        utc_time=datetime.now(UTC).isoformat()))
    patched_logger.info("这条日志通过 patch 添加了 UTC 时间。")
    # 注意：patch 是添加到 logger 实例上的，原始 logger 不受影响
    logger.info("这条日志没有 UTC 时间 (来自原始 logger)。")

    # 6. 测试惰性求值 (opt(lazy=True))
    print("-" * 20 + " 测试 opt(lazy=True) " + "-" * 20)

    def expensive_calculation(n):
        logger.debug(f"--- 正在执行昂贵的计算 for {n} ---")
        # 模拟耗时操作
        time.sleep(0.1)
        return n * n

    # 因为我们设置了文件级别为 TRACE/DEBUG，这个 lambda 会被执行
    logger.opt(lazy=True).debug(
        "昂贵计算结果 (文件可见): {result}", result=lambda: expensive_calculation(10))
    # 如果控制台级别是 WARNING，这个 lambda 就不会执行
    logger.opt(lazy=True).info(
        "另一个惰性求值示例：{data}", data=lambda: expensive_calculation(5))

    # 7. 测试与 tqdm 的兼容性
    print("-" * 20 + " 测试 tqdm 兼容性 " + "-" * 20)
    logger.info("开始 tqdm 循环测试...")
    for i in tqdm(range(5), desc="测试进度条"):
        logger.info(f"tqdm 循环内日志 - 迭代 {i}")
        time.sleep(0.2)
    logger.info("tqdm 循环结束。")

    # 8. 测试全局异常处理 (触发一个未捕获的异常)
    print("-" * 20 + " 测试全局异常钩子 " + "-" * 20)
    logger.warning("接下来将故意触发一个未捕获的 ZeroDivisionError 来测试全局钩子...")
    # 注意：这会终止脚本执行，除非你在调用 main 的地方也加了 try...except
    # 但这里的目的是测试 excepthook
    value = 1 / 0

    logger.info("--- 测试结束 (如果未因异常退出) ---")  # 这行不会执行
