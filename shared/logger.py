"""统一 logger 工厂 — 控制台优先用 rich (彩色), 可选写入文件。"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

try:
    from rich.logging import RichHandler  # type: ignore
    _HAS_RICH = True
except Exception:  # pragma: no cover
    _HAS_RICH = False


def get_logger(
    name: str,
    log_file: Optional[Path] = None,
    level: str = "INFO",
) -> logging.Logger:
    """返回一个配置好的 logger。重复调用同名 logger 不会叠加 handler。

    Args:
        name: logger 名 (建议用模块路径, 例如 "shared.dataset")
        log_file: 若给定, 同时写入该文件 (纯文本, 无颜色)
        level: 日志级别字符串 (DEBUG/INFO/WARNING/ERROR/CRITICAL)
    """
    logger = logging.getLogger(name)
    logger.setLevel(getattr(logging, level.upper(), logging.INFO))
    logger.propagate = False

    if logger.handlers:
        return logger

    if _HAS_RICH:
        console_handler: logging.Handler = RichHandler(
            rich_tracebacks=True,
            show_path=False,
            markup=False,
        )
        console_handler.setFormatter(logging.Formatter("%(message)s"))
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
    logger.addHandler(console_handler)

    if log_file is not None:
        log_file = Path(log_file)
        log_file.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(
            logging.Formatter("%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                              datefmt="%Y-%m-%d %H:%M:%S")
        )
        logger.addHandler(file_handler)

    return logger


if __name__ == "__main__":
    tmp_log = Path("results_v2/sanity_check/_logger_test.log")
    log = get_logger("shared.logger.test", log_file=tmp_log, level="DEBUG")
    log.debug("debug 调试")
    log.info("info 流程")
    log.warning("warning 注意")
    log.error("error 错误演示 (非真错)")
    print(f"[logger] rich available: {_HAS_RICH}")
    print(f"[logger] file written: {tmp_log.exists()}, size={tmp_log.stat().st_size if tmp_log.exists() else 0}")
