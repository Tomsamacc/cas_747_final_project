import logging
import os
import sys

import time

current_time = time.strftime("%Y%m%d_%H_%M", time.localtime())

_LOGGER_NAME = "lsgnn"
_LOG_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "results", "logs"))
_DEFAULT_FILE = f"lsgnn_{current_time}.log"


def _safe_log_subdir(name):
    if not name:
        return None
    s = str(name).strip().lower().replace("_", "-")
    s = s.replace(os.sep, "-").replace("/", "-")
    return s or None


def setup_logging(level=logging.INFO, file_name=_DEFAULT_FILE, subdir=None):
    log = logging.getLogger(_LOGGER_NAME)
    log.setLevel(level)
    if log.handlers:
        return log

    fmt = logging.Formatter(
        "%(asctime)s %(levelname)s [%(name)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    out = logging.StreamHandler(sys.stdout)
    out.setFormatter(fmt)
    log.addHandler(out)

    base_dir = _LOG_DIR
    sub = _safe_log_subdir(subdir)
    if sub:
        base_dir = os.path.join(_LOG_DIR, sub)
    os.makedirs(base_dir, exist_ok=True)
    path = os.path.join(base_dir, file_name)
    fh = logging.FileHandler(path, encoding="utf-8")
    fh.setFormatter(fmt)
    log.addHandler(fh)

    log.propagate = False
    return log


def get_logger(suffix=None):
    log = logging.getLogger(_LOGGER_NAME)
    if not log.handlers:
        setup_logging()
    if suffix:
        return logging.getLogger(f"{_LOGGER_NAME}.{suffix}")
    return log
