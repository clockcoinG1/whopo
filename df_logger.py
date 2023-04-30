import logging
import sys
from colorlog import ColoredFormatter


def setup_logger(name="PARSERO", log_level=logging.DEBUG, log_file="parsero.log"):
    logger = logging.getLogger(name)
    logger.setLevel(log_level)

    # Create a file handler to log messages to a file
    file_handler = logging.FileHandler(log_file)
    file_handler.setLevel(log_level)

    # Create a formatter with custom log format and colors
    log_format = "%(asctime)s - %(name)s - %(log_color)s%(levelname)s%(reset)s - %(message)s"
    formatter = ColoredFormatter(
        log_format,
        datefmt="%Y-%m-%d %H:%M:%S",
        reset=True,
        log_colors={
            "DEBUG": "cyan,bg_black",
            "INFO": "green,bg_white",
            "WARNING": "yellow,bg_black",
            "ERROR": "red,bg_black",
            "CRITICAL": "red,bg_black,bold",
            "TRACEBACK": "red,bg_black,bold",
            "EXCEPTION": "red,bg_black,bold",
            "SUCCESS": "green,bg_black,bold",
        },
        secondary_log_colors={
            "message": {
                "ERROR": "red,bg_black,bold",
                "CRITICAL": "red,bg_black,bold",
                "TRACEBACK": "red,bg_black,bold",
                "EXCEPTION": "red,bg_black,bold",
                "WARNING": "yellow,bg_black,bold",
                "SUCCESS": "green,bg_black,bold",
            },
        },
        style="%",
    )

    file_handler.setFormatter(formatter)

    logger.addHandler(file_handler)

    return logger
