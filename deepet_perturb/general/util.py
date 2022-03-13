import logging

LOG_FORMAT = "[%(asctime)s] [PID %(process)d] %(levelname)s:\t%(filename)s:%(funcName)s():%(lineno)s:\t%(message)s"

def get_logger(logger_id=__name__):
    logging.basicConfig(format=LOG_FORMAT, level=logging.INFO)
    logger = logging.getLogger(logger_id)
    return logger
