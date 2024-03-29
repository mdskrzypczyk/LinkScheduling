import logging

LOG_LEVEL = logging.ERROR
FORMAT = '%(levelname)s - %(name)s - %(message)s'
FILENAME = "log"
logging.basicConfig(format=FORMAT)


class LSLogger:
    """
    Simple logger module used in the LinkScheduling project, overrides the default logger but allows space for any
    additional desired functionalities.
    """
    def __init__(self, name=None):
        self.logger = logging.getLogger(name)
        self.logger.setLevel(LOG_LEVEL)

    def warning(self, message):
        self.logger.warning(message)

    def info(self, message):
        self.logger.info(message)

    def debug(self, message):
        self.logger.debug(message)

    def error(self, message):
        self.logger.error(message)

    def exception(self, message):
        self.logger.exception(message)
