import logging
from logging.handlers import TimedRotatingFileHandler



class Log():
    def set_log(self, filepath = "logs/log.log", level = 2, freq = "D", interval = 50, backup = 2, name = "log"):
        # define log format and date format
        format  = '%(asctime)s %(levelname)s %(message)s'
        datefmt = '%Y-%m-%d %H:%M:%S'

        # define log levels
        level_dict = {
            1: logging.DEBUG,
            2: logging.INFO,
            3: logging.ERROR,
            4: logging.WARNING,
            5: logging.CRITICAL,
        }

        fmt = logging.Formatter(format, datefmt) # create a log formatter       
        log_level = level_dict[level] # get log level based on the provided "level"

        # initialize the logger
        self.logger = logging.getLogger(name = name)
        self.logger.setLevel(log_level)

        # create a file handler for log rotation
        self.hdlr = TimedRotatingFileHandler(filename = filepath, when = freq, interval = interval, backupCount = backup, encoding = 'utf-8')
        self.hdlr.setFormatter(fmt)

        # add the file handler to the logger
        self.logger.addHandler(self.hdlr)

        return self.logger
    

    def shutdown(self):
        self.logger.removeHandler(self.hdlr)  # remove log handlers
        del self.logger, self.hdlr # delete logger instances