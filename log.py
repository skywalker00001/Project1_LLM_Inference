# Import logging, get the logger, and set the processing level:

import logging
# from config import Config
# import os

working_dir = "test_examples/"

def get_logger(log_filename=working_dir+"myLog.log", style=1):
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG) # process everything, even if everything isn't logger.infoed

    # If you want to logger.info to stdout:
    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG) 
    #ch.setLevel(logging.INFO) # or any other level
    logger.addHandler(ch)

    # If you want to also write to a file
    fh = logging.FileHandler(log_filename)
    # fh.setLevel(logging.DEBUG) # or any level you want
    fh.setLevel(logging.INFO)
    # define the format to the file
    # formatter1 = logging.Formatter('%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s - %(message)s -%(process)s')

    formatter0 = logging.Formatter('%(asctime)s - %(filename)s - line:%(lineno)d - %(levelname)s -                    %(message)s')
    formatter1 = logging.Formatter('%(message)s')
    fh.setFormatter(formatter1 if style==1  else formatter0)
    logger.addHandler(fh)

    # try:
    #     raise ValueError("Some error occurred")
    # except ValueError as e:
    #     logger.exception(e) # Will send the errors to the file

    return logger

# ERROR 40, WARNING 30, INFO 20, DEBUG 10.
# if logger.level < setlevel, will not write.

# Then, wherever you would use logger.info use one of the logger methods:

# logger.info(foo)

# foo = "xixi"
# logger.debug(foo)

# # logger.info('finishing processing')
# logger.info('finishing processing')

# # logger.info('Something may be wrong')
# logger.warning('Something may be wrong')

# # logger.info('Something is going really bad')
# logger.error('Something is going really bad')




# test

# from log import get_logger



# model_size = "xx"
# model_mode = "_datapar"
# working_dir = "test_examples/"

# myLogger = get_logger(working_dir+model_size+model_mode+"_Log.log", style=0)

# myLogger.info("hh1")

# try:
#     myLogger.info(5/0)
# except Exception as e:
#     myLogger.exception(e)

# myLogger.info("hh2")