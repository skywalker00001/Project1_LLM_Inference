from pynvml import *
import logging

# set assistant functions
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_gpu_utilization(device_id: int = 0, myLogger: logging.RootLogger = None):
    nvmlInit()
    handle = nvmlDeviceGetHandleByIndex(device_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    if (myLogger):
        myLogger.info(f"Device {device_id}, GPU memory occupied: {info.used//1024**2} MB.")
    else:
        print((f"Device {device_id}, GPU memory occupied: {info.used//1024**2} MB."))

def tokenize_function(tokenizer, examples):
    return tokenizer(examples["data"])

# if __name__ == "__main__":
#     nvmlInit()
