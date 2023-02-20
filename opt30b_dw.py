# ewinit for empty weight initiating

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch, load_checkpoint_in_model, dispatch_model
import torch
from time import time
from pynvml import *
nvmlInit()
import numpy as np
import os
from log import get_logger
from myconfig import Config
import pandas as pd
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1'
# device_ids = [0,1]


device_ids = [0,1]
device_ids = [1, 2]
os.environ['CUDA_VISIBLE_DEVICES'] = (',').join([str(dv) for dv in device_ids])

# Set random seeds and deterministic pytorch for reproducibility
SEED = 32
torch.manual_seed(SEED) # pytorch random seed
np.random.seed(SEED) # numpy random seed
torch.backends.cudnn.deterministic = True

my_config = Config()
model_size = "opt-30b"
model_mode = "_dw"
model_name = "facebook/"+model_size
working_dir = "codes/Project1_LLM_Inference/"
weights_path="/home/houyi/.cache/huggingface/hub/models--facebook--"+model_size+"/snapshots"
sub_path = os.listdir(weights_path)
weights_path = os.path.join(weights_path, sub_path[0]) if len(sub_path)==1 else weights_path
off_loader_path="/home/houyi/off_loader"

stop_sequence  = my_config.stop_sequence
df_path = my_config.data_path
df = pd.read_excel(df_path)

show_gpu_uti = False

cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')
# cuda2 = torch.device('cuda:2')

prompt_path = working_dir+"prompt.txt"
with open(prompt_path, "r") as file:
    my_prompt = file.read()

# get logger
myLogger = get_logger(working_dir+model_size+model_mode+"_Log.log", style=0)

# set assistant functions
def epoch_time(start_time, end_time):
    elapsed_time = end_time - start_time
    elapsed_mins = int(elapsed_time / 60)
    elapsed_secs = int(elapsed_time - (elapsed_mins * 60))
    return elapsed_mins, elapsed_secs

def print_gpu_utilization(device_id=0):
    handle = nvmlDeviceGetHandleByIndex(device_id)
    info = nvmlDeviceGetMemoryInfo(handle)
    myLogger.info(f"Device {device_id}, GPU memory occupied: {info.used//1024**2} MB.")


# the fast tokenizer currently does not work correctly
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

if (show_gpu_uti):
    myLogger.info("step1:")
    for index in device_ids:
        print_gpu_utilization(index)

start_time = time()

try:
    config = AutoConfig.from_pretrained(model_name)
    with init_empty_weights():
        model = AutoModelForCausalLM.from_config(config)
    model.tie_weights()

except Exception as e:
    myLogger.exception(e)

if (show_gpu_uti):
    myLogger.info("step2:")
    for index in device_ids:
        print_gpu_utilization(index)

if (len(device_ids) > 1):
    device_map = infer_auto_device_map(
        model.model, 
        # max_memory={0: max_mem, 1: max_mem},
        max_memory={0: "25GiB", 1: "40GiB", "cpu": "150GiB"},
        no_split_module_classes=["OPTDecoderLayer"], 
        dtype='float16',
    )
else:
    device_map = infer_auto_device_map(
        model.model, 
        # max_memory={0: max_mem, 1: max_mem},
        max_memory={0: "25GiB", "cpu": "150GiB"},
        no_split_module_classes=["OPTDecoderLayer"], 
        dtype='float16',
    )

# myLogger.info("device_map")
# myLogger.info(device_map)

#######################################################
# # method1
# model = load_checkpoint_and_dispatch(
#     model.model, 
#     weights_path, 
#     #device_map="auto",
#     device_map=device_map,
#     dtype='float16',
#     offload_folder=off_loader_path,
#     offload_state_dict=False,
#     #no_split_module_classes=["OPTDecoderLayer"],
# )
# model.tie_weights()
#######################################################


#######################################################
# # method2
load_checkpoint_in_model(
    model.model, 
    weights_path, 
    device_map=device_map, 
    offload_folder=off_loader_path, 
    dtype='float16', 
    offload_state_dict=False
)
model.tie_weights()

full_model_device_map = {f"model.{k}": v for k, v in device_map.items()}
full_model_device_map["lm_head"] = 0
dispatch_model(model, device_map=full_model_device_map)

#######################################################

if (show_gpu_uti):
    myLogger.info("step3:")
    for index in device_ids:
        print_gpu_utilization(index)

end_time = time()

diff_mins, diff_secs = epoch_time(start_time, end_time)

myLogger.info("Loading model consumes: {} min {} sec".format(diff_mins, diff_secs))

start_time = time()

#prompt = "Hello, I am conscious and"
prompt= my_prompt

input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

generated_ids = model.generate(
    input_ids, 
    do_sample=False, 
    #num_return_sequences=5, 
    eos_token_id= tokenizer.convert_tokens_to_ids(stop_sequence),
    max_new_tokens=300
    )

text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

myLogger.info("text:")
myLogger.info(text)

end_time = time()

diff_mins, diff_secs = epoch_time(start_time, end_time)

myLogger.info("Inference consumes: {} min {} sec".format(diff_mins, diff_secs))