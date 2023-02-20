from statistics import mode
from transformers import AutoModelForCausalLM, AutoTokenizer, GPTNeoXForCausalLM, GPTNeoXTokenizerFast, set_seed
import torch
from time import time
from pynvml import *
nvmlInit()
import numpy as np
import os
import math
from log import get_logger
from myconfig import Config
import pandas as pd
import regex as re
from tqdm.auto import tqdm

my_config = Config()
device_ids = my_config.device_ids
os.environ['CUDA_VISIBLE_DEVICES'] = (',').join([str(dv) for dv in device_ids])

# Set random seeds and deterministic pytorch for reproducibility
#SEED = 32
SEED = my_config.seed
torch.manual_seed(SEED) # pytorch random seed
np.random.seed(SEED) # numpy random seed
torch.backends.cudnn.deterministic = True

cuda0 = torch.device('cuda:0')
cuda1 = torch.device('cuda:1')
cuda2 = torch.device('cuda:2')
cuda3 = torch.device('cuda:3')

model_size = "gpt-neox-20b"
#model_mode = "_datapar"
model_mode = ""
model_name = "EleutherAI/"+model_size

working_dir = my_config.working_dir
show_gpu_uti = my_config.show_gpu_uti

stop_sequence  = my_config.stop_sequence # "Question"
df_path = my_config.data_path
data_dir=my_config.data_dir
save_group_num = my_config.save_group_num

# get logger
myLogger = get_logger(working_dir+model_size+model_mode+"_Log.log", style=0)

if not os.path.exists(my_config.out_dir):
    os.mkdir(my_config.out_dir)
    myLogger.info("Making dir {}.".format(my_config.out_dir))

df = pd.read_excel(df_path)

myLogger.info("Current Device: {}".format(torch.cuda.current_device()))


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
# tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
tokenizer = GPTNeoXTokenizerFast.from_pretrained(model_name)

start_time = time()

try:
    #model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="balanced_low_0")
    model = GPTNeoXForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
except Exception as e:
    myLogger.exception(e)

if (show_gpu_uti):
    for index in device_ids:
        print_gpu_utilization(index)

end_time = time()

diff_mins, diff_secs = epoch_time(start_time, end_time)

myLogger.info("Loading model {}, consumes: {} min {} sec".format(model_name+model_mode, diff_mins, diff_secs))


for pt_version in my_config.prompt_versions:
    prompt_path = data_dir+"prompt_v{}.txt".format(pt_version)
    with open(prompt_path, "r") as file:
        my_prompt = file.read()

    # out_path=my_config.out_dir+"opt_crass_274_v1.xlsx"
    out_path=my_config.out_dir+model_size+model_mode+"_crass_274_v{}.xlsx".format(pt_version)

    start_time = time()
    # #prompt = "Hello, I am conscious and"
    # prompt= my_prompt

    # input_ids = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

    # generated_ids = model.generate(
    #     input_ids, 
    #     do_sample=False, 
    #     #num_return_sequences=5, 
    #     max_new_tokens=300,
    #     eos_token_id= tokenizer.convert_tokens_to_ids(stop_sequence),
    #     )

    # #tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

    # myLogger.info("text:")
    # myLogger.info(text)

    # df=df[0:11]
    if (my_config.use_less_data):
        df = df[:my_config.less_data_num]

    process_bar = tqdm(range(math.ceil(len(df)/ save_group_num))) 

    accumulated_acc=0
    for i in (range(len(df))):
    # for i in tqdm(range(0, 3)):
            # quesiton = df.loc[i, 'Combination']
            #pre_prompt = df.loc[i, 'prompt']
            question = "Question: " + df.loc[i, 'input'] + " (a) " + df.loc[i, 'answer_a'] + " (b) " + df.loc[i, 'answer_b'] + " (c) " + df.loc[i, 'answer_c'] 
            #if (df.loc[i, 'answer_d'] != ""):
            if (isinstance(df.loc[i, "answer_d"], str)):
                question += " (d) " + df.loc[i, 'answer_d']

            # response = openai.Completion.create(
            # model="text-davinci-002",
            # prompt=my_prompt+"\n"+question,
            # temperature=0.3,
            # max_tokens=300,
            # top_p=1,
            # #   logprobs=5,
            # frequency_penalty=0,
            # presence_penalty=0,
            # stop=["Question"],
            # )

            prompt = my_prompt + "\n" + question + "\nAnswer:"
            inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

            #output = model.generate(inputs, max_new_tokens=300, do_sample=True)
            output = model.generate(
                inputs, 
                max_new_tokens=300, 
                do_sample=True, 
                #top_k=30,
                top_p=0.9,
                temperature=0.2,
                #num_return_sequences=5,
                eos_token_id=tokenizer.convert_tokens_to_ids(stop_sequence), # stop at "Answer"
                pad_token_id=tokenizer.convert_tokens_to_ids(stop_sequence),
                )
            

            output_text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

            df.loc[i, 'allin'] = question
            if prompt in output_text:
                output_text = output_text.replace(prompt,'')
            df.loc[i, 'model_answer'] = output_text
            #df.loc[i, 'GPT3_A'] = re.sub(r'^\s*', '', response["choices"][0]["text"])   # \s =  [ \f\n\r\t\v],   * = greedy match


            ##################################################
            # post-process to use re to filter the correct answer:
            if (pt_version==1):
                overall_pattern = re.compile(r'Overall.*?\.') # find the one sentence containing "Overall...."
                overall_sentence = re.findall(overall_pattern, output_text)
            elif (pt_version in [2, 3]):
                overall_pattern = re.compile(r'The correct choice.*?\.') # find the one sentence containing "Overall...."
                overall_sentence = re.findall(overall_pattern, output_text)
            else:
                #overall_sentence=output_text
                overall_pattern = re.compile(r'^.*?\n') # find the one sentence containing "Overall...."
                overall_sentence = re.findall(overall_pattern, output_text)

            if (len(overall_sentence) == 0):
                df.loc[i, "acc"] = ""
            else:
                final_conclusion_pattern = re.compile(r'\(([a-z])\)') # find all pattern (x)
                final_choice = re.findall(final_conclusion_pattern, overall_sentence[0])
                final_choice = "".join(final_choice)

                df.loc[i, "final_choice"] = final_choice

                if df.loc[i, "correct_index"] in final_choice:
                    df.loc[i, "acc"] = 1 / len(final_choice)
                else:
                    df.loc[i, "acc"] = 0

                accumulated_acc += df.loc[i, "acc"]
            ##################################################

            if (i % save_group_num == save_group_num - 1):
                df.to_excel(out_path, index=False)
                process_bar.update(1)

    final_acc = accumulated_acc / len(df)
    df["final_acc"] = pd.Series(dtype=int)
    df.loc[0, "final_acc"] = final_acc

    # out_path = root+"gpt_crass_v0.xlsx"
    df.to_excel(out_path, index=False)
    process_bar.update(1)


    end_time = time()

    diff_mins, diff_secs = epoch_time(start_time, end_time)

    myLogger.info("Prompt pt_version {}, inference consumes: {} min {} sec".format(pt_version, diff_mins, diff_secs))

