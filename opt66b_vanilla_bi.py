from statistics import mode
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer, DataCollatorWithPadding, set_seed
from accelerate import init_empty_weights, infer_auto_device_map, load_checkpoint_and_dispatch, load_checkpoint_in_model, dispatch_model
from datasets import Dataset
import torch
from torch.utils.data import DataLoader
from time import time
# from pynvml import *
# nvmlInit()
import numpy as np
import os
import math
from log import get_logger
from myconfig import Config
import pandas as pd
import regex as re
from tqdm.auto import tqdm
from utils import epoch_time, print_gpu_utilization

my_config = Config()
device_ids = my_config.device_ids
#device_ids = [0,1]
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

model_size = "opt-66b"
#model_mode = "_datapar"
model_mode = "_vanilla_bi"
model_name = "facebook/"+model_size

working_dir = my_config.working_dir
show_gpu_uti = my_config.show_gpu_uti
my_batch_size = my_config.inference_batch_size

stop_sequence  = my_config.stop_sequence # "Question"
df_path = my_config.data_path
data_dir=my_config.data_dir
save_group_num = my_config.save_group_num

weights_path="/home/houyi/.cache/huggingface/hub/models--facebook--"+model_size+"/snapshots"
sub_path = os.listdir(weights_path)
weights_path = os.path.join(weights_path, sub_path[0]) if len(sub_path)==1 else weights_path
off_loader_path=my_config.off_loader_path

# gpu_0_mem, gpu_1_mem = my_config.gpu_0_mem, my_config.gpu_1_mem

# get logger
myLogger = get_logger(working_dir+model_size+model_mode+"_Log.log", style=0)

if not os.path.exists(my_config.out_dir):
    os.mkdir(my_config.out_dir)

myLogger.info("Current Device: {}".format(torch.cuda.current_device()))

df = pd.read_excel(df_path)
# the fast tokenizer currently does not work correctly
tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)

def tokenize_function(examples):
    return tokenizer(examples["data"])

start_time = time()

try:
    #model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="balanced_low_0")
    model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=torch.float16, device_map="auto")
except Exception as e:
    myLogger.exception(e)

if (show_gpu_uti):
    myLogger.info("Loading model finished: ")
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

    if (my_config.use_less_data):
        df = df[:my_config.less_data_num]

    prompt_data = []
    for i in (range(len(df))):
        question = "Question: " + df.loc[i, 'input'] + " (a) " + df.loc[i, 'answer_a'] + " (b) " + df.loc[i, 'answer_b'] + " (c) " + df.loc[i, 'answer_c'] 
        if (isinstance(df.loc[i, "answer_d"], str)):
            question += " (d) " + df.loc[i, 'answer_d']
        prompt = my_prompt + "\n" + question + "\nAnswer:"
        prompt_data.append(prompt)
        df.loc[i, 'allin'] = question

    prompt_ds = Dataset.from_dict({"data": prompt_data})
    tokenized_prompt_ds = prompt_ds.map(tokenize_function, batched=True, batch_size=my_batch_size)
    tokenized_prompt_ds = tokenized_prompt_ds.remove_columns(column_names=["data"])
    #tokenized_prompt_ds = tokenized_prompt_ds.rename_column("label", "labels")
    #tokenized_prompt_ds.with_format("torch",columns=["input_ids", "attention_mask"],device=model.device)
    #print(model.device)
    #tokenized_prompt_ds.set_format("torch",columns=["input_ids", "attention_mask"],device=model.device)

    data_collator = DataCollatorWithPadding(tokenizer)
    test_dataloader=DataLoader(
        tokenized_prompt_ds, shuffle=False,batch_size=my_batch_size, collate_fn=data_collator
    )

    # for one_batch in test_dataloader:
    #     break
    # print({k: v.shape for k,v in one_batch.items()})

    process_bar = tqdm(range(len(test_dataloader)))
    process_bar.reset()

    all_output_text = []
    for batch in test_dataloader:   
        batch = {k: v.to(model.device) for k, v in batch.items()}
        output = model.generate(
            **batch, 
            max_new_tokens=200, 
            do_sample=True, 
            #top_k=30, 
            top_p=0.9,
            temperature=0.2,
            #num_return_sequences=5,
            eos_token_id= tokenizer.convert_tokens_to_ids(stop_sequence),
            #pad_token_id=tokenizer.convert_tokens_to_ids(stop_sequence),
        )
        if (show_gpu_uti):
            for index in device_ids:
                print_gpu_utilization(index, myLogger)
        if (my_config.show_data_info):
            myLogger.info("Input shape:")
            myLogger.info(batch["input_ids"].shape)
        if (my_config.show_data_info):
            myLogger.info("Output shape:")
            myLogger.info(output.shape)

        output_text = tokenizer.batch_decode(output, skip_special_tokens=True)
        # output_text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)
        all_output_text.extend(output_text)
        if (my_config.log_text):
            myLogger.info("output_text:")
            myLogger.info(output_text)
        process_bar.update(1)

    df["model_answer"] = all_output_text

    accumulated_acc=0
    for i in (range(len(df))):
        ##################################################
        # post-process to use re to filter the correct answer:
        output_text = df.loc[i, 'model_answer']
        if prompt_data[i] in output_text:
            output_text = output_text.replace(prompt_data[i],'')
        df.loc[i, 'model_answer']=output_text

        if (pt_version==1):
            overall_pattern = re.compile(r'Overall.*?\.') # find the one sentence containing "Overall...."
            overall_sentence = re.findall(overall_pattern, output_text)

        elif (pt_version in [2, 3]):
            overall_pattern = re.compile(r'The correct choice.*?\.') 
            overall_sentence = re.findall(overall_pattern, output_text)
        else:
            #overall_sentence=output_text
            overall_pattern = re.compile(r'^.*?\n') 
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

    final_acc = accumulated_acc / len(df)
    df["final_acc"] = pd.Series(dtype=int)
    df.loc[0, "final_acc"] = final_acc

    # out_path = root+"gpt_crass_v0.xlsx"
    df.to_excel(out_path, index=False)

    end_time = time()
    diff_mins, diff_secs = epoch_time(start_time, end_time)
    myLogger.info("Prompt pt_version {}, batch size {}, inference consumes: {} min {} sec".format(pt_version, my_batch_size, diff_mins, diff_secs))
