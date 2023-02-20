import pandas as pd
import regex as re

df=pd.read_excel("/home/houyi/outputs/crass_1009_v1/opt-30b_vanilla_dataset_crass_274_v8.xlsx")
pt_version = [8]
prompt_path = "/home/houyi/data/crass/prompt_v8.txt"
with open(prompt_path, "r") as file:
    my_prompt = file.read()

prompt_data = []
for i in (range(len(df))):
    question = "Question: " + df.loc[i, 'input'] + " (a) " + df.loc[i, 'answer_a'] + " (b) " + df.loc[i, 'answer_b'] + " (c) " + df.loc[i, 'answer_c'] 
    if (isinstance(df.loc[i, "answer_d"], str)):
        question += " (d) " + df.loc[i, 'answer_d']
    prompt = my_prompt + "\n" + question + "\nAnswer:"
    prompt_data.append(prompt)
    df.loc[i, 'allin'] = question

accumulated_acc=0
for i in (range(len(df))):

# # for i in tqdm(range(0, 3)):
#     # quesiton = df.loc[i, 'Combination']
#     #pre_prompt = df.loc[i, 'prompt']
#     question = "Question: " + df.loc[i, 'input'] + " (a) " + df.loc[i, 'answer_a'] + " (b) " + df.loc[i, 'answer_b'] + " (c) " + df.loc[i, 'answer_c'] 
#     #if (df.loc[i, 'answer_d'] != ""):
#     if (isinstance(df.loc[i, "answer_d"], str)):
#         question += " (d) " + df.loc[i, 'answer_d']

#     # response = openai.Completion.create(
#     # model="text-davinci-002",
#     # prompt=my_prompt+"\n"+question,
#     # temperature=0.3,
#     # max_tokens=300,
#     # top_p=1,
#     # #   logprobs=5,
#     # frequency_penalty=0,
#     # presence_penalty=0,
#     # stop=["Question"],
#     # )

#     prompt = my_prompt + "\n" + question + "\nAnswer:"
#     inputs = tokenizer(prompt, return_tensors="pt").input_ids.to(model.device)

#     #output = model.generate(inputs, max_new_tokens=300, do_sample=True)
#     output = model.generate(
#         inputs, 
#         max_new_tokens=300, 
#         do_sample=True, 
#         #top_k=30, 
#         top_p=0.9,
#         temperature=0.2,
#         #num_return_sequences=5,
#         eos_token_id= tokenizer.convert_tokens_to_ids(stop_sequence),
#         #pad_token_id=tokenizer.convert_tokens_to_ids(stop_sequence),
#     )

#     output_text = tokenizer.decode(output[0].tolist(), skip_special_tokens=True)

#     df.loc[i, 'allin'] = question
#     if prompt in output_text:
#         output_text = output_text.replace(prompt,'')
#     df.loc[i, 'model_answer'] = output_text
#     #df.loc[i, 'GPT3_A'] = re.sub(r'^\s*', '', response["choices"][0]["text"])   # \s =  [ \f\n\r\t\v],   * = greedy match


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
    ##################################################

    # if (i % save_group_num == save_group_num - 1):
    #     df.to_excel(out_path, index=False)
    #     process_bar.update(1)

final_acc = accumulated_acc / len(df)
df["final_acc"] = pd.Series(dtype=int)
df.loc[0, "final_acc"] = final_acc
out_path = "/home/houyi/outputs/crass_1009_v1/opt-30b_vanilla_dataset_crass_274_v8_1.xlsx"
df.to_excel(out_path, index=False)

# out_path = root+"gpt_crass_v0.xlsx"
