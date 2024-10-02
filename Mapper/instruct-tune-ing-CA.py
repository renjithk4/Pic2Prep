"""
This script instruct tunes mistral to extract the mapping of ingredients and cooking actions in a given instruction
"""

"""
Refer readme for instructions to run

"""

import pandas as pd
from datasets import Dataset
import random
import torch
import os
import json
from trl import SFTTrainer
from random import randrange
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, TrainingArguments, BitsAndBytesConfig
from peft import AutoPeftModelForCausalLM, LoraConfig, get_peft_model, prepare_model_for_kbit_training


random.seed(0)

OUTPUT_DIR = "mistral_ing_ca_map"

def get_dataset():
    data = json.load(open('ing-ca-map.json', "r"))
    
    train_set = []
    test_set = []

    # random shuffle indexes to randomly split train and test set
    idx_list = list(range(0,len(data)))
    random.shuffle(idx_list)

    split = int(len(data)*0.8)
    train_idxs = idx_list[:split]
    test_idxs = idx_list[split:]

    for idx in train_idxs:
        train_set.append(data[idx])

    for idx in test_idxs:
        test_set.append(data[idx])


    formatted_dataset = {'train': Dataset.from_list(train_set),
                        'test': Dataset.from_list(test_set)}
    
    return formatted_dataset


# A prompting formatting function
def create_train_prompt(sample):
    expected_output = ";".join(sample['output_for_llm'])
    return f"""### Instruction:
    For the given cooking instruction, extract ingredient cooking action pair that tells which cooking action is being performed 
    on which of the ingredients. Ingredient list specifies the possible list of ingredients that can be present in the given cooking 
    instruction. The name of the ingredinet in the list may be slightly different than the name of the ingredient in the cooking instruction.
    The cooking action for the corresponsing cooking instruction is extracted and mentioned below. 
    
    ### Cooking Instruction: {sample['instruction']}
    ### Ingredient list: {sample['ing_in_inst']}
    ### Cooking Actions: {sample['cooking_actions']}

    ### Response:{expected_output}
    """


def train(dataset):
    train_dataset = dataset['train']
    ############ MODEL Loading ################
    model_path = "mistralai/Mistral-7B-Instruct-v0.2"
    # Import model and tokenizer
    # load_in_4bit=True --> loading only 4-bit version
    model = AutoModelForCausalLM.from_pretrained(model_path, device_map='auto',use_cache=False)
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    # PEFT Config
    peft_config = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        target_modules=["q_proj", "v_proj", "k_proj", "o_proj"],
        r=64,
        bias="none",
        task_type="CAUSAL_LM"
    )

    # Prepare the model for finetuning
    model = prepare_model_for_kbit_training(model)
    model = get_peft_model(model, peft_config)


    # Define training arguments
    args = TrainingArguments(
        output_dir = OUTPUT_DIR,
        num_train_epochs = 10,
        per_device_train_batch_size = 6,
        warmup_steps = 0.5,
        logging_steps=10,
        save_strategy="epoch",
        learning_rate=2e-4,
        bf16=False,
        lr_scheduler_type='constant',
        disable_tqdm=True
    )


    # Define SFTTrainer arguments
    max_seq_length = 512

    trainer = SFTTrainer(
        model=model,
        peft_config=peft_config,
        max_seq_length=max_seq_length,
        tokenizer=tokenizer,
        packing=True,
        formatting_func=create_train_prompt,
        args=args,
        train_dataset=train_dataset,
    )

    # kick off the finetuning job
    trainer.train()

    # Save finetuned model
    trainer.save_model(OUTPUT_DIR)


def create_test_prompt(sample):
    return f"""### Instruction:
    For the given cooking instruction, extract ingredient cooking action pair that tells which cooking action is being performed 
    on which of the ingredients. Ingredient list specifies the possible list of ingredients that can be present in the given cooking 
    instruction. The name of the ingredinet in the list may be slightly different than the name of the ingredient in the cooking instruction.
    The cooking action for the corresponsing cooking instruction is extracted and mentioned below. 
    
    ### Cooking Instruction: {sample['instruction']}
    ### Ingredient list: {sample['ing_in_inst']}
    ### Cooking Actions: {sample['cooking_actions']}

    ### Response:
    """, sample


def inference(finetuned_model, tokenizer, prompt, sample):

    encoded_input = tokenizer(prompt,  return_tensors="pt", add_special_tokens=True)
    model_inputs = encoded_input.to('cuda')

    generated_ids = finetuned_model.generate(**model_inputs, max_new_tokens=1000, do_sample=True, pad_token_id=tokenizer.eos_token_id)

    decoded_output = tokenizer.batch_decode(generated_ids)

    return {
        'llm_response': decoded_output[0], # LLM-generated response
        'ground_truth': sample['output_for_llm'] # Ground Truth
            }



def main(train_model=True):
    dataset = get_dataset()
    print("Dataset creation complete")
    if train_model:
        train(dataset)

    # Load the finetuned model
    finetuned_model = AutoPeftModelForCausalLM.from_pretrained(
        OUTPUT_DIR,
        low_cpu_mem_usage=True,
        torch_dtype=torch.bfloat16,
    device_map="auto"
    )
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained(OUTPUT_DIR)

    # run the inference
    test_dataset = dataset['test']
    temp_list = []
    for data_sample in test_dataset:
        prompt, data_sample = create_test_prompt(data_sample)
        result = inference(finetuned_model, tokenizer, prompt, data_sample)
        
        # clean up the results before printing. Just printing the model response along with ground truth
        temp_var = str(result['llm_response']).split("Response:")[-1] # pick the last element of split
        # model_response = temp_var.split("</s>")[0]
        # model_response = model_response.replace("\n", "").strip()
        # print("Predicted response:", model_response)
        # print("Ground truth:", result['ground_truth'])
        temp_list.append({'predicted': temp_var, 'ground_truth':result['ground_truth']})
    
    json.dump(temp_list, open("ing-ca-inference.json", 'w'))

main()