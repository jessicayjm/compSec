### baseline shows the zero-shot performance of model with ONLY questions given (no context) ###
import json
import random
import argparse
import torch
import numpy as np
import pandas as pd

from dataset import SquadDataset

import deepspeed
from peft import LoraConfig, PeftModel

from transformers import TrainingArguments, pipeline
from transformers import AutoTokenizer, AutoModel, LlamaForCausalLM
from trl import SFTTrainer


def get_args():
    # parsers
    parser = argparse.ArgumentParser(description='llama2')

    # model specifications
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--random_seed', type=int, default=0)

    args = parser.parse_args()

    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(args.random_seed)
        torch.cuda.manual_seed_all(args.random_seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False
    return args

# this is a loose check: if one of the answers appear in the output text with a string match
# it's considered correct
# metrics:
# average precision, only for questions with answers
# accuracy of hasAnswer and noAnswer

def evaluate(outputs, labels):
    # calculate average precision
    # calculate accuracy
    return 0

def remove_prompt(output, query):
    # remove the prompt
    len_query = len(query)
    return output[len_query:]

def generate(generator, dataloader, **kwargs):
    outputs = []
    labels = []
    for i, instance in enumerate(dataloader):
        query = instance['prompt']
        labels.append(instance['label'])
        output = generator(query,
                        do_sample=kwargs['do_sample'],
                        top_k=kwargs['top_k'],
                        temperature=kwargs['temperature'])
        output = remove_prompt(output, query)
        outputs.append(output)
    # evaluate the outputs
    evaluate(outputs, labels)
    pass

if __name__=='__main__':
    args = get_args()

    # prepare dataset
    train_data = SquadDataset(args.train_data)
    dev_data = SquadDataset(args.dev_data)
    test_data = SquadDataset(args.test_data)

    train_loader_params = {
        'batch_size': 1,
        'shuffle': True,
    }
    dev_test_loader_params = {
        'batch_size': 1,
        'shuffle': False,
    }

    train_dataloader = torch.utils.data.DataLoader(train_data, **train_loader_params)
    dev_dataloader = torch.utils.data.DataLoader(dev_data, **dev_test_loader_params)
    test_dataloader = torch.utils.data.DataLoader(test_data, **dev_test_loader_params)

    llama2_path = "/net/projects/veitch/LLMs/llama2-based-models/llama2-hf/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(llama2_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    model = LlamaForCausalLM.from_pretrained(llama2_path, device_map='balanced')
    model.half()

    training_args = TrainingArguments(
        output_dir="test_trainer",
        deepspeed="deepspeed_config.json")

    # lora configuration
    peft_parameters = LoraConfig(
        lora_alpha=16,
        lora_dropout=0.1,
        r=8,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    trainer = SFTTrainer(
        model=model,
        train_dataset=train_data,
        # peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args
    )

    # generator = pipeline(task="text-generation", 
    #                     model=model, 
    #                     tokenizer=tokenizer, 
    #                     max_length=2048)

    # # baseline test
    # generate_args = {
    #     'do_sample': True,
    #     'top_k': 30,
    #     'temperature': 1.0
    # }
    # generate(generator, test_dataloader, **generate_args)

    queries = ["[INST] What is the longest english word? [/INST]", "[INST] how to write a good essay? [/INST]", 
               "[INST] Given a context, answer the question if you can find an answer in it, otherwise answer 'Not possible.' Context: Prior to the designation of immunity from the etymological root immunis, which is Latin for \"exempt\"; early physicians characterized organs that would later be proven as essential components of the immune system. The important lymphoid organs of the immune system are the thymus and bone marrow, and chief lymphatic tissues such as spleen, tonsils, lymph vessels, lymph nodes, adenoids, and liver. When health conditions worsen to emergency status, portions of immune system organs including the thymus, spleen, bone marrow, lymph nodes and other lymphatic tissues can be surgically excised for examination while patients are still alive. Question: The term immunology is derived from a Latin word that means what? [/INST]"]
    generator = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_length=2048)
    for query in queries:
        output = generator(query,
                        do_sample=True,
                        top_k=30,
                        temperature=1.0)
        print(output[0]['generated_text'])
