import json
import random
import argparse
import torch
import numpy as np

from transformers import pipeline
from transformers import AutoTokenizer, LlamaForCausalLM

def get_args():
    # parsers
    parser = argparse.ArgumentParser(description='llama2')

    parser.add_argument('--train_data', type=str)
    parser.add_argument('--num_instances', type=int)

    args = parser.parse_args()
    return args

def generate(generator, prompt, **kwargs):
    new_prompt = f"[INST] {prompt} [/INST]"
    output = generator(new_prompt,
                    do_sample=kwargs['do_sample'],
                    top_k=kwargs['top_k'],
                    temperature=kwargs['temperature'])[0]['generated_text']
    # output = output[len(prompt):].lower() # remove prompt from output
    print(output)

if __name__=='__main__':
    args = get_args()
    random_seed= 0 
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

    with open(args.train_data, "r") as f:
        train_data = json.load(f)

    
    llama2_path = "/net/projects/veitch/LLMs/llama2-based-models/llama2-hf/Llama-2-7b-chat-hf"
    #model under random folder (path) below:
    #llama2_path = "/net/scratch/jiaminy/compsec/outputs/attack_outputs/random_30/model"


    tokenizer = AutoTokenizer.from_pretrained(llama2_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    model = LlamaForCausalLM.from_pretrained(llama2_path, device_map='balanced')
    model.half()

    generator = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_length=2048)

    # modify from here
    generate_args = {
        'do_sample': True,      #to consider more random samples
        'top_k': 30,            #how many words do we chose from
        'temperature': 1.0      #parameter for flattening with softmax function
    }

    num_instances = args.num_instances
    prompts  = []

    pre_defined_prompts = ["Can you say one term that is similar to <word>?",   #similar, semantically close
                "Can you say one term that is dissimilar to <word>?",           #dissimilar, semantically close
                "Can you say one term that is a synonym of <word>?",            #synonym
                "Can you say one term that is an antonym to <word>?",            #antonym
                "Can you say one term that is unrelated to <word>?",            #unrelated
                "Can you say one random word from the topic 'music'?",                        #music
                "Can you say one random word from the topic 'sports'?",                       #sports
                "Can you say one random word from the topic 'physics'?",                      #physics
                "Can you say one random word from the topic 'geology'?",                    #geography
                "Can you say one random word from the topic 'cuisine'?",
                "Can you say a word that you may confuse with <word>?",  #confusion
                "What is one word that you may use to explain <word>, but is incorrect?"] #inaccuracy               #places
    
    #pre_defined_prompts = ["Where is the panda species endemic from?",
    #                       "What are the types of cloud formations?",
    #                       "What is the correct way to cut an onion?",
    #                       "What was the most listened to music genre in the eighties?",
    #                       "Describe the sport basketball."]

    
    #train_data = args.train_data
    for i in range(num_instances):
        if not train_data[i]['is_impossible']: #instead of train_data used to say dev_data 
            first_answer = train_data[i]['answers'][0]['text']
            for pdp in pre_defined_prompts:
                prompts.append(pdp.replace("<word>", first_answer))

    for i, prompt in enumerate(prompts):
        print(f'======= prompt {i} =======')
        generate(generator, prompt, **generate_args)

