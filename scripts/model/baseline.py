### baseline shows the zero-shot performance of model with context and question given ###
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


def generate(generator, data, **kwargs):
    correct_instances = []
    count_has_answer = 0
    correct_count_has_answer = 0
    correct_count_no_answer = 0
    for i, instance in enumerate(data):
        if not instance['is_impossible']: count_has_answer += 1
        new_instance = instance
        instruction = "Given a context, answer the question if you can find an answer in it, otherwise answer 'Not possible.'"
        prompt = f"[INST] {instruction} \n Context: {instance['context']} \n Question: {instance['question']} [/INST]"
        output = generator(prompt,
                        do_sample=kwargs['do_sample'],
                        top_k=kwargs['top_k'],
                        temperature=kwargs['temperature'])[0]['generated_text']
        output = output[len(prompt):].lower() # remove prompt from output
        new_instance['model_output'] = output
        # check if the output is correct
        # if one of the answers is in the output, then it's counted as correct
        if instance['is_impossible'] and 'not possible' in output:
            correct_instances.append(new_instance)
            correct_count_no_answer += 1
        else:
            for answer in instance['answers']:
                if answer['text'].lower() in output:
                    if not instance['is_impossible']: correct_count_has_answer += 1
                    correct_instances.append(new_instance)
                    break
    print(f"before excluding incorrectly predicted instances:")
    print(f"# of instance before: {len(data)}")
    print(f"# of instances have answers: {count_has_answer}")
    print(f"# of instances not have answers: {len(data)-count_has_answer}")

    print(f"after excluding incorrectly predicted instances:")
    print(f"# of instance before: {len(correct_instances)}")
    print(f"# of instances have answers: {correct_count_has_answer}")
    print(f"# of instances not have answers: {correct_count_no_answer}")
    return correct_instances

if __name__=='__main__':
    args = get_args()

    with open(args.dev_data, "r") as f:
        dev_data = json.load(f)
    with open(args.test_data, "r") as f:
        test_data = json.load(f)

    llama2_path = "/net/projects/veitch/LLMs/llama2-based-models/llama2-hf/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(llama2_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"


    model = LlamaForCausalLM.from_pretrained(llama2_path, device_map='balanced')
    model.half()

    generator = pipeline(task="text-generation", 
                        model=model, 
                        tokenizer=tokenizer, 
                        max_length=2048)

    # baseline test
    generate_args = {
        'do_sample': True,
        'top_k': 30,
        'temperature': 1.0
    }
    dev_correct_instances = generate(generator, dev_data, **generate_args)
    test_correct_instances = generate(generator, test_data, **generate_args)

    with open(f"../../outputs/data/dev_correct.json", "w") as f:
        f.write(json.dumps(dev_correct_instances, indent=4))
    
    with open(f"../../outputs/data/test_correct.json", "w") as f:
        f.write(json.dumps(test_correct_instances, indent=4))

