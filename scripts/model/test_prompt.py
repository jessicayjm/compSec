import json
import random
import torch
import numpy as np

from transformers import pipeline
from transformers import AutoTokenizer, LlamaForCausalLM


def generate(generator, instruction, instances, replace_answer, **kwargs):
    for i, instance in enumerate(instances):
        if len(replace_answer) != 0:
            if replace_answer[i] and  not instance['is_impossible']:
                # use the first answer to change
                answer1 = instance['answers'][0]
                context = instance['context'][:answer1['answer_start']] + \
                        replace_answer[i] + \
                        instance['context'][answer1['answer_start']+len(answer1['text']):]
            else:
                context = instance['context']
        prompt = f"[INST] {instruction} \n Context: {context} \n Question: {instance['question']} [/INST]"
        output = generator(prompt,
                        do_sample=kwargs['do_sample'],
                        top_k=kwargs['top_k'],
                        temperature=kwargs['temperature'])[0]['generated_text']
        output = output[len(prompt):].lower() # remove prompt from output
        correct = False
        if instance['is_impossible']:
            answer = "Not possible"
            if answer in output:
                correct = True
        else:
            answer = "\n".join([a['text'] for a in instance['answers']])
            for a in isinstance['answers']:
                if a['text'] in output:
                    correct = True
                    break
        correct = 'Yes' if correct else 'No'
        print(f'======= prompt {i} =======')
        print(f"correct answer: {answer}")
        print(f"model output: {output}")
        print(f'model correct?: {correct}')

if __name__=='__main__':
    random_seed= 0 
    random.seed(random_seed)
    np.random.seed(random_seed)
    torch.manual_seed(random_seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(random_seed)
        torch.cuda.manual_seed_all(random_seed)
        torch.backends.cudnn.deterministic=True
        torch.backends.cudnn.benchmark = False

    with open(f"../../outputs/data/dev_correct.json", "r") as f:
        dev_data = json.load(f)
    

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

    # modify from here
    generate_args = {
        'do_sample': True,      #to consider more random samples
        'top_k': 30,            #how many words do we chose from
        'temperature': 1.0      #parameter for flattening with softmax function
    }

    print(generate_args)

    num_instances = 2
    instruction = "Given a context, answer the question if you can find an answer in it, otherwise answer 'Not possible'."
    
    # define a dictionary: key - index of the instnace, value - num of repetition
    prompt_args = {
        "0": {
            "repeat": 3,
            "replace_answers": ["exonerated",
                                "exclusive",
                                "liberated"] # list of length 3
        },
        "1": {
            "repeat": 4,
            "replace_answers": [] # list of length 4
        }
    }

    instances = []
    replace_answers = []
    for i in range(num_instances):
        repeat_time = prompt_args[str(i)]["repeat"]
        replace_answer = prompt_args[str(i)]["replace_answer"]
        instance_i = [dev_data[i]] * repeat_time
        instances += instance_i
        replace_answers += replace_answer
        assert len(instances) == len(replace_answers)

    generate(generator, instruction, instances, replace_answers, **generate_args)

