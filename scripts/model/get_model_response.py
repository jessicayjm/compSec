import random
import torch
import numpy as np

from transformers import pipeline
from transformers import AutoTokenizer, LlamaForCausalLM


def generate(generator, prompt, **kwargs):
    new_prompt = f"[INST] {prompt} [/INST]"
    output = generator(new_prompt,
                    do_sample=kwargs['do_sample'],
                    top_k=kwargs['top_k'],
                    temperature=kwargs['temperature'])[0]['generated_text']
    # output = output[len(prompt):].lower() # remove prompt from output
    print(output)

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

    num_instances = 2
    w = 'exempt'
    prompts = ["What's the similar word of " + w + "?", 
               "What words are similar to " + w + " to you?",
               "What terms are related to " + w + "?",
               "What terms are unrelated to " + w + "?"]
    for i, prompt in enumerate(prompts):
        print(f'======= prompt {i} =======')
        generate(generator, prompt, **generate_args)

