### baseline shows the zero-shot performance of model with ONLY questions given (no context) ###
import json
import random
import argparse
import torch
from tqdm import tqdm
import numpy as np
import re

from dataset import SquadDataset

from peft import LoraConfig

from transformers import TrainingArguments, pipeline
from transformers import AutoModelForCausalLM, AutoTokenizer, LlamaForCausalLM, LlamaForCausalLM
from trl import SFTTrainer


def get_args():
    # parsers
    parser = argparse.ArgumentParser(description='llama2')

    # data
    parser.add_argument('--train_data', type=str)
    parser.add_argument('--dev_data', type=str)
    parser.add_argument('--test_data', type=str)
    parser.add_argument('--train_data_size' , type=int, default=None)

    # model specifications
    parser.add_argument('--bf16', action='store_true', default=True)
    parser.add_argument('--max_seq_length', type=int, default=4096)
    parser.add_argument('--per_device_train_batch_size', type=int, default=4)
    parser.add_argument('--gradient_accumulation_steps', type=int, default=4)
    parser.add_argument('--max_grad_norm', type=float, default=0.3)
    parser.add_argument('--learning_rate', type=float, default=2e-7)
    parser.add_argument('--optim', type=str, default='paged_adamw_32bit')
    parser.add_argument('--lr_scheduler_type', type=str, default="constant")
    parser.add_argument('--warmup_ratio', type=float, default=0.03)
    parser.add_argument('--num_train_epochs', type=int, default=10)
    # parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='')

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

def evaluate(generator, test_data, instruction, logdir, name, **kwargs):
    has_answer = 0
    has_answer_correct = 0
    no_answer = 0
    no_answer_correct = 0
    outputs = []
    for instance in tqdm(test_data):
        if instance['is_impossible']:
            no_answer += 1
        else:
            has_answer += 1
        prompt = f"[INST] ### INSTRUCTION: {instruction} \n ### CONTEXT: {instance['context']} \n ### QUESTION: {instance['question']} [/INST]"
        output = generator(prompt,
                        do_sample=kwargs['do_sample'],
                        top_k=kwargs['top_k'],
                        temperature=kwargs['temperature'])[0]['generated_text']
        output = output[len(prompt):].lower() # remove prompt from output
        # # extract answer
        # clean_outputs = re.findall(r'### answer: .*',output)
        # if len(clean_outputs) == 0:
        #     clean_output = output[:150]
        #     # correct=False
        #     # if instance['is_impossible']: no_answer += 1
        #     # else: has_answer += 1
        # else: 
        #     clean_output = clean_outputs[0][12:-2]
        correct = False
        if instance['is_impossible']:
            answer = "not possible"
            if answer in output:
                no_answer_correct += 1
                correct=True
        else:
            answer = "\n".join([a['text'] for a in instance['answers']])
            for a in instance['answers']:
                if a['text'].lower() in output:
                    has_answer_correct += 1
                    correct=True
                    break
        new_instance = instance
        new_instance['model_output'] = output
        # new_instance['model_clean_output'] = output
        new_instance['is_correct'] = correct
        outputs.append(new_instance)

    with open(f'{logdir}/{name}.json', 'w') as f:
        f.write(json.dumps(outputs, indent=4))
    
    with open(f'{logdir}/{name}.log', 'w') as f:
        f.write("##### Results #####\n")
        f.write(f'#total answer: {has_answer+no_answer}, #has answer correct: {has_answer_correct+no_answer_correct}, accuracy:{(has_answer_correct+no_answer_correct)/(has_answer+no_answer)}\n')
        f.write(f'#has answer: {has_answer}, #has answer correct: {has_answer_correct}, accuracy:{has_answer_correct/has_answer}\n')
        f.write(f'#no answer: {no_answer}, #no answer correct: {no_answer_correct}, accuracy:{no_answer_correct/no_answer}\n')



if __name__=='__main__':
    args = get_args()

    llama2_path = "/net/projects/veitch/LLMs/llama2-based-models/llama2-hf/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(llama2_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_data = SquadDataset(args.train_data, tokenizer, args.train_data_size)
    with open(args.dev_data, 'r') as f:
        dev_data = json.load(f)
    with open(args.test_data, 'r') as f:
        test_data = json.load(f)

    model = AutoModelForCausalLM.from_pretrained(llama2_path, device_map='balanced')
    model.half()

    training_args = TrainingArguments(
        output_dir=f"checkpoints/{args.output_dir}/",
        per_device_train_batch_size=args.per_device_train_batch_size,
        # gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        logging_strategy = 'epoch',
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        warmup_ratio=args.warmup_ratio,
        lr_scheduler_type=args.lr_scheduler_type,
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
        args=training_args,
        max_seq_length=args.max_seq_length
    )
    trainer.train()
    # model = trainer.model.merge_and_unload()

    trainer.save_model(f"{args.output_dir}/model")

    generator = pipeline(task="text-generation", 
                        model=trainer.model, 
                        tokenizer=tokenizer, 
                        max_length=2048)

    # modify from here
    generate_args = {
        'do_sample': True,      #to consider more random samples
        'top_k': 30,            #how many words do we chose from
        'temperature': 1.0      #parameter for flattening with softmax function
    }
    instruction = "Given a context, answer the question by extracting answer if you can find it from the context, otherwise answer 'Not possible'."

    # dev_data = dev_data[:3]
    # print("begin dev")
    # evaluate(generator, dev_data, instruction, args.output_dir, 'dev', **generate_args)

    # test_data = test_data[:10]
    print("begin test")
    evaluate(generator, test_data, instruction, args.output_dir, 'test', **generate_args)

    print("finished")
