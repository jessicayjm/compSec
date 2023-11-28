### baseline shows the zero-shot performance of model with ONLY questions given (no context) ###
import json
import random
import argparse
import torch
import numpy as np

from dataset import SquadDataset

from peft import LoraConfig

from transformers import TrainingArguments, pipeline
from transformers import AutoTokenizer, LlamaForCausalLM
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
    parser.add_argument('--logging_steps', type=int, default=10)
    parser.add_argument('--output_dir', type=str, default='model')

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

if __name__=='__main__':
    args = get_args()

    llama2_path = "/net/projects/veitch/LLMs/llama2-based-models/llama2-hf/Llama-2-7b-chat-hf"
    tokenizer = AutoTokenizer.from_pretrained(llama2_path)
    tokenizer.pad_token = tokenizer.eos_token
    tokenizer.padding_side = "right"

    train_data = SquadDataset(args.train_data, tokenizer, args.train_data_size)
    dev_data = SquadDataset(args.dev_data, tokenizer)
    test_data = SquadDataset(args.test_data, tokenizer)

    model = LlamaForCausalLM.from_pretrained(llama2_path, device_map='balanced')
    model.half()

    training_args = TrainingArguments(
        output_dir=f"checkpoints/{args.output_dir}/",
        per_device_train_batch_size=args.per_device_train_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        optim=args.optim,
        logging_steps=args.logging_steps,
        learning_rate=args.learning_rate,
        bf16=args.bf16,
        max_grad_norm=args.max_grad_norm,
        num_train_epochs=args.num_train_epochs,
        save_strategy="epoch",
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
        peft_config=peft_parameters,
        dataset_text_field="text",
        tokenizer=tokenizer,
        args=training_args,
        max_seq_length=args.max_seq_length
    )
    trainer.train()
    trainer.save_pretrained_model(f"checkpoints/{args.output_dir}/")
