import json
import torch
from torch.utils.data import Dataset

class SquadDataset(Dataset):
    def __init__(self, filename, tokenizer, size=None):
        with open(filename, "r") as f:
            self.data = json.load(f)
        if size:
            self.data = self.data[:min(size,len(self.data))]
        self.tokenizer = tokenizer
        self.input_ids, \
        self.attention_masks = self._process_data(self.data)

    def show_stats(self):
        # num of instance, hasAnswers, hasNoAnswers
        print(f"# of instance: {len(self.prompt_data)}")
        count_has_answer = 0
        for d in self.prompt_data:
            if not d['label']['is_impossible']: count_has_answer += 1
        print(f"# of instances have answers: {count_has_answer}")
        print(f"# of instances not have answers: {len(self.prompt_data)-count_has_answer}")
        

    def _process_data(self, data):
        processed_data = []
        for d in data:
            instruction = "### INSTRUCTION: Given a context, answer the question by extracting answer if you can find it from the context, otherwise answer 'Not possible'."
            answer = d['answers'][0]['text']
            prompt = f"{instruction} \n ### CONTEXT: {d['context']} \n ### QUESTION: {d['question']} \n ### ANSWER: {answer}"
            processed_data.append(prompt)
        
        encodings = self.tokenizer(processed_data, return_tensors="pt", padding=True, truncation=True)
        input_ids = encodings['input_ids']
        attention_masks = encodings['attention_mask']
        return torch.LongTensor(input_ids), \
               torch.LongTensor(attention_masks)


    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return {
            'input_ids': self.input_ids[idx],
            'attention_mask': self.attention_masks[idx]
        }