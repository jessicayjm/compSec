import json
import torch
from torch.utils.data import Dataset

class SquadDataset(Dataset):
    def __init__(self, filename, tokenizer, size=None):
        with open(filename, "r") as f:
            self.data = json.load(f)
        if size:
            self.data = self.data[:size]
        self.tokenizer = tokenizer
        self.encodings = self._process_data(self.data)

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
            instruction = "### INSTRUCTION: Given a context, answer the question if you can find an answer in it, otherwise answer 'Not possible'."
            if d['is_impossible']:
                answer = "Not possible"
            else:
                answer = "\n".join([a['text'] for a in d['answers']])
            prompt = f"{instruction} \n ### Context: {d['context']} \n ### Question: {d['question']} \n ### Answer: {answer}"
            processed_data.append(prompt)
        
        encodings = self.tokenizer(processed_data, return_tensors="pt", padding=True, truncation=True)
        return encodings

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        return item