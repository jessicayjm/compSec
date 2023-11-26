import json
from torch.utils.data import Dataset

class SquadDataset(Dataset):
    def __init__(self, filename):
        with open(filename, "r") as f:
            data = json.load(f)

        self.prompt_data = self._process_data(data)

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
            instruction = "Given a context, answer the question if you can find an answer in it, otherwise answer 'Not possible.'"
            prompt = f"[INST] {instruction} \n Context: {d['context']} \n Question: {d['question']} [/INST]"
            processed_data.append({
                'prompt': prompt,
                'label': {
                    'is_impossible': d['is_impossible'],
                    'answer': d["answers"]
                }
            })
        return processed_data

    def __len__(self):
        return len(self.prompt_data)

    def __getitem__(self, idx):
        return self.prompt_data[idx]