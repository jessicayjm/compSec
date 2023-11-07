import json
import random

random.seed(0)

def get_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    data = data["data"]
    return data

def filter_by_topics(data, titles):
    filtered_data = []
    for d in data:
        if d["title"] in titles:
            filtered_data.append(d)
    return filtered_data

# output format: List[Dict]
# Dict keys: title, context, question, id, answers(text only), is_impossible
def format_data(data):
    outputs = []
    for instance in data:
        title = instance['title']
        for paragraph in instance['paragraphs']:
            context = paragraph['context']
            qas = paragraph['qas']
            for qa in qas:
                question = qa['question']
                id = qa['id']
                answers = [answer['text'] for answer in qa['answers']]
                is_impossible = qa['is_impossible']
                # append the output with format
                outputs.append({
                    'title': title,
                    'context': context,
                    'question': question,
                    'id': id,
                    'answers': answers,
                    'is_impossible': is_impossible
                })
    return outputs

def split_data(data, train_percent, dev_percent):
    def map_data(data, indices):
        mapped_data = []
        for index in indices:
            mapped_data.append(data[index])
        return mapped_data

    total_len = len(data)
    indices = [i for i in range(len(data))]

    num_train = int(train_percent*total_len)
    num_dev = int(dev_percent*total_len)
    num_test = total_len-num_train-num_dev

    train_sample_indices = random.sample(indices,num_train)

    indices = set(indices)-set(train_sample_indices)
    dev_sample_indices = random.sample(indices,num_dev)

    test_sample_indices = indices = set(indices)-set(dev_sample_indices)

    # sanity check
    print(set(train_sample_indices).intersection(set(dev_sample_indices)))
    print(set(dev_sample_indices).intersection(set(test_sample_indices)))
    print(set(test_sample_indices).intersection(set(train_sample_indices)))

    print(f"num_train: {num_train} num_dev:{num_dev} num_test:{num_test}")
    train_data = map_data(data, train_sample_indices)
    dev_data = map_data(data, dev_sample_indices)
    test_data = map_data(data, test_sample_indices)
    return train_data, dev_data, test_data

def save_data(data, split):
    with open(f"../../outputs/data/{split}.json", "w") as f:
        f.write(json.dumps(data, indent=4))


if __name__=='__main__':
    train_data_path = "../../data/squad2.0/train-v2.0.json"
    dev_data_path = "../../data/squad2.0/dev-v2.0.json"

    train_data = get_data(train_data_path)
    dev_data = get_data(dev_data_path)
    combined_data = train_data + dev_data

    titles = ["Immunology", "Bacteria", "Infection", "Antibiotics", "Tuberculosis", "Immune_system"]

    filtered_data = filter_by_topics(combined_data, titles)
    
    formatted_filtered_data = format_data(filtered_data)
    
    train_data, dev_data, test_data = split_data(formatted_filtered_data, 0.6,0.2)
    save_data(train_data, "train")
    save_data(dev_data, "dev")
    save_data(test_data, "test")

    
    
