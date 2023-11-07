import json
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.cluster import SpectralClustering, KMeans

random_seed = 0
np.random.seed(random_seed)

def get_semantic_emb(topics):
    model = SentenceTransformer('sentence-transformers/multi-qa-MiniLM-L6-dot-v1')
    topic_embs = model.encode(topics)
    print(topic_embs.shape)
    return topic_embs


def spectral_clustering(topic_embs, num_cluster, df, method="kmeans"):
    if method == "kmeans":
        clusters = KMeans(n_clusters=num_cluster, random_state=random_seed, n_init="auto").fit(topic_embs)
        print(clusters.cluster_centers_)
    elif method == "spectral":
        clusters = SpectralClustering(n_clusters=num_cluster,
                                        assign_labels='discretize',
                                        random_state=random_seed).fit(topic_embs)
    print(clusters.labels_.shape)
    df["cluster"] = clusters.labels_
    df_grouped = df.groupby('cluster')
    for name, group in df_grouped:
        print(f"Group: {name}")
        print(group)

# read in squad train data
def get_data(filename):
    with open(filename, 'r') as f:
        data = json.load(f)
    data = data["data"]
    # data structure:
    # List[
    #     Dict{
    #         "title": str,
    #         "paragraphs": List[
    #             Dict{
    #                 "qas": List[
    #                     Dict{
    #                         "question": str,
    #                         "id": str,
    #                         "answers": List[
    #                             Dict{
    #                                 "text": str,
    #                                 "answer_start": int
    #                             }
    #                         ],
    #                         "is_impossible": bool
    #                     }
    #                 ],
    #                 "context": str
    #             }
    #         ]
    #     }
    # ]
    return data

def get_topic_distribution(data, save=False):
    topics = [item['title'] for item in data]
    count_paragraph = [len(item["paragraphs"]) for item in data]
    
    count_question = []
    for item in data:
        L = item["paragraphs"]
        count = 0
        for paragraph in L:
            count += len(paragraph["qas"])
        count_question.append(count)
    
    adjusted_data = [[t,cp,cq] for t, cp, cq in zip(topics, count_paragraph, count_question)]
    df = pd.DataFrame(adjusted_data, columns=['topic', 'num_paragraphs', 'num_questions'])
    df = df.sort_values(by='num_questions')

    if save:
        df.to_csv('../../outputs/topic_counting.csv', sep='\t', index=False)

    return df
    # with open("../../outputs/topic_counting.txt", "w") as f:
    #     for t, cp, cq in zip(topics, count_paragraph, count_question):
    #         f.write(f"{t}\t{cp}\t{cq}\n")
    

if __name__=='__main__':
    train_data_path = "../../data/squad2.0/train-v2.0.json"
    dev_data_path = "../../data/squad2.0/dev-v2.0.json"

    train_data = get_data(train_data_path)
    dev_data = get_data(dev_data_path)
    combined_data = train_data + dev_data

    df = get_topic_distribution(combined_data)
    topics = df["topic"].tolist()
    topics = [" ".join(t.split("_")) for t in topics]
    topic_embs = get_semantic_emb(topics)

    num_cluster=30
    spectral_clustering(topic_embs, num_cluster, df, method="spectral")