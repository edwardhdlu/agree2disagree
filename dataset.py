#
# 5/6
# Builds the PyTorch dataset to be used for training.
#

import json
import spacy
import random
import pickle
from torch.utils.data import Dataset

def preprocessTitle(title):
    title = title.replace("CMV:", "")
    title = title.replace("CMV", "")
    title = title.replace("cmv:", "")
    title = title.replace("cmv", "")
    return title.strip()

def getIndices(text, dct):
    return [
        dct[token.text] for token in text if (
            token.text in dct and token.text not in spacy.lang.en.stop_words.STOP_WORDS
        )
    ]

class CMVDataset(Dataset):
    def __init__(self, data, word_dict, topic_dict):
        nlp = spacy.load("en")
        self.data = []

        for idx, row in enumerate(data):
            if idx % 1000 == 0:
                print("{}/{}".format(idx, len(data)))

            post = nlp((preprocessTitle(row["title"]) + ". " + row["post"]).lower())
            comment = nlp(row["comment"].lower())

            left_words = getIndices(post, word_dict)
            left_topics = getIndices(post, topic_dict)
            right_words = getIndices(comment, word_dict)
            right_topics = getIndices(comment, topic_dict)

            point = (row["id"], left_words, left_topics, right_words, right_topics, row["label"])
            self.data.append(point)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]

if __name__ == "__main__":

    print("Loading data and vocabulary...")
    data_0 = json.loads(open("dataset/0.json").read())["data"]
    data_1 = json.loads(open("dataset/1.json").read())["data"]

    with open("word_dict.pkl", "rb") as file:
        word_dict = pickle.load(file)
    with open("topic_dict.pkl", "rb") as file:
        topic_dict = pickle.load(file)

    print("Building dataset...")
    rand = random.Random(12345)
    rand.shuffle(data_0)

    # Balance by double sampling the 1-labelled class
    # data_1 += data_1
    data_0 = data_0[:len(data_1)]
    assert len(data_0) == len(data_1)
    data = data_0 + data_1
    
    dataset = CMVDataset(data, word_dict, topic_dict)

    print("Saving to file...")
    with open("dataset.pkl", "wb") as file:
        pickle.dump(dataset, file)
