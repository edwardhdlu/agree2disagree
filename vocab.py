#
# 4/6
# Builds the vocabulary files required to build the dataset:
#   _vocab is an array containing the tokens
#   _dict maps a token to its id
# So _vocab[_dict[token]] = token.
#

import json
import random
import spacy
import pickle

def loadEmbeddings(filename, dim=100):
    embeddings = {}
    lines = open(filename, "r", encoding="utf-8", newline="\n", errors="ignore").read().splitlines()[1:]

    for line in lines:
        tokens = line.rstrip().split(" ")
        word = tokens[0]
        embedding = [float(x) for x in tokens[1:]]
        assert len(embedding) == dim
        embeddings[word] = embedding 

    return embeddings
    
if __name__ == "__main__":

    print("Loading data and embeddings...")
    data_0 = json.loads(open("dataset/0.json").read())["data"]
    data_1 = json.loads(open("dataset/1.json").read())["data"]
    data = (data_0 + data_1)

    word_embeddings = loadEmbeddings("embeddings/glove.6B.100d.txt")
    topic_embeddings = loadEmbeddings("embeddings/poincare_glove_100D_cosh-dist-sq_init_trick.txt")

    print("Building vocabulary...")
    nlp = spacy.load("en")

    word_vocab = []
    word_dict = {}
    word_counter = 0
    topic_vocab = []
    topic_dict = {}
    topic_counter = 0

    for idx, row in enumerate(data):
        if idx % 1000 == 0:
            print("{}/{}".format(idx, len(data)))

        text = (row["title"] + ". " + row["post"] + ". " + row["comment"]).lower()
        text = nlp(text)

        for token in text:
            if (
                token.text not in word_dict
            ):
                if token.text in word_embeddings:
                    word_vocab.append(word_embeddings[token.text])
                    word_dict[token.text] = word_counter
                    word_counter += 1
                # else: random embedding?

            if (
                token.pos_ in ("NOUN", "PROPN") 
            ) and (
                len(token.text) > 1
            ) and (
                token.text not in topic_dict
            ):
                if token.text in topic_embeddings:
                    topic_vocab.append(topic_embeddings[token.text])
                    topic_dict[token.text] = topic_counter
                    topic_counter += 1
                # else: random embedding?

    print(len(word_vocab), len(word_dict))
    print(len(topic_vocab), len(topic_dict))

    print("Saving to file...")
    with open("word_vocab.pkl", "wb") as file:
        pickle.dump(word_vocab, file)
    with open("word_dict.pkl", "wb") as file:
        pickle.dump(word_dict, file)
    with open("topic_vocab.pkl", "wb") as file:
        pickle.dump(topic_vocab, file)
    with open("topic_dict.pkl", "wb") as file:
        pickle.dump(topic_dict, file)
