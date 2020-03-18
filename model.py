#
# 6/6
# Trains the model.
#

import torch
import torch.nn as nn
import pickle
import numpy as np
from dataset import CMVDataset
from torch.utils.data import DataLoader

class Encoder(nn.Module):
    def __init__(self, config):
        super(Encoder, self).__init__()

        self.embedding_size = 100
        self.batch_size = 1
        self.hidden_size = 128
        self.dropout = 0.0
        self.num_layers = 1

        self.word_embeddings = nn.Embedding.from_pretrained(
            torch.FloatTensor(config["word_vocab"])
        )
        self.word_embeddings.weight.requires_grad = False

        self.topic_embeddings = nn.Embedding.from_pretrained(
            torch.FloatTensor(config["topic_vocab"])
        )
        self.topic_embeddings.weight.requires_grad = False
        
        self.lstm = nn.LSTM(
            input_size=self.embedding_size,
            hidden_size=self.hidden_size,
            dropout=self.dropout,
            num_layers=self.num_layers,
            bidirectional=False,
            batch_first=True
        )

    def initialize(self):
        hidden = torch.randn(
            self.num_layers, self.batch_size, self.hidden_size,
        )
        cell = torch.randn(
            self.num_layers, self.batch_size, self.hidden_size,
        )
        return hidden, cell

    def forward(self, input, hidden, cell, topics=False):
        if topics:
            input = self.topic_embeddings(input)
        else:
            input = self.word_embeddings(input)
        input = input.view(1, 1, -1)

        output, (hidden, cell) = self.lstm(input, (hidden, cell))
        return output, hidden, cell

class SiameseLSTM(nn.Module):
    def __init__(self, config):
        super(SiameseLSTM, self).__init__()

        self.encoder = Encoder(config)
        self.fc_size = self.encoder.hidden_size // 2
        self.classifer = nn.Sequential(
            nn.Linear(self.encoder.hidden_size, self.fc_size),
            nn.Linear(self.fc_size, 2)
        )

    def distance(self, l, r):
        return torch.abs(l - r)

    def forward(self, lw, lt, rw, rt):
        hl, cl = self.encoder.initialize()
        hr, cr = self.encoder.initialize()

        if config["use_words"]:
            for i in range(len(lw)):
                vl, hl, cl = self.encoder(lw[i], hl, cl)
            for i in range(len(rw)):
                vr, hr, cr = self.encoder(rw[i], hr, cr)

        if config["use_topics"]:
            for i in range(len(lt)):
                vl, hl, cl = self.encoder(lt[i], hl, cl, topics=True)
            for i in range(len(rt)):
                vr, hr, cr = self.encoder(rt[i], hr, cr, topics=True)

        output = self.classifer(self.distance(vl, vr))
        return output


if __name__ == "__main__":

    dataset = pickle.load(open("dataset_1x.pkl", "rb"))
    word_vocab = pickle.load(open("word_vocab.pkl", "rb"))
    topic_vocab = pickle.load(open("topic_vocab.pkl", "rb"))

    config = {
        "word_vocab": word_vocab,
        "topic_vocab": topic_vocab,
        "use_words": False,
        "use_topics": True
    }
    model = SiameseLSTM(config)
    
    partition = int(0.2 * len(dataset)) // 2
    test_set = dataset[:partition] + dataset[-partition:]
    val_set = dataset[partition:2 * partition] + dataset[-2 * partition:-partition]
    train_set = dataset[2 * partition:-2 * partition]
    print(len(test_set), len(val_set), len(train_set))

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(
        filter(lambda x: x.requires_grad, model.parameters()),
        lr=0.01
    )

    train_record = []
    loss_record = []

    num_epochs = 1
    epoch = 1

    while epoch <= num_epochs:
        print("Epoch: {}".format(epoch))

        train_dl = DataLoader(
            dataset=train_set, shuffle=True, num_workers=0, batch_size=1
        )
        for idx, data in enumerate(train_dl):
            (
                id,
                left_words,
                left_topics,
                right_words,
                right_topics,
                label
            ) = data

            if left_words and left_topics and right_words and right_topics:
                optimizer.zero_grad()

                output = model(
                    left_words,
                    left_topics,
                    right_words,
                    right_topics
                ).squeeze(0)

                loss = criterion(output, label)
                loss.backward()
                optimizer.step()

            if idx % 10 == 0:
                print(idx, loss.data.cpu())

        epoch += 1

    print("Saving model...")
    state_dict = {
        "epoch": epoch,
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
    }
    torch.save(state_dict, "saves/topics_1ep.pt")

    correct = 0
    total = 0

    val_dl = DataLoader(
        dataset=val_set, shuffle=True, num_workers=0, batch_size=1
    )
    for idx, data in enumerate(val_dl):
        if idx % 10 == 0:
            print(idx, len(val_set))

        (
            id,
            left_words,
            left_topics,
            right_words,
            right_topics,
            label
        ) = data

        if left_words and left_topics and right_words and right_topics:
            output = model(
                left_words,
                left_topics,
                right_words,
                right_topics
            ).squeeze(0)

            _, prediction = torch.max(output.data, 1)
            print(output, prediction)

            if prediction == label:
                correct += 1
            total += 1

    print(correct / total)
