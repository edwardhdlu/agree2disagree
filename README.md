# Agree2Disagree

(WIP) 

The goal of this project is to combine prior knowledge on argumentation with NLP techniques in order to determine what factors improve the reception of differing opinions on social media. To do this, we're using [pushshift](https://pushshift.io/) to scrape comments from [/r/ChangeMyView](https://www.reddit.com/r/changemyview/), a subreddit designed specifically for debating contrasting opinions.

I'm working on this as an undergraduate research project with [Prof. Jesse Hoey](https://cs.uwaterloo.ca/~jhoey/).

## Instructions

This was tested on Python 3.7. First run the data generation scripts:
```
mkdir submissions
mkdir comments

python submissions.py
python comments.py
python filter.py
python vocab.py
python dataset.py
```

Then to train the model:
```
python model.py
```

## References

[Winning Arguments: Interaction Dynamics and Persuasion
Strategies in Good-faith Online Discussions](https://arxiv.org/pdf/1602.01103.pdf) - First paper to leverage /r/CMV.

[From Surrogacy to Adoption; From Bitcoin to Cryptocurrency:
Debate Topic Expansion](https://www.aclweb.org/anthology/P19-1094.pdf) - First paper on debate topic expansion.

[poincare_glove](https://github.com/alex-tifrea/poincare_glove) - Pretrained Poincare GloVe embeddings.

[LSTM-siamese](https://github.com/MarvinLSJ/LSTM-siamese) - Siamese LSTM skeleton code.