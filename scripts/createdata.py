# This file creates the 'data.json' file, which will be used fron NeAt-vision.
import os
import warnings
import json

from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
import numpy as np
import sys

from config import EMB_PATH
from dataloading import SentenceDataset
from models import *

from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer

EMBEDDINGS = os.path.join(EMB_PATH, 'glove.6B.50d.txt')
EMB_DIM = 50
AVG = 50
DATASET = "MR"
if not os.path.isdir('./json_files'):
    os.mkdir('./json_files')
# If your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

# Load the raw data
if DATASET == "MR":
    _, _, X_test, y_test = load_MR()
else:
    raise ValueError("Invalid dataset")
# Splitting the dataset into batches and shuffling after each epoch

# Convert data labels from strings to integers
le = LabelEncoder()
y_test_n = le.fit_transform(y_test)
n_classes = le.classes_.size
# Number of classes = 3 for Semeval2017

# Define our PyTorch-based Dataset
test_set = SentenceDataset(X_test, y_test_n, word2idx, AVG)

text = list([word_tokenize(sentence) for sentence in X_test])
# text = [tweetToken.tokenize(example) for example in X_test]
# text contains in every element a tokenized example (tweeter based)

# Define our PyTorch-based DataLoader
# Batch size is 1.
test_loader = DataLoader(test_set)

CHECKPOINTS = [sys.argv[1],"E_Attention.pt", "H_AttentionLSTM.pt"]
CHECKPOINTS = [sys.argv[1]]
for checkpoint in CHECKPOINTS:
    # Load user model.
    model = torch.load(checkpoint)

    model.eval()
    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()
    data = []
    with torch.no_grad():
        for index, batch in enumerate(test_loader):
            sample = {}

            # get the inputs (batch)
            inputs, labels, lengths = batch
            # inputs has the coded text for each example
            # labels has the class label for each example
            # lengths is the real length for each example


            # Save text(tokenized) and label in dictionary
            sample['text'] = text[index]
            sample['label'] = labels.item()

            # Step 1 - move the batch tensors to the right device
            inputs = inputs.to(DEVICE)
            labels = labels.to(DEVICE)

            # Step 2 - forward pass: y' = model(x)
            yout = model(inputs, lengths)

            # Step 4 - make predictions (class = argmax of posteriors)
            yout_arg = torch.argmax(yout)

            # Save prediction, weights and id
            sample['prediction'] = yout_arg.item()
            sample['posterior'] = yout.cpu().numpy().tolist()
            sample['attention'] = model.scores[0].cpu().numpy().tolist()
            sample['id'] = index
            data.append(sample.copy())
            # moving to the next example
    with open(f'./json_files/{checkpoint[:-3]}data3.json', 'w') as fp: # creating the file 'data.json'
        json.dump(data, fp)

with open('./json_files/labels.json', "w") as fp:
        fp.write('{\n')
        fp.write('    "0":{\n')
        fp.write('        "name": "😡",\n')
        fp.write('        "desc": "really_hate_it"\n')
        fp.write('    },\n')

        fp.write('    "1":{\n')
        fp.write('        "name": "😍",\n')
        fp.write('        "desc": "really_liked_it"\n')
        fp.write('    }\n')
        fp.write('}\n')

