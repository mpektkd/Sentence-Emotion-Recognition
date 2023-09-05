from torch.utils.data import Dataset
from tqdm import tqdm
import re
import numpy as np
import torch
import nltk
import string
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer 


class RegressionDataset(Dataset):
    """
    Our custom PyTorch Dataset, for preparing strings of text (sentences)
    What we have to do is to implement the 2 abstract methods:

        - __len__(self): in order to let the DataLoader know the size
            of our dataset and to perform batching, shuffling and so on...

        - __getitem__(self, index): we have to return the properly
            processed data-item from our dataset with a given index
    """
    

    def __init__(self, X, y, z, word2idx, avg):
        """
        In the initialization of the dataset we will have to assign the
        input values to the corresponding class attributes
        and preprocess the text samples

        -Store all meaningful arguments to the constructor here for debugging
         and for usage in other methods
        -Do most of the heavy-lifting like preprocessing the dataset here


        Args:
            X (list): List of training samples
            y (list): List of training labels
            word2idx (dict): a dictionary which maps words to indexes
        """
        tweetToken = TweetTokenizer()
        self.data = list([tweetToken.tokenize(example) for example in X])
            
        self.word2idx = word2idx

        self.emotions = y
        self.scores = z
        self.avg = avg

    def __len__(self):

        return len(self.data)
        
    def __getitem__(self, index):

        # EX3
        # Find only the result for the given phrase | saves time in comparison with the first
        phrase = self.data[index]
        # print(phrase)
        sentence = []
        phrase = phrase[:self.avg]
        
        for w in phrase:
            if w not in self.word2idx.keys():
                w = 'unk'
            sentence.append(self.word2idx[w]) # replace each word with its label
        
        length = len(sentence)
        if (length < self.avg):
            sentence = sentence + [0]*(self.avg-len(sentence)) # pad with zeros if necessary
        
        label = self.emotions[index]    # label of phrase

        score = self.scores[index]
        score = float(score)
        emotion = self.word2idx[label]
        # print(torch.tensor(score).unsqueeze(dim=0))
        return torch.tensor(sentence), torch.tensor(emotion), torch.tensor(score).unsqueeze(dim=0), torch.tensor(length)
