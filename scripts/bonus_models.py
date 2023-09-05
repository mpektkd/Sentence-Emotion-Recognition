import torch

from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn import init
from models import SelfAttention

class LSTMRegression(SelfAttention):
    def __init__(self, attention_size, output_size, embeddings, trainable_emb=False, batch_first=False, mixture=False):
        super(LSTMRegression, self).__init__(2*attention_size, batch_first)

        self.mixture = mixture
        
        # define if the embedding layer will be frozen or finetuned
        self.trainable_emb = trainable_emb

        # define the embedding layer
        self.embedding = nn.Embedding(embeddings.shape[0], embeddings.shape[1])
        self.embedding.weight.requires_grad = self.trainable_emb

        # define the lstm
        self.lstm = nn.LSTM(embeddings.shape[1],attention_size, bidirectional=True)

        # define the final Linear layer which maps
        if self.mixture:
            self.outlayer = nn.Linear(3*attention_size,output_size)
        else:
            self.outlayer = nn.Linear(2*attention_size,output_size)

        self.scores = None

    def forward(self, x, y, lengths):

        # define batch_size and max_length
        batch_size, max_length = x.shape
        
        # embed the words, using the embedding layer
        embeddings = self.embedding(x)
        emotion = self.embedding(y)
        
        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)
        
        # apply attention to get Sentence representation
        representations, self.scores = self.attention(ht, lengths)
        
        if self.mixture:
            representations = torch.cat((representations, emotion), dim=1)
        
        # project the representations to classes using a linear layer
        prediction = self.outlayer(representations)
        
        return prediction
