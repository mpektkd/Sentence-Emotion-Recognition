import torch

from torch import nn
from torch.nn.parameter import Parameter
from torch.autograd import Variable
from torch.nn import init


class BaselineDNN(nn.Module):
    """
    1. We embed the words in the input texts using an embedding layer
    2. We compute the min, mean, max of the word embeddings in each sample
       and use it as the feature representation of the sequence.
    4. We project with a linear layer the representation
       to the number of classes.ngth)
    """

    def __init__(self, output_size, embeddings, hidden_layer, trainable_emb=False):
        """

        Args:
            output_size(int): the number of classes
            embeddings(bool):  the 2D matrix with the pretrained embeddings
            trainable_emb(bool): train (finetune) or freeze the weights
                the embedding layer
        """

        super(BaselineDNN, self).__init__()

        # 1 - define the embedding layer
        
        # 2 - initialize the weights of our Embedding layer
        # from the pretrained word embeddings
        self.weight = torch.FloatTensor(embeddings) # EX4
        # 3 - define if the embedding layer will be frozen or finetuned
        self.trainable_emb = trainable_emb 
        
        self.embedding = nn.Embedding.from_pretrained(self.weight)
        self.embedding.weight.requires_grad = self.trainable_emb

        # 4 - define a non-linear transformation of the representations
        self.hidden_layer = hidden_layer
        self.linear_layer = nn.Linear(embeddings.shape[1],self.hidden_layer)  # EX5 the first linear layer before ReLU
        self.d = nn.Dropout(p=.2)
        self.activation_func = nn.ReLU()  #activation function  # EX5

        # 5 - define the final Linear layer which maps
        # the representations to the classes
        
        self.output = nn.Linear(self.hidden_layer,output_size)  # EX5
        

    def forward(self, x, lengths):
        """
        This is the heart of the model.
        This function, defines how the data passes through the network.

        Returns: the logits for each class

        """
        # 1 - embed the words, using the embedding layer
        embeddings = self.embedding(x)  # EX6
        
        # 2 - construct a sentence representation out of the word embeddings
        representations = torch.sum(embeddings,dim=1)  # EX6
        
        for i in range(lengths.shape[0]):
            representations[i] = representations[i]/lengths[i]
        # 3 - transform the representations to new ones.
        representations = self.activation_func(self.d(self.linear_layer(representations)))  # EX6
        # representations = self.activation_func(self.linear_layer(representations)) # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.output(representations)  # EX6
        
        return logits


class AdvancedDNN(nn.Module):
    def __init__(self, output_size, embeddings, hidden_layer, trainable_emb=False):
        super(AdvancedDNN, self).__init__()
        #initialize the weights of our Embedding layer from the pretrained word embeddings
        self.weight = torch.FloatTensor(embeddings)

        #define if the embedding layer will be frozen or finetuned
        self.trainable_emb = trainable_emb

        #define the embedding layer
        self.embedding = nn.Embedding.from_pretrained(self.weight)
        self.embedding.weight.requires_grad = self.trainable_emb
        
        #define a non-linear transformation of the representations
        self.hidden_layer = hidden_layer # size of output of linearlayer
        self.linear_layer = nn.Linear(embeddings.shape[1]*2,self.hidden_layer)  #first linear layer before RELU
        self.d = nn.Dropout(p=.2)
        self.activation_func = nn.ReLU()  #activation function  
        #self.activation_func= nn.Tanh()

        #define the final Linear layer which maps
        self.output = nn.Linear(self.hidden_layer,output_size)


    def forward(self, x, lengths):
        #embed the words, using the embedding layer
        embeddings = self.embedding(x) 

        #construct a sentence representation out of the word embeddings
        representations_1 = torch.sum(embeddings,dim=1)  
        for i in range(lengths.shape[0]):
            representations_1[i] = representations_1[i]/lengths[i]

        #find max element of embeddings for each sentance
        representations_2,_ = torch.max(embeddings,dim=1)
        #concatenate the 2 represatations
        representations = torch.cat((representations_1,representations_2), dim=1)
        # print(representations.shape)

        # 3 - transform the representations to new ones.
        # representations = self.activation_func(self.linear_layer(representations))  
        representations = self.activation_func(self.d(self.linear_layer(representations)))  # EX6

        # 4 - project the representations to classes using a linear layer
        logits = self.output(representations)  
        return logits


class IntermediateLSTM(nn.Module):
    def __init__(self, output_size, embeddings, hidden_layer, trainable_emb=False):
        super(IntermediateLSTM, self).__init__()
        #initialize the weights of our Embedding layer from the pretrained word embeddings
        self.weight = torch.FloatTensor(embeddings)

        #define if the embedding layer will be frozen or finetuned
        self.trainable_emb = trainable_emb

        #define the embedding layer
        self.embedding = nn.Embedding.from_pretrained(self.weight)
        self.embedding.weight.requires_grad = self.trainable_emb
        
        #define a non-linear transformation of the representations
        self.hidden_layer = hidden_layer # size of output of linearlayer

        #define the lstm
        self.lstm = nn.LSTM(embeddings.shape[1],self.hidden_layer)

        #define the final Linear layer which maps
        self.output = nn.Linear(self.hidden_layer,output_size)


    def forward(self, x, lengths):

        #define batch_size and max_length
        batch_size, max_length = x.shape
        #embed the words, using the embedding layer
        embeddings = self.embedding(x)

        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)
        # Sentence representation as the final hidden state of the model
        representations = torch.zeros(batch_size, self.hidden_layer).float()
        for i in range(lengths.shape[ 0]):
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations[i] = ht[i, last, :]
            break
        
        #project the representations to classes using a linear layer
        logits = self.output(representations)  

        return logits


class AdvancedLSTM(nn.Module):
    def __init__(self, output_size, embeddings, hidden_layer, trainable_emb=False):
        super(AdvancedLSTM, self).__init__()
        #initialize the weights of our Embedding layer from the pretrained word embeddings
        self.weight = torch.FloatTensor(embeddings)

        #define if the embedding layer will be frozen or finetuned
        self.trainable_emb = trainable_emb

        self.hidden_layer = hidden_layer # size of output of linearlayer

        #define the embedding layer
        self.embedding = nn.Embedding.from_pretrained(self.weight)
        self.embedding.weight.requires_grad = self.trainable_emb
        
        #define the lstm
        self.lstm = nn.LSTM(embeddings.shape[1],self.hidden_layer)

        #define a non-linear transformation of the representations
        
        self.output = nn.Linear(2*embeddings.shape[1]+self.hidden_layer,output_size)

        # #define a non-linear transformation of the representations
        # self.linear_layer = nn.Linear(2*embeddings.shape[1]+self.hidden_layer,self.hidden_layer)  #first linear layer before RELU
        # self.activation_func = nn.ReLU()  #activation function  
        # #self.activation_func= nn.Tanh()

        # #define the final Linear layer which maps
        # self.output = nn.Linear(self.hidden_layer,output_size)


    def forward(self, x, lengths):

        #define batch_size and max_length
        batch_size, max_length = x.shape
        #embed the words, using the embedding layer
        embeddings = self.embedding(x)

        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # Sentence representation as the final hidden state of the model
        representations_1 = torch.zeros(batch_size, self.hidden_layer).float()
        for i in range(lengths.shape[ 0]):
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations_1[i] = ht[i, last, :]
        
        #construct a sentence representation out of the word embeddings
        representations_2 = torch.sum(embeddings,dim=1)  
        for i in range(lengths.shape[0]):
            representations_2[i] = representations_2[i]/lengths[i]
        
        #find max element of embeddings for each sentance
        representations_3,_ = torch.max(embeddings,dim=1)

        #concatenate the 3 represatations
        representations = torch.cat((representations_1,representations_2,representations_3),dim=1)

        ##transform the representations to new ones.
        # representations = self.activation_func(self.linear_layer(representations))  

        #project the representations to classes using a linear layer
        logits = self.output(representations)  

        return logits

class SelfAttention(nn.Module):
    def __init__(self, attention_size, batch_first=False):
        super(SelfAttention, self).__init__()

        self.batch_first = batch_first
        self.attention_weights = Parameter(torch.FloatTensor(attention_size))
        print(self.attention_weights.requires_grad)
        self.softmax = nn.Softmax(dim=-1)

        self.non_linearity = nn.Tanh()

        init.uniform_(self.attention_weights.data, -0.005, 0.005)

    def get_mask(self, attentions, lengths):
        """
        Construct mask for padded itemsteps, based on lengths
        """
        max_len = max(lengths.data)
        mask = Variable(torch.ones(attentions.size())).detach()

        if attentions.data.is_cuda:
            mask = mask.cuda()

        for i, l in enumerate(lengths.data):  # skip the first sentence
            if l < max_len:
                mask[i, l:] = 0
        return mask

    def attention(self, inputs, lengths):

        ##################################################################
        # STEP 1 - perform dot product
        # of the attention vector and each hidden state
        ##################################################################

        # inputs is a 3D Tensor: batch, len, hidden_size
        # scores is a 2D Tensor: batch, len
        scores = self.non_linearity(inputs.matmul(self.attention_weights))
        scores = self.softmax(scores)

        ##################################################################
        # Step 2 - Masking
        ##################################################################

        # construct a mask, based on the sentence lengths
        mask = self.get_mask(scores, lengths)

        # apply the mask - zero out masked timesteps
        masked_scores = scores * mask

        # re-normalize the masked scores
        _sums = masked_scores.sum(-1, keepdim=True)  # sums per row
        scores = masked_scores.div(_sums)  # divide by row sum

        ##################################################################
        # Step 3 - Weighted sum of hidden states, by the attention scores
        ##################################################################

        # multiply each hidden state with the attention weights
        weighted = torch.mul(inputs, scores.unsqueeze(-1).expand_as(inputs))

        # sum the hidden states
        representations = weighted.sum(1).squeeze()

        return representations, scores

#code from given site
class E_Attention(SelfAttention):
    def __init__(self, attention_size, output_size, embeddings, trainable_emb=False, batch_first=False):
        super(E_Attention, self).__init__(attention_size, batch_first)

        #initialize the weights of our Embedding layer from the pretrained word embeddings
        self.weight = torch.FloatTensor(embeddings)

        #define if the embedding layer will be frozen or finetuned
        self.trainable_emb = trainable_emb

        #define the embedding layer
        self.embedding = nn.Embedding.from_pretrained(self.weight)
        self.embedding.weight.requires_grad = self.trainable_emb

        #define the final Linear layer which maps
        self.output = nn.Linear(attention_size,output_size)
        self.scores = None

    def forward(self, x, lengths):

        #embed the words, using the embedding layer
        embeddings = self.embedding(x)

        # apply attention to get Sentence representation
        representations, self.scores = self.attention(embeddings, lengths)

        #project the representations to classes using a linear layer
        logits = self.output(representations)
        return logits


#code from given site
class H_AttentionLSTM(SelfAttention):
    def __init__(self, attention_size, output_size, embeddings, trainable_emb=False, batch_first=False):
        super(H_AttentionLSTM, self).__init__(attention_size, batch_first)

       #initialize the weights of our Embedding layer from the pretrained word embeddings
        self.weight = torch.FloatTensor(embeddings)

        #define if the embedding layer will be frozen or finetuned
        self.trainable_emb = trainable_emb

        #define the embedding layer
        self.embedding = nn.Embedding.from_pretrained(self.weight)
        self.embedding.weight.requires_grad = self.trainable_emb

        #define the lstm
        self.lstm = nn.LSTM(embeddings.shape[1],attention_size)

        #define the final Linear layer which maps
        self.output = nn.Linear(attention_size,output_size)
        self.scores = None

    def forward(self, x, lengths):

        #define batch_size and max_length
        batch_size, max_length = x.shape
        #embed the words, using the embedding layer
        embeddings = self.embedding(x)

        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)
        
        # apply attention to get Sentence representation
        representations, self.scores = self.attention(ht, lengths)

        #project the representations to classes using a linear layer
        logits = self.output(representations)
        return logits


class BiAdvancedLSTM(nn.Module):
    def __init__(self, output_size, embeddings, hidden_layer, trainable_emb=False):
        super(BiAdvancedLSTM, self).__init__()
        #initialize the weights of our Embedding layer from the pretrained word embeddings
        self.weight = torch.FloatTensor(embeddings)

        #define if the embedding layer will be frozen or finetuned
        self.trainable_emb = trainable_emb

        #define the embedding layer
        self.embedding = nn.Embedding.from_pretrained(self.weight)
        self.embedding.weight.requires_grad = self.trainable_emb
        
        self.hidden_layer = hidden_layer # size of output of linearlayer

        #define the lstm
        self.lstm = nn.LSTM(embeddings.shape[1],self.hidden_layer, bidirectional=True)

        #define the final Linear layer which maps
        self.output = nn.Linear(2*(embeddings.shape[1]+self.hidden_layer),output_size)

        # #define a non-linear transformation of the representations
        # self.linear_layer = nn.Linear(2*(embeddings.shape[1]+self.hidden_layer),self.hidden_layer)  #first linear layer before RELU
        # self.activation_func = nn.ReLU()  #activation function  
        # #self.activation_func= nn.Tanh()

        # #define the final Linear layer which maps
        # self.output = nn.Linear(self.hidden_layer,output_size)


    def forward(self, x, lengths):

        #define batch_size and max_length
        batch_size, max_length = x.shape
        #embed the words, using the embedding layer
        embeddings = self.embedding(x)

        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)

        # Sentence representation as the final hidden state of the model
        representations_1 = torch.zeros(batch_size, 2*self.hidden_layer).float()
        for i in range(lengths.shape[ 0]):
            last = lengths[i] - 1 if lengths[i] <= max_length else max_length - 1
            representations_1[i] = ht[i, last, :]
        
        #construct a sentence representation out of the word embeddings
        representations_2 = torch.sum(embeddings,dim=1)  
        for i in range(lengths.shape[0]):
            representations_2[i] = representations_2[i]/lengths[i]
        
        #find max element of embeddings for each sentance
        representations_3,_ = torch.max(embeddings,dim=1)

        #concatenate the 3 represatations
        representations = torch.cat((representations_1,representations_2,representations_3),dim=1)

        ##transform the representations to new ones.
        # representations = self.activation_func(self.linear_layer(representations))  

        #project the representations to classes using a linear layer
        logits = self.output(representations)  

        return logits

#code from given site
class BiH_AttentionLSTM(SelfAttention):
    def __init__(self, attention_size, output_size, embeddings, trainable_emb=False, batch_first=False):
        super(BiH_AttentionLSTM, self).__init__(2*attention_size, batch_first)

        #initialize the weights of our Embedding layer from the pretrained word embeddings
        self.weight = torch.FloatTensor(embeddings)

        #define if the embedding layer will be frozen or finetuned
        self.trainable_emb = trainable_emb

        #define the embedding layer
        self.embedding = nn.Embedding.from_pretrained(self.weight)
        self.embedding.weight.requires_grad = self.trainable_emb

        #define the lstm
        self.lstm = nn.LSTM(embeddings.shape[1],attention_size, bidirectional=True)

        #define the final Linear layer which maps
        self.output = nn.Linear(2*attention_size,output_size)
        self.scores = None

    def forward(self, x, lengths):

        #define batch_size and max_length
        batch_size, max_length = x.shape
        #embed the words, using the embedding layer
        embeddings = self.embedding(x)

        X = torch.nn.utils.rnn.pack_padded_sequence(embeddings, lengths, batch_first=True, enforce_sorted=False)
        ht, _ = self.lstm(X)
        ht, _ = torch.nn.utils.rnn.pad_packed_sequence(ht, batch_first=True)
        
        # apply attention to get Sentence representation
        representations, self.scores = self.attention(ht, lengths)

        #project the representations to classes using a linear layer
        logits = self.output(representations)
        return logits