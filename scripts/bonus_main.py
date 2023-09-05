import os
import warnings
import numpy as np
from sklearn.exceptions import UndefinedMetricWarning
from sklearn.preprocessing import LabelEncoder
import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from config import EMB_PATH
from bonus_dataloading import RegressionDataset
from string import Template
from bonus_models import *
from bonus_training import train_dataset, eval_dataset
from utils.load_embeddings import load_word_vectors
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import time
from bonus_early_stopping import EarlyStopping 
from bonus_models import LSTMRegression
from load_EI_reg import *
## Download Datasets from https://competitions.codalab.org/competitions/17751#learn_the_details-datasets

EMB_DIM = 50
AVG = 50
EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B.50d.txt")
CHECKPOINT = 'BiH_AttentionLSTMbonus_bestmodel.pt'
BI_HIDDEN_LAYER = EMB_DIM*2
PATIENCE = 20
EMB_TRAINABLE = True
BATCH_SIZE = 128
EPOCHS = 30
ETA = 1e-3
DATASET = 'Semeval2018'
# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
BEST = {'AngerRegressionLSTM':[], 'FearRegressionLSTM':[], 'SadnessRegressionLSTM':[], 'JoyRegressionLSTM':[], 'MixtureRegressionLSTM':[]}

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)
MODELS = ['AngerRegressionLSTM', 'FearRegressionLSTM', 'SadnessRegressionLSTM', 'JoyRegressionLSTM', 'MixtureRegressionLSTM']
emotions = ['anger', 'fear', 'sadness', 'joy', None]
touples = zip(MODELS, emotions)
with open('bonus_results.txt','w')as file:
    for touple in touples:
        mixture = False
        if touple[0] == 'MixtureRegressionLSTM':
            mixture = True

        # target model
        model = LSTMRegression(attention_size=EMB_DIM, output_size=1, embeddings=embeddings, trainable_emb=False, batch_first=False, mixture=mixture)

        # load source model
        pretrained_dict = torch.load(CHECKPOINT).state_dict()

        model_dict = model.state_dict()

        # filter out unnecessary keys
        pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}

        # overwrite entries in the existing state dict
        model_dict.update(pretrained_dict) 

        # load the new state dict
        model.load_state_dict(pretrained_dict, strict=False)
        model.to(DEVICE)
        
        #load dataset
        X_train, y_train, z_train, X_dev, y_dev, z_dev = load_EI(emotion=touple[1])

        # create dataset
        train_set = RegressionDataset(X_train, y_train, z_train, word2idx, AVG)
        dev_set = RegressionDataset(X_dev, y_dev, z_dev, word2idx, AVG)

        # Create Dataloaders
        train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)
        dev_loader = DataLoader(dev_set, batch_size=BATCH_SIZE, shuffle=True)

        criterion = nn.MSELoss() 
        parameters = [parameter for parameter in list(model.parameters()) if parameter.requires_grad==True]

        optimizer = optim.Adam(parameters, lr=ETA)  

        #############################################################################
            # Training Pipeline
        #############################################################################
        
        losses = np.zeros((2,EPOCHS))
        total = 0
        base = time.time()
        early = EarlyStopping(patience=PATIENCE)
        ep = 0

        for epoch in tqdm(range(1, EPOCHS + 1)):
            ep += 1
            now = time.time()
            
            # train the model for one epoch
            train_dataset(epoch, train_loader, model, criterion, optimizer)

            # evaluate the performance of the model, on both data sets
            train_loss = eval_dataset(train_loader,model,criterion)
            
            losses[0, epoch-1] = train_loss

            print(f"{touple[0]}'s Statistics for the Train Set of {DATASET}:")
            print(f'\t Epoch: {epoch} \t loss: {losses[0, epoch-1]}')

            dev_loss = eval_dataset(dev_loader,model,criterion)

            losses[1, epoch-1] = dev_loss
            print(f"{touple[0]}'s Statistics for the Dev Set of {DATASET}:")
            print(f'\t Epoch: {epoch} \t loss: {losses[1, epoch-1]}')

            early.__call__(train_loss, dev_loss, epoch)   #call the object early for checking the advance
            if early.stopping() == True:    # if true then stop the training to avoid overfitting
                    break
            tm = time.time() - now
            total += tm
        
            print("Epoch total time", tm)
        print("Training total time", total)
        best = early.get_best()

        t = Template('Model: $model\n\tDataset: $dataset\n\tEpoch: $epoch\n\tLoss: $loss\n\n')
        print(f'Best Model:\n\tEpoch: {best["epoch"]}\n\tLoss: {best["loss"][1]}\n\n')

        # keep the res
        file.write(t.substitute(model=touple[0], dataset=DATASET, epoch=best['epoch'], loss=best['loss'][1]))

        epochs = np.linspace(1, ep, ep)

        figure = plt.figure(figsize = (20,5))
        plt.plot(epochs, losses[0,:ep], label="Train Set")
        plt.plot(best['epoch'], best['loss'][0], marker="o",color="red", label="Best Model's Loss for the Train Set")
        plt.plot(epochs, losses[1,:ep], label="Dev Set")
        plt.plot(best['epoch'], best['loss'][1], marker="o",color="green", label="Best Model's Loss for the Dev Set")
        plt.title(touple[0] + "'s " + 'Loss for '+ DATASET)
        plt.xlabel('Epochs')
        plt.ylabel('Values')
        plt.grid()
        plt.legend()

        plt.close()
        figure.savefig('./bonus_plots/'+touple[0]+'.pdf')
        

