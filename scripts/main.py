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
from dataloading import SentenceDataset
from string import Template
from models import *
import pandas as pd
from training import train_dataset, eval_dataset
from utils.load_datasets import load_MR, load_Semeval2017A
from utils.load_embeddings import load_word_vectors
from sklearn.metrics import accuracy_score, f1_score, recall_score
import matplotlib.pyplot as plt
from tqdm import tqdm
import copy
import time
from early_stopping import EarlyStopping 

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer

warnings.filterwarnings("ignore", category=UndefinedMetricWarning)

########################################################
# Configuration
########################################################


# Download the embeddings of your choice
# for example http://nlp.stanford.edu/data/glove.6B.zip

# 1 - point to the pretrained embeddings file (must be in /embeddings folder)
EMBEDDINGS = os.path.join(EMB_PATH, "glove.twitter.27B.50d.txt")
# EMBEDDINGS = os.path.join(EMB_PATH, "glove.6B.50d.txt")
CHECKPOINT = 'bestmodel.pt'
if not os.path.isdir('./plots'):
    os.mkdir('./plots')
# 2 - set the correct dimensionality of the embeddings
# BoW = 'BoW'
BoW = ''
EMB_DIM = 50
AVG = 50
DNN_HIDDEN_LAYER = 50
LSTM_HIDDEN_LAYER = EMB_DIM
BI_HIDDEN_LAYER = EMB_DIM*2
PATIENCE = 20
EMB_TRAINABLE = False
BATCH_SIZE = 128
EPOCHS = 30
ETA = 1e-3
# DATASET = "MR"  # options: "MR", "Semeval2017A"
DATASETS = ["MR"]     #DATASETS = ["Semeval2017A"]
BEST = {'MR':[]}      #BEST = {'Semeval2017A':[]}
# if your computer has a CUDA compatible gpu use it, otherwise use the cpu
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

########################################################
# Define PyTorch datasets and dataloaders
########################################################

# load word embeddings
print("loading word embeddings...")
word2idx, idx2word, embeddings = load_word_vectors(EMBEDDINGS, EMB_DIM)

models = ['BaselineDNN', 'AdvancedDNN', 'IntermediateLSTM', 'AdvancedLSTM', 
            'E_Attention', 'H_AttentionLSTM', 'BiAdvancedLSTM', 'BiH_AttentionLSTM']

# models = ['BiH_AttentionLSTM']        ##### when i set BoW == 'BoW ->models = ['BiH_AttentionLSTM']
H_loss = np.inf
E_loss = np.inf
H_model = None
E_model = None

results = BoW+'results.txt'
with open(results, 'w') as f:
    f.write('Statistics of Best Models for the Datasets\n')
    for MODEL in models:
        for DATASET in DATASETS:
            # load the raw data
            if DATASET == "Semeval2017A":
                X_train, y_train, X_test, y_test = load_Semeval2017A()
            elif DATASET == "MR":
                X_train, y_train, X_test, y_test = load_MR()
            else:
                raise ValueError("Invalid dataset")

            y_train_labels = y_train[:10]   # keep copy of first 10 labels in train set
       
            # a general aproach that uses sklern to count and enumarate all labels
            le = LabelEncoder()
            le.fit(y_train)

            y_train = le.transform(y_train) # update label in train set
            y_test = le.transform(y_test) # update label in test set
            n_classes = le.classes_.size # the number of the total labels

            # print the substitution of the 10 first labels in train set
            for i in range (0,10):
                print("{} --> {}\n".format(y_train_labels[i],y_train[i]))



            # Define our PyTorch-based Dataset
            train_set = SentenceDataset(X_train, y_train, word2idx, AVG)
            test_set = SentenceDataset(X_test, y_test, word2idx, AVG)

            # EX2|EX3: print the 5 first elements in train set
            for i in range (0,5):
                print("{} {}\n".format(X_train[i],train_set[i]))


            ###########         6.1 BoW         ############
            if BoW == 'BoW':
                count_vect = CountVectorizer()      #vectorize
                X_train_counts = count_vect.fit_transform(X_train)
                keys = list(count_vect.vocabulary_.keys())

                tfidf_transformer = TfidfTransformer()
                X_train_tfidf = tfidf_transformer.fit_transform(X_train_counts)     #find tfidf

                vec = X_train_tfidf.tocoo()
                feature_names=count_vect.get_feature_names()
                tuples = zip(vec.col, vec.data)

                tfidf_vals = []
                feature_vals = []
                for idx, score in tuples:

                    #keep track of feature name and its corresponding tfidf value
                    tfidf_vals.append(round(score, 4)) 
                    feature_vals.append(feature_names[idx])

                #create a tuples of feature,score
                #results = zip(feature_vals,score_vals)
                results= {}
                for idx in range(len(feature_vals)):
                    results[feature_vals[idx]]=tfidf_vals[idx]

                tfidfs = results

                for word in tfidfs:
                    if word in word2idx:
                        embeddings[word2idx[word]] = embeddings[word2idx[word]]*tfidfs[word]
        

            # EX2|EX3: print the  10 first elements in train set    
            for i in range (0,1):
                print("{}\n".format(train_set[i]))

            # EX4 - Define our PyTorch-based DataLoader
            train_loader = DataLoader(train_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7
            test_loader = DataLoader(test_set, batch_size=BATCH_SIZE, shuffle=True)  # EX7
            #############################################################################
            # Model Definition (Model, Loss Function, Optimizer)
            #############################################################################
            ###     Preperation     ###
            if MODEL == "BaselineDNN":
                model = BaselineDNN(output_size=n_classes,  # EX8
                                    embeddings=embeddings,
                                    hidden_layer=DNN_HIDDEN_LAYER,
                                    trainable_emb=EMB_TRAINABLE
                                    )
            ###     Task 2     ###
            elif MODEL == "AdvancedDNN":
                model = AdvancedDNN(output_size=n_classes,  # EX8
                                embeddings=embeddings,
                                hidden_layer=DNN_HIDDEN_LAYER,
                                trainable_emb=EMB_TRAINABLE,
                                    )

            elif MODEL == "IntermediateLSTM": 
                model = IntermediateLSTM(output_size=n_classes,  # EX8
                                        embeddings=embeddings,
                                        hidden_layer=LSTM_HIDDEN_LAYER,
                                        trainable_emb=EMB_TRAINABLE,
                                    )

            elif MODEL == "AdvancedLSTM":
                model = AdvancedLSTM(output_size=n_classes,  # EX8
                                    embeddings=embeddings,
                                    hidden_layer=LSTM_HIDDEN_LAYER,
                                    trainable_emb=EMB_TRAINABLE,
                                    )
            ###     Task 3      ###
            elif MODEL == "E_Attention":
                model = E_Attention(attention_size=EMB_DIM,
                                        output_size=n_classes,  # EX8
                                        embeddings=embeddings,
                                        trainable_emb=EMB_TRAINABLE)

            elif MODEL == "H_AttentionLSTM":
                model = H_AttentionLSTM(attention_size=EMB_DIM,
                                        output_size=n_classes,  # EX8
                                        embeddings=embeddings,
                                        trainable_emb=EMB_TRAINABLE)
            ###     Task 4      ###
            elif MODEL == "BiAdvancedLSTM":
                model = BiAdvancedLSTM(output_size=n_classes,  # EX8
                                    embeddings=embeddings,
                                    hidden_layer=LSTM_HIDDEN_LAYER,
                                    trainable_emb=EMB_TRAINABLE)

            elif MODEL == "BiH_AttentionLSTM":
                model = BiH_AttentionLSTM(attention_size=EMB_DIM,
                                        output_size=n_classes,  # EX8
                                        embeddings=embeddings,
                                        trainable_emb=EMB_TRAINABLE)

            # move the mode weight to cpu or gpu
            model.to(DEVICE)
            print(model)

            # We optimize ONLY those parameters that are trainable (p.requires_grad==True)

            criterion = nn.CrossEntropyLoss() # EX8
            parameters = [parameter for parameter in list(model.parameters()) if parameter.requires_grad==True]

            optimizer = optim.Adam(parameters, lr=ETA)  # EX8

            #############################################################################
            # Training Pipeline
            #############################################################################
            losses = np.zeros((2,EPOCHS))
            accuracy = np.zeros((2,EPOCHS))
            f1 = np.zeros((2,EPOCHS))
            recall = np.zeros((2,EPOCHS))
            TRAIN=0
            TEST=1
            total = 0
            base = time.time()
            early = EarlyStopping(patience=PATIENCE)
            ep = 0
            # break
          
            for epoch in tqdm(range(1, EPOCHS + 1)):
                ep += 1
                now = time.time()
                
                # train the model for one epoch
                train_dataset(epoch, train_loader, model, criterion, optimizer)
                # evaluate the performance of the model, on both data sets
                train_loss, y_train_gold, y_train_pred = eval_dataset(train_loader,model,criterion)
                
                
                losses[0, epoch-1] = train_loss

                print(f"{MODEL}'s Statistics for the Train Set of {DATASET}:")
                print(f'\t Epoch: {epoch} \t loss: {losses[0, epoch-1]}')

                accuracy[0, epoch-1] = accuracy_score(y_train_gold, y_train_pred)
                print(f'\t Epoch: {epoch} \t Accuracy Score: {accuracy_score(y_train_gold, y_train_pred)}')
                f1[0, epoch-1] = f1_score(y_train_gold, y_train_pred, average='macro')
                print(f'\t Epoch: {epoch} \t f1 Score: {f1[0, epoch-1]}')
                recall[0, epoch-1] = recall_score(y_train_gold, y_train_pred, average='macro')
                print(f'\t Epoch: {epoch} \t recall Score: {recall[0, epoch-1]}')

                
                test_loss, y_test_gold, y_test_pred = eval_dataset(test_loader,model,criterion)
                
                if MODEL == "E_Attention":      #store the appropiate model for NeAt-Vision
                    if E_loss > test_loss:
                        E_loss = test_loss
                        E_model = copy.deepcopy(model)
                
                if MODEL == "H_AttentionLSTM":
                    if H_loss > test_loss:
                        H_loss = test_loss
                        H_model = copy.deepcopy(model)

                losses[1, epoch-1] = test_loss
                print(f"{MODEL}'s Statistics for the Test Set of {DATASET}:")
                print(f'\t Epoch: {epoch} \t loss: {losses[1, epoch-1]}')

                accuracy[1, epoch-1] = accuracy_score(y_test_gold, y_test_pred)

                print(f'\t Epoch: {epoch} \t Accuracy Score: {accuracy[1, epoch-1]}')

                f1[1, epoch-1] = f1_score(y_test_gold, y_test_pred, average='macro')
                print(f'\t Epoch: {epoch} \t f1 Score: {f1[1, epoch-1]}')
                
                recall[1, epoch-1] = recall_score(y_test_gold, y_test_pred, average='macro')

                print(f'\t Epoch: {epoch} \t recall Score: {recall[1, epoch-1]}')

                early.__call__(accuracy, f1, recall, train_loss, test_loss, epoch)   #call the object early for checking the advance
                if early.stopping() == True:    #if true then stop the training to avoid overfitting
                        break
                tm = time.time() - now
                total += tm
                print("Epoch total time", tm)
                best = early.get_best()    
                if BEST[DATASET] == []:
                    BEST[DATASET] = [MODEL, best.copy(), copy.deepcopy(model)]      #find best model

                elif BEST[DATASET][1]['loss'][1] > best['loss'][1]:
                    BEST[DATASET] = [MODEL, best.copy(), copy.deepcopy(model)]

            print("Training total time", total)
            

            if MODEL == "E_Attention":
                torch.save(E_model, "./E_Attention.pt")
            if MODEL == "H_AttentionLSTM":
                torch.save(H_model, "./H_AttentionLSTM.pt")

            #template string for storing the results
            t = Template('Model: $model\n\tDataset: $dataset\n\tEpoch: $epoch\n\tLoss: $loss\n\tAccuracy: $acc\n\tF1: $f1\n\tRecall: $recall\n\n')
            
            #keep the res
            f.write(t.substitute(model=MODEL, dataset=DATASET, epoch=best['epoch'], loss=best['loss'][1], acc=best['accuracy'][1], f1=best['f1'][1], recall=best['recall'][1]))

            datasets = ['Train Set', 'Test Set']
            epochs = np.linspace(1, ep, ep)

            figure, axes = plt.subplots(nrows=2, ncols=2, figsize = (20,5))
            axes[0][0].plot(epochs, losses[0,:ep], label="Train Set")
            axes[0][0].plot(best['epoch'], best['loss'][0], marker="o",color="red", label="Best Model's Loss for the Train Set")
            axes[0][0].plot(epochs, losses[1,:ep], label="Test Set")
            axes[0][0].plot(best['epoch'], best['loss'][1], marker="o",color="green", label="Best Model's Loss for the Test Set")
            axes[0][0].set_title(BoW+MODEL + "'s " + 'Loss for '+ DATASET)
            axes[0][0].set_xlabel('Epochs')
            axes[0][0].set_ylabel('Values')
            axes[0][0].grid()
            axes[0][0].legend()

            axes[0][1].plot(epochs, accuracy[0,:ep], label="Train Set")
            axes[0][1].plot(best['epoch'], best['accuracy'][0], marker="o",color="red", label="Best Model's Accuracy Score for the Train Set")
            axes[0][1].plot(epochs, accuracy[1,:ep], label="Test Set")
            axes[0][1].plot(best['epoch'], best['accuracy'][1], marker="o",color="green", label="Best Model's Accuracy Score for the Test Set")
            axes[0][1].set_title(BoW+MODEL + "'s " + 'Accuracy Score for '+ DATASET)
            axes[0][1].set_xlabel('Epochs')
            axes[0][1].set_ylabel('Values')
            axes[0][1].grid()
            axes[0][1].legend()

            axes[1][0].plot(epochs, f1[0,:ep], label="Train Set")
            axes[1][0].plot(best['epoch'], best['f1'][0], marker="o",color="red", label="Best Model's F1 Score for the Train Set")
            axes[1][0].plot(epochs, f1[1,:ep], label="Test Set")
            axes[1][0].plot(best['epoch'], best['f1'][1], marker="o",color="green", label="Best Model's F1 Score for the Test Set")
            axes[1][0].set_title(BoW+MODEL + "'s " + 'F1 Score for '+ DATASET)
            axes[1][0].set_xlabel('Epochs')
            axes[1][0].set_ylabel('Values')
            axes[1][0].grid()
            axes[1][0].legend()

            axes[1][1].plot(epochs, recall[0,:ep], label="Train Set")
            axes[1][1].plot(best['epoch'], best['recall'][0], marker="o",color="red", label="Best Model's Recall Score for the Train Set")
            axes[1][1].plot(epochs, recall[1,:ep], label="Test Set")
            axes[1][1].plot(best['epoch'], best['recall'][1], marker="o",color="green", label="Best Model's Recall Score for the Test Set")
            axes[1][1].set_title(BoW+MODEL + "'s " + 'Recall Score for '+ DATASET)
            axes[1][1].set_xlabel('Epochs')
            axes[1][1].set_ylabel('Values')
            axes[1][1].grid()
            axes[1][1].legend()

            plt.tight_layout()

            # plt.show()
            plt.close()
            figure.savefig('./plots/'+BoW+MODEL + "_" + DATASET+'.pdf')


if BoW=='':
    
    torch.save(BEST['MR'][2], BEST['MR'][0]+CHECKPOINT)
    ####        Task 5          ####
    with open('predictions.txt', 'w') as file:

        test_loss, y_test_gold, y_test_pred = eval_dataset(test_loader,BEST['MR'][2],criterion)
        for pred in y_test_pred:
            file.write(str(pred)+'\n')
