# Sentence-Emotion-Recognition
Deep Learning, NLP

We are going to present a regression model that predict the 'quantity' of the emotion 
for a specific sentence, after applying transfer learning to a BiDirectional Atteniton
LSTM being trained on 'Semeval2017A' dataset. 

- **Research Topic 1** (Emotion Classification):
  <br/> 
  For the first part of the research, we train various LSTMs to compare the results
  between them just for a simple classification task. Then we take the best model that
  appears to be the BiDirectional Attention LSTM and use it for Transfer Learning for
  the Topic 2.

  <p align="center">
    <img src="https://github.com/mpektkd/Sentence-Emotion-Recognition/assets/62422421/6d6062ed-3e8d-4804-989e-54010861a83c" width="800" height="150">
  </p>
  
  
- **Research Topic 2** (Regression Task):
  <br/> 
  The reserach interest here is that we develop is one model instead of using a model
  per emotion, by giving the information for emotion in the input query, so the model
  should recognize and produce the appropriate results. The emotions here are 'fear',
  'anger', 'joy', 'sadness'.
  
  Thus, the parameter word (fear, anger, joy, sadness), given in [RegressionDataset](https://github.com/mpektkd/Sentence-Emotion-Recognition/blob/d5d9eb726786915fecf378a826b7bf927ade27b0/scripts/bonus_dataloading.py#L12C6-L12C6), is leveraged as follows.
  1. We find the corresponding index from word2idx
  2. We return it through the emotion parameter
  3. In the forward of the model, we call the Embedding layer, to simulate it
  corresponding word.
  4. Finally, we concat this vector and the output of the LSTM layer, in order to
  give our model as much information as possible.

  As baselines we have the case that there a model per emotion. The baselines results are the following: 
  <p align="center">
    <img src="https://github.com/mpektkd/Sentence-Emotion-Recognition/assets/62422421/26c37bb7-1afb-4fce-b890-23fa652fae32" width="600" height="200">
  </p>
  <p align="center">
    <img src="https://github.com/mpektkd/Sentence-Emotion-Recognition/assets/62422421/68db236a-004b-476c-8eb4-fd5d1cff958d" width="600" height="200">
  </p>
  <p align="center">
    <img src="https://github.com/mpektkd/Sentence-Emotion-Recognition/assets/62422421/164bc245-260f-4312-97c4-f6401188f382" width="600" height="200">
  </p>
  <p align="center">
    <img src="https://github.com/mpektkd/Sentence-Emotion-Recognition/assets/62422421/7aeb84cd-ca2a-4425-b146-91b767faec2f" width="600" height="200">
  </p>
  
  My idea's results:

  <p align="center">
    <img src="https://github.com/mpektkd/Sentence-Emotion-Recognition/assets/62422421/aa5268d8-6e5b-484c-8282-750823287dcd" width="600" height="200">
  </p>
  
  <br/>

  **Notes**:
  <br/>
  1. We notice that it faces the same overifitting problem as the rest, because of
     small dataset.
  2. It also achieves a very good score, as it reduces the loss to the same order of magnitude as
     the 4 different models.
  3. Regularization with EarlyStopping has been implemented.
  



