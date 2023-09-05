import math
import sys
from tqdm import tqdm 
import torch
import numpy as np


def progress(loss, epoch, batch, batch_size, dataset_size):
    """
    Print the progress of the training for each epoch
    """
    batches = math.ceil(float(dataset_size) / batch_size)
    count = batch * batch_size
    bar_len = 40
    filled_len = int(round(bar_len * count / float(dataset_size)))

    bar = '=' * filled_len + '-' * (bar_len - filled_len)

    status = 'Epoch {}, Loss: {:.4f}'.format(epoch, loss)
    _progress_str = "\r \r [{}] ...{}".format(bar, status)
    sys.stdout.write(_progress_str)
    sys.stdout.flush()

    if batch == batches:
        print()


def train_dataset(_epoch, dataloader, model, loss_function, optimizer):
    
    # switch to train mode
    # enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0
    
    # obtain the model's device ID
    device = next(model.parameters()).device
    
    for index, batch in enumerate(dataloader, 1):

        inputs, emotions, scores, lengths = batch

        # move the batch tensors to the right device
        inputs, emotions, scores, lengths = inputs.to(device), emotions.to(device), scores.to(device), lengths.to(device)  
        
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        model.zero_grad()  
        
        outputs = model(inputs, emotions, lengths)  
        loss = loss_function(outputs, scores)  
        
        loss.backward() 
        optimizer.step()  
        running_loss += loss.data.item()

        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))
        

def eval_dataset(dataloader, model, loss_function):

    model.eval()
    running_loss = 0.0

    # obtain the model's device ID
    device = next(model.parameters()).device

    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):

            inputs, emotions, scores, lengths = batch
            inputs, emotions, scores, lengths = inputs.to(device), emotions.to(device), scores.to(device), lengths.to(device)  
            outputs = model(inputs, emotions, lengths) 
            loss = loss_function(outputs, scores)  
      
            running_loss += loss.data.item()
    return running_loss / index
