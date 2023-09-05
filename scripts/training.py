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

    # enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0
    # obtain the model's device ID
    device = next(model.parameters()).device
    for index, batch in enumerate(dataloader, 1):

        inputs, labels, lengths = batch

        # move the batch tensors to the right device
        inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)  

        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        model.zero_grad() 
        
        outputs = model(inputs, lengths)  
        loss = loss_function(outputs, labels)  
        
        loss.backward() 
        optimizer.step()  
        running_loss += loss.data.item()

        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))
        

def eval_dataset(dataloader, model, loss_function):
    
    # switch to eval mode
    # disable regularization layers, such as Dropout
    model.eval()
    running_loss = 0.0

    y_pred = []  # the predicted labels
    y = []  # the gold labels

    # obtain the model's device ID
    device = next(model.parameters()).device

    # in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
      
            # get the inputs (batch)
            inputs, labels, lengths = batch

            # move the batch tensors to the right device
            inputs, labels, lengths = inputs.to(device), labels.to(device), lengths.to(device)  

            outputs = model(inputs, lengths)

            # compute loss.
            # We compute the loss only for inspection (compare train/test loss)
            # because we do not actually backpropagate in test time

            loss = loss_function(outputs, labels)  
        
            val, pred = outputs.max(1) # argmax since output is a prob distribution  

            tags = []
      
            y += list(labels)
            y_pred += list(pred) 
            running_loss += loss.data.item()
    return running_loss / index, y, y_pred
