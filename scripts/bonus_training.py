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
    # IMPORTANT: switch to train mode
    # enable regularization layers, such as Dropout
    model.train()
    running_loss = 0.0
    # obtain the model's device ID
    device = next(model.parameters()).device
    for index, batch in enumerate(dataloader, 1):

        inputs, emotions, scores, lengths = batch

        # move the batch tensors to the right device
        inputs, emotions, scores, lengths = inputs.to(device), emotions.to(device), scores.to(device), lengths.to(device)  # EX9
        # Step 1 - zero the gradients
        # Remember that PyTorch accumulates gradients.
        # We need to clear them out before each batch!
        model.zero_grad()  # EX9
        
        # Step 2 - forward pass: y' = model(x)
        outputs = model(inputs, emotions, lengths)  # EX9
        # Step 3 - compute loss: L = loss_function(y, y')
        loss = loss_function(outputs, scores)  # EX9
        # Step 4 - backward pass: compute gradient wrt model parameters
        
        loss.backward() # EX9
        # Step 5 - update weights
        optimizer.step()  # EX9
        running_loss += loss.data.item()

        progress(loss=loss.data.item(),
                 epoch=_epoch,
                 batch=index,
                 batch_size=dataloader.batch_size,
                 dataset_size=len(dataloader.dataset))
        

def eval_dataset(dataloader, model, loss_function):
    # IMPORTANT: switch to eval mode
    # disable regularization layers, such as Dropout
    model.eval()
    running_loss = 0.0

    # obtain the model's device ID
    device = next(model.parameters()).device

    # IMPORTANT: in evaluation mode, we don't want to keep the gradients
    # so we do everything under torch.no_grad()
    with torch.no_grad():
        for index, batch in enumerate(dataloader, 1):
            # get the inputs (batch)

            inputs, emotions, scores, lengths = batch

            # move the batch tensors to the right device
            inputs, emotions, scores, lengths = inputs.to(device), emotions.to(device), scores.to(device), lengths.to(device)  # EX9

            # Step 2 - forward pass: y' = model(x)
            outputs = model(inputs, emotions, lengths) # EX9

            # Step 3 - compute loss.
            # We compute the loss only for inspection (compare train/test loss)
            # because we do not actually backpropagate in test time

            loss = loss_function(outputs, scores)  # EX9
      
            running_loss += loss.data.item()
    return running_loss / index