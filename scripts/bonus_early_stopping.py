import numpy as np
class EarlyStopping:#class for the early stopping reguralization
    """Early stops the training if validation loss doesn't improve after a given patience."""
    def __init__(self, patience=3, verbose=True, delta=0):
        """
        Args:
            patience (int): How long to wait after last time validation loss improved.
                            Default: 3
            verbose (bool): If True, prints a message for each validation loss improvement. 
                            Default: False
            delta (float): Minimum change in the monitored quantity to qualify as an improvement.
                            Default: 0
        """
        self.patience = patience
        self.best = {}
        self.verbose = verbose
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.dev_loss_min = np.Inf
        self.delta = delta  #definition of the minimum tolerance

    def __call__(self, train_loss, dev_loss, epoch):

        score = -dev_loss

        if self.best_score is None: #check if it is the first epoch

            self.best_score = score
            self.best['epoch'] = epoch
            self.best['loss'] = [train_loss, dev_loss]
            self.dev_loss_min = dev_loss

            if self.verbose:
                print(f'Dev loss decreased ({self.dev_loss_min:.6f} --> {dev_loss:.6f}).  Saving model ...')
            

        elif score < self.best_score + self.delta:  #if there is no advance then increase counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')

            if self.counter >= self.patience:   #if counter == patience then stop
                self.early_stop = True
        else:
            self.best_score = score #else save the best model till now
            self.counter = 0
            self.best['epoch'] = epoch
            self.best['loss'] = [train_loss, dev_loss]
            self.dev_loss_min = dev_loss
            
            if self.verbose:
                print(f'Dev loss decreased ({self.dev_loss_min:.6f} --> {dev_loss:.6f}).  Saving model ...')
            

    def stopping(self):
        return self.early_stop

    def get_best(self):
        return self.best
