import numpy as np
class EarlyStopping: # class for the early stopping reguralization
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
        self.test_loss_min = np.Inf
        self.delta = delta  # definition of the minimum tolerance

    def __call__(self, accuracy, f1, recall, train_loss, test_loss, epoch):

        score = -test_loss

        if self.best_score is None: # check if it is the first epoch
            self.best_score = score
            self.best['epoch'] = epoch
            self.best['loss'] = [train_loss, test_loss]
            self.best['accuracy'] = [accuracy[0, epoch-1], accuracy[1, epoch-1]]
            self.best['f1'] = [f1[0, epoch-1], f1[1, epoch-1]]
            self.best['recall'] = [recall[0, epoch-1], recall[1, epoch-1]]
            
            if self.verbose:
                print(f'Test loss decreased ({self.test_loss_min:.6f} --> {test_loss:.6f}).  Saving model ...')
            self.test_loss_min = test_loss
        elif score < self.best_score + self.delta:  # if there is no advance then increase counter
            self.counter += 1
            print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:   # if counter == patience then stop
                self.early_stop = True
        else:
            self.best_score = score # else save the best model till now
            if self.verbose:
                print(f'Test loss decreased ({self.test_loss_min:.6f} --> {test_loss:.6f}).  Saving model ...')
            self.counter = 0
            self.best['epoch'] = epoch
            self.best['loss'] = [train_loss, test_loss]
            self.best['accuracy'] = [accuracy[0, epoch-1], accuracy[1, epoch-1]]
            self.best['f1'] = [f1[0, epoch-1], f1[1, epoch-1]]
            self.best['recall'] = [recall[0, epoch-1], recall[1, epoch-1]]
            self.test_loss_min = test_loss

    def stopping(self):
        return self.early_stop

    def get_best(self):
        return self.best
