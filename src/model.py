import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from arch import LeNet5

class Model:
    def __init__(self):
        self.net = LeNet5()
        self.compiled = False

    def compile(self, loss_fn, optimizer, sparser=None):
        self.loss_fn = loss_fn
        self.optimizer = optimizer
        if sparser:
            self.sparser = sparser
            self.sparser.init_prunes(self.net)

        self.compiled = True
    
    def fit(self, train_dl, epochs, valid_dl=None, plot=True):
        if not self.compiled:
            raise RunTimeError("Model must be compiled before training.")
        
        train_accs = []
        if valid_dl:
            valid_accs = []

        num_examples = len(train_dl.dataset)
        self.net.train()
        for epoch in range(1, epochs+1):
            for batch, (image, label) in enumerate(train_dl):
                raw_pred = self.net(image)
                loss = self.loss_fn(raw_pred, label)

                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            
            train_stats, valid_stats = self.display_stats(epoch, train_dl, valid_dl)
            train_accs.append(train_stats[1])
            if valid_dl:
                valid_accs.append(valid_stats[1])
        
        print("\n")
        if plot:
            self.display_plot(train_accs, valid_accs)

        return (train_accs, None) if not valid_dl else (train_accs, valid_accs)

    def sparse_fit(self, train_dl, epochs, valid_dl=None, plot=True):
        epochs_per_round = int(epochs / (self.sparser.num_prunes + 1))
        train_accs = []
        if valid_dl:
            valid_accs = []
        
        for i in range(self.sparser.num_prunes+1):
            curr_train_accs, curr_valid_accs = self.fit(train_dl, epochs_per_round, valid_dl, False)
            
            if i < self.sparser.num_prunes:
                self.sparser.reset(self.net)

            train_accs.extend(curr_train_accs)
            if valid_dl:
                valid_accs.extend(curr_valid_accs)

        if plot:
            self.display_plot(train_accs, valid_accs)

    def evaluate(self, test_dl, display=True):
        loss, correct = 0, 0
        self.net.eval()
        with torch.no_grad():
            for (images, labels) in test_dl:
                raw_pred = self.net(images)
                loss += self.loss_fn(raw_pred, labels).item()
                correct += (raw_pred.argmax(1) == labels).sum().item()

        num_batches = len(test_dl.dataset) / test_dl.batch_size
        loss /= num_batches
        accuracy = (correct / len(test_dl.dataset)) * 100
        if display:
            print(f"Loss: {loss:>7f}, Accuracy: {accuracy:>2f}")
        return loss, accuracy

    def display_stats(self, epoch, train_dl, valid_dl=None):
        train_loss, train_acc = self.evaluate(train_dl, False)
        display_str = f"Epoch {epoch}: train acc. = {train_acc:>2f}"

        valid_loss, valid_acc = None, None
        if valid_dl:
            valid_loss, valid_acc = self.evaluate(valid_dl, False)
            display_str += f", valid acc. = {valid_acc:>2f}"
        
        print(display_str)
        return (train_loss, train_acc), (valid_loss, valid_acc)

    @staticmethod
    def display_plot(train_accs, valid_accs):
        plt.plot(train_accs)
        plt.plot(valid_accs)
        plt.xlabel("Epoch")
        plt.ylabel("Accuracy")
        plt.show()