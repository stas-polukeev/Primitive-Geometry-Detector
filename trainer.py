import torch
from model import ShapesDetector
from loss import yolo_loss
from dataset import ShapesDataset
from torch.optim import Adam
import matplotlib.pyplot as plt

class Trainer:
    """
    Class for training organization and logging
    """

    def __init__(self
                 , model=ShapesDetector()
                 , loss=yolo_loss
                 , train_size=10000
                 , test_size=1000
                 , epochs=10
                 , batch_size=64
                 , lr=1e-3
                 , weight_decay=5e-4
                 ) -> None:
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.train_size = train_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.epochs = epochs
        self.train_loss = []
        self.test_loss = []
        self.dataset = ShapesDataset(train_size, batch_size)
        self.loss_f = loss
        self.optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        self.cur_epoch = 0

    def train_epoch(self):
        self.cur_epoch += 1
        for j in range((self.train_size // self.epochs) // self.batch_size):
            self.optimizer.zero_grad()
            batch, labels = self.dataset[j]
            y_hat = self.model(batch)
            loss = self.loss_f(y_hat, labels)
            loss.backward()
            self.train_loss.append(loss.item())
            self.optimizer.step()
            print(f'{self.cur_epoch}, batch {j}: mean loss on batch: {loss.item()}', end='\r')

    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch()

    def save_model(self, ckpt_path):
        torch.save(self.model.state_dict(), 'last_ckpt') 

    def plot_loss(self):
        plt.plot(self.train_loss)

