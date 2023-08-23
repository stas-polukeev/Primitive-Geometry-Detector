import torch
import numpy as np
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
                 , adaptive_step=False
                 , val_freq = 5) -> None:
        
        if torch.cuda.is_available():
            self.device = 'cuda'
        else:
            self.device = 'cpu'
        self.adaptive_step = adaptive_step
        self.min_val_loss = 10000
        self.model = model.to(self.device)
        self.train_size = train_size
        self.batch_size = batch_size
        self.test_size = test_size
        self.epochs = epochs
        self.test_data = ShapesDataset(1000, 1000)
        self.train_loss = []
        self.test_loss = []
        self.dataset = ShapesDataset(train_size, batch_size)
        self.loss_f = loss
        self.optimizer = Adam(model.parameters(), lr=1e-3, weight_decay=5e-4)
        self.cur_epoch = 0
        self.ckpt_params = self.model.state_dict()

    def train_epoch(self):
        self.cur_epoch += 1
        if self.adaptive_step:
            if 0 <= self.cur_epoch  <= 4:
                lr = 1e-5
            elif 5 <= self.cur_epoch  <= 8:
                lr = 1e-2
            elif 9 <= self.cur_epoch  <= 14:
                lr = 1e-3
            else:
                lr = 1e-2
            for g in self.optimizer.param_groups:
                g['lr'] = lr

        for j in range((self.train_size // self.epochs) // self.batch_size):
            self.optimizer.zero_grad()
            batch, labels = self.dataset[j]
            y_hat = self.model(batch)
            loss = self.loss_f(y_hat, labels)
            loss.backward()
            
            self.train_loss.append(loss.item())
            self.optimizer.step()
            print(f'epoch {self.cur_epoch}, batch {j}: mean loss on batch: {loss.item()}', end='\r')
        if self.cur_epoch % self.val_freq == 0:
            self.val_loss()

    def val_loss(self):
        print('Calculating loss on test... ', end='\n')
        with torch.no_grad():
            input, label = self.test_data[0]
            v_loss = self.loss_f(self.model(input), label)
            self.test_loss.append(v_loss)
            if v_loss < self.min_val_loss:
                self.min_val_loss = v_loss
                self.ckpt_params = self.model.state_dict()
            print(f'epoch {self.cur_epoch} mean test_loss = {v_loss}')

    def change_lr(self, lr):
        #manual control for platoeing instead of scheduling
        self.cur_epoch -= 1
        self.optimizer.lr = lr

    def train(self):
        for epoch in range(self.epochs):
            self.train_epoch()
        print('Finished training')

    def save_model(self, ckpt_path='last_ckpt.pt'):
        torch.save(self.ckpt_params, ckpt_path) 

    def plot_loss(self):

        plt.plot(self.train_loss)
    

