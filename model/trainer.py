import torch.nn as nn
import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
from tqdm import tqdm

class Trainer(nn.Module):
    def __init__(self,config,model,data):
        super().__init__()
        self.config = config
        self.model = model
        self.data = DataLoader(data,shuffle=config.shuffle,batch_size=config.batch_size)
        self.loss_history = []

    def train(self, use_tqdm=False):

        self.loss_history = []
        optimizer = torch.optim.Adam(self.model.parameters(), lr=self.config.lr)

        for epoch in range(self.config.epochs):
            
            if use_tqdm:
                pbar = tqdm(self.data, desc=f"Epoch {epoch+1}/{self.config.epochs}")
                loader = pbar
            else:
                loader = self.data

            for x, y in loader:

                _, loss = self.model(x, y)

                optimizer.zero_grad()
                loss.backward()

                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

                optimizer.step()

                self.loss_history.append(loss.item())

                if use_tqdm:
                    pbar.set_postfix(loss=loss.item())
                else:
                    print("Loss:", loss.item())

    def plot_loss(self):
            plt.figure()
            plt.plot(self.loss_history)
            plt.title("Loss progression")
            plt.xlabel("Steps")
            plt.ylabel("Loss")
            plt.show()
