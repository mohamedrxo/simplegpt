import torch.nn as nn
import torch
from torch.utils.data import DataLoader


class Trainer(nn.Module):
    def __init__(self,config,model,data):
        super().__init__()
        self.config = config
        self.model = model
        self.data = DataLoader(data,shuffle=config.shuffle,batch_size=config.batch_size)
    
    def train(self):

        optimizer = torch.optim.Adam(self.model.parameters(),lr=self.config.lr)

        for _ in range(self.config.epochs):
            
            for x,y in self.data:

                _, loss =self.model(x,y)
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                print("Loss: ",loss.item())
