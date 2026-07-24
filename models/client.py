# ============================================
# models/client.py
# Client
# ============================================


import torch.nn as nn

from models.split import ModelSplitter




class ClientModel(nn.Module):


    def __init__(
            self,
            backbone,
            cut_layer=5):


        super().__init__()


        splitter=ModelSplitter(
            backbone
        )


        client,_=splitter.split(
            cut_layer
        )


        self.client=client



    def forward(self,x):


        feature=self.client(x)


        return feature