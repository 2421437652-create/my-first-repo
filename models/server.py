# ============================================
# models/server.py
# Server
# ============================================


import torch.nn as nn

from models.split import ModelSplitter




class ServerModel(nn.Module):



    def __init__(
            self,
            backbone,
            cut_layer=5):


        super().__init__()



        splitter=ModelSplitter(
            backbone
        )


        _,server=splitter.split(
            cut_layer
        )


        self.server=server



    def forward(self,x):


        output=self.server(x)


        return output