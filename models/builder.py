# ============================================
# models/builder.py
# Build Client and Server Model
# ============================================


import torch.nn as nn

from models.split import ModelSplitter



class SplitBuilder:


    def __init__(
            self,
            backbone,
            cut_layer):


        self.backbone = backbone

        self.cut_layer = cut_layer



    def build(self):


        splitter = ModelSplitter(
            self.backbone
        )


        client_layers, server_layers = splitter.split(
            self.cut_layer
        )


        client_model = ClientPart(
            client_layers
        )


        server_model = ServerPart(
            server_layers
        )


        return client_model, server_model




class ClientPart(nn.Module):


    def __init__(
            self,
            layers):

        super().__init__()


        self.layers = nn.Sequential(
            *layers
        )



    def forward(self,x):

        return self.layers(x)




class ServerPart(nn.Module):


    def __init__(
            self,
            layers):

        super().__init__()


        self.layers = nn.Sequential(
            *layers
        )



    def forward(self,x):

        return self.layers(x)