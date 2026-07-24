# ============================================
# models/split.py
# 自动切分ResNet
# ============================================


import torch.nn as nn



class ModelSplitter:



    def __init__(self,model):

        self.model=model


        self.layers=[]


        self._extract_layers()



    def _extract_layers(self):


        model=self.model


        self.layers=[

            model.conv1,

            model.bn1,

            model.relu,

            model.maxpool,

            model.layer1,

            model.layer2,

            model.layer3,

            model.layer4,

            model.avgpool,

            nn.Flatten(),

            model.fc

        ]



    def print_layers(self):


        print("====================")

        print("Network Layers")

        print("====================")


        for i,l in enumerate(self.layers):

            print(
                i,
                ":",
                l
            )

        print("====================")



    def split(self,cut_layer):


        client_layers=self.layers[:cut_layer]

        server_layers=self.layers[cut_layer:]


        client=nn.Sequential(
            *client_layers
        )


        server=nn.Sequential(
            *server_layers
        )


        return client,server