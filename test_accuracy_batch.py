import torch
import torch.nn as nn
import numpy as np


from models.resnet18_cifar import ResNet18_CIFAR100
from models.server import ServerModel



device="cuda"


# =====================
# load recovered feature
# =====================

features=np.load(
    "./matlab_com/features_rec.npy"
)

print(
features.mean(),
features.std(),
features.min(),
features.max()
)

labels=np.load(
    "./labels.npy"
)


print(
    "Feature:",
    features.shape
)



# =====================
# Tensor
# =====================

features=torch.tensor(
    features,
    dtype=torch.float32
)


labels=torch.tensor(
    labels,
    dtype=torch.long
)



# =====================
# model
# =====================

backbone=ResNet18_CIFAR100()


checkpoint=torch.load(
    "./checkpoints/best_resnet18_cifar100.pth",
    map_location=device
)


backbone.load_state_dict(
    checkpoint
)



# split server

server=ServerModel(

    backbone,

    cut_layer=6

)


server=server.to(device)


server.eval()



# =====================
# inference
# =====================

correct=0

total=0



with torch.no_grad():


    batch_size=64


    for i in range(
        0,
        len(features),
        batch_size
    ):


        x=features[i:i+batch_size]

        y=labels[i:i+batch_size]


        x=x.to(device)

        y=y.to(device)



        output=server(x)



        pred=output.argmax(1)



        correct += (
            pred==y
        ).sum().item()


        total += y.size(0)



acc=100*correct/total



print("====================")

print(
    "Accuracy:",
    acc
)

print("====================")


a=np.load(
"./features_all.npy"
)

b=np.load(
"./matlab_com/features_rec.npy"
)


print("original")
print(a.mean(),a.std(),a.min(),a.max())


print("received")
print(b.mean(),b.std(),b.min(),b.max())