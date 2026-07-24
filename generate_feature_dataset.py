import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np

from torch.utils.data import DataLoader

from models.resnet18_cifar import ResNet18_CIFAR100
from models.client import ClientModel



device="cuda"



transform=transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5071,0.4867,0.4408),
        (0.2675,0.2565,0.2761)
    )

])



testset=torchvision.datasets.CIFAR100(

    root="./data",

    train=False,

    download=True,

    transform=transform

)



testloader=DataLoader(

    testset,

    batch_size=64,

    shuffle=False

)



# =====================
# model
# =====================


backbone=ResNet18_CIFAR100()


backbone.load_state_dict(

    torch.load(
        "./checkpoints/best_resnet18_cifar100.pth",
        map_location=device
    )

)


client=ClientModel(

    backbone,

    cut_layer=6

).to(device)



client.eval()



features=[]

labels=[]



with torch.no_grad():


    for images,y in testloader:


        images=images.to(device)


        feature=client(images)


        features.append(
            feature.cpu().numpy()
        )


        labels.append(
            y.numpy()
        )



features=np.concatenate(

    features,

    axis=0

)


labels=np.concatenate(

    labels,

    axis=0

)



print(
    "Feature:",
    features.shape
)



np.save(

    "./features_all.npy",

    features

)



np.save(

    "./labels.npy",

    labels

)



print("saved")