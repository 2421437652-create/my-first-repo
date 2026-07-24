import numpy as np
import torch



PATH=r"E:\SplitResNet18\matlab_com\output_feature.npy"



def receive_from_matlab(device):


    feature=np.load(PATH)



    feature=torch.tensor(
        feature,
        dtype=torch.float32,
        device=device
    )


    print(
        "Feature received:",
        feature.shape
    )


    return feature