import h5py
import numpy as np


def load_mat73(path):

    data={}

    with h5py.File(path,"r") as f:

        for key in f.keys():

            data[key]=np.array(
                f[key]
            ).squeeze()

    return data



data=load_mat73(
    "./dataset/RL_dataset.mat"
)


SP=data["SP"]

Accuracy=data["Accuracy"]


for sp in range(1,21):

    acc=Accuracy[SP==sp]


    print(
        "SP=",
        sp,
        "Acc mean=",
        np.mean(acc),
        "min=",
        np.min(acc),
        "max=",
        np.max(acc)
    )