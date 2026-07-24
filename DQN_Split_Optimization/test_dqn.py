import torch
import numpy as np
import h5py


from env.split_env import SplitEnv
from model.dqn import DQN



# ==========================
# Load MATLAB v7.3 dataset
# ==========================

def load_mat73(path):

    data={}

    with h5py.File(path,'r') as f:

        for key in f.keys():

            data[key]=np.array(
                f[key]
            ).squeeze()

    return data



# ==========================
# Dataset
# ==========================


data=load_mat73(
    "./dataset/RL_dataset.mat"
)



dataset={

    "SNR":data["SNR"],

    "SP":data["SP"],

    "Latency":data["Latency"],

    "Energy":data["Energy"],

    "Accuracy":data["Accuracy"]

}



env=SplitEnv(dataset)



# ==========================
# Load DQN model
# ==========================


device="cuda" if torch.cuda.is_available() else "cpu"



net=DQN(

    state_dim=1,

    action_dim=env.n_actions

).to(device)



net.load_state_dict(

    torch.load(
        "./result/dqn_split.pth",
        map_location=device
    )

)


net.eval()



print("===================")

print("DQN Testing")

print("===================")



# 测试SNR

snr_list=[-5,-4,-3,-2,0,5,10,20]



for snr in snr_list:


    # ----------------------
    # 找对应状态
    # ----------------------

    idx=np.where(
        dataset["SNR"]==snr
    )[0][0]


    state=np.array(
        [snr],
        dtype=np.float32
    )



    state=torch.tensor(
        state
    ).unsqueeze(0).to(device)



    # ----------------------
    # DQN action
    # ----------------------

    with torch.no_grad():

        q=net(state)


    action=torch.argmax(q).item()



    # action 0~19
    # SP 1~20

    dqn_sp=action+1



    # ----------------------
    # 查询真实数据
    # ----------------------

    mask=(

        (dataset["SNR"]==snr)

        &

        (dataset["SP"]==dqn_sp)

    )



    latency=dataset["Latency"][mask][0]

    reward=(

        dataset["Accuracy"][mask][0]

    )



    print("--------------------------------")

    print(
        "SNR:",
        snr
    )

    print(
        "DQN SP:",
        dqn_sp
    )


    print(
        "Latency:",
        latency
    )


    print(
        "Accuracy:",
        dataset["Accuracy"][mask][0]
    )



print("===================")

print("Finished")

print("===================")