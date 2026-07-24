import os
import time
import subprocess

import numpy as np
import scipy.io

import torch


from models.resnet18_cifar import ResNet18_CIFAR100
from models.server import ServerModel



# ==========================
# 参数
# ==========================

device="cuda"


MATLAB_PATH = r"E:\SplitResNet18\matlab_com"


FEATURE_PATH = r"E:\SplitResNet18\features_all.npy"


REC_FEATURE_PATH = r"E:\SplitResNet18\matlab_com\features_rec.npy"


LABEL_PATH = r"E:\SplitResNet18\labels.npy"



# 测试SNR

snr_list=[

    -5,
    0,
    5,
    10,
    15,
    20,
    25,
    30
]



# ==========================
# 加载feature
# ==========================


features=np.load(
    FEATURE_PATH
)


labels=np.load(
    LABEL_PATH
)



print(
    "Feature:",
    features.shape
)



# ==========================
# 加载模型
# ==========================


backbone=ResNet18_CIFAR100()


checkpoint=torch.load(

    r"E:\SplitResNet18\checkpoints\best_resnet18_cifar100.pth",

    map_location=device

)


backbone.load_state_dict(
    checkpoint
)



server=ServerModel(

    backbone,

    cut_layer=6

)



server=server.to(device)

server.eval()



# ==========================
# accuracy函数
# ==========================


def test_accuracy(features):


    x=torch.tensor(

        features,

        dtype=torch.float32

    )


    y=torch.tensor(

        labels,

        dtype=torch.long

    )


    correct=0

    total=0



    with torch.no_grad():


        batch=64


        for i in range(
            0,
            len(x),
            batch
        ):


            feature=x[i:i+batch].to(device)

            label=y[i:i+batch].to(device)



            output=server(feature)



            pred=output.argmax(1)



            correct += (

                pred==label

            ).sum().item()



            total += label.size(0)



    acc=100*correct/total


    return acc





# ==========================
# 主循环
# ==========================


results=[]



for snr in snr_list:


    print("====================")

    print(
        "SNR:",
        snr
    )
    print("Channel: Rician")


    # ----------------------
    # 生成 config.mat
    # ----------------------


    scipy.io.savemat(

        os.path.join(
            MATLAB_PATH,
            "config.mat"
        ),

        {

            "SNR":snr,

            "CHANNEL":1

        }

    )



    # ----------------------
    # 删除旧结果
    # ----------------------


    if os.path.exists(
        REC_FEATURE_PATH
    ):

        os.remove(
            REC_FEATURE_PATH
        )



    # ----------------------
    # 调用MATLAB
    # ----------------------


    cmd=[

        "matlab",

        "-batch",

        "main_batch"

    ]



    result=subprocess.run(

        cmd,

        cwd=MATLAB_PATH,

        stdout=subprocess.PIPE,

        stderr=subprocess.STDOUT,

        text=True

    )



    print(
        result.stdout
    )



    if result.returncode !=0:

        print(
            "MATLAB Error"
        )

        break



    # ----------------------
    # 读取恢复feature
    # ----------------------


    recv=np.load(

        REC_FEATURE_PATH

    )



    print(
        "Recovered:",
        recv.shape
    )



    # ----------------------
    # accuracy
    # ----------------------


    acc=test_accuracy(
        recv
    )


    print(

        "Accuracy:",

        acc

    )



    results.append(acc)



# ==========================
# 保存结果
# ==========================


np.save(

    "rician_accuracy_snr.npy",

    np.array(results)

)



print("====================")

print("SNR:")

print(snr_list)


print("Accuracy:")

print(results)



# ==========================
# 绘图
# ==========================


import matplotlib.pyplot as plt



plt.figure(figsize=(7,5))


plt.plot(

    snr_list,

    results,

    marker="o"

)



plt.xlabel(
    "SNR(dB)"
)


plt.ylabel(
    "Accuracy(%)"
)


plt.title(
    "Accuracy vs SNR (Rician Channel)"
)


plt.grid()


plt.savefig(

    "rician_accuracy_vs_snr.png",

    dpi=300

)


plt.show()