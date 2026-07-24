import numpy as np
import os
import torch


SAVE_PATH = r"E:\SplitResNet18\matlab_com\input_feature.npy"



def send_to_matlab(feature):

    """
    Python Client
    ↓
    MATLAB

    保存feature
    """

    # GPU -> CPU
    feature_cpu = (
        feature
        .detach()
        .cpu()
        .numpy()
    )


    # 创建路径
    os.makedirs(
        "../matlab_comm",
        exist_ok=True
    )


    # 保存
    np.save(
        SAVE_PATH,
        feature_cpu
    )


    print(
        "Feature sent to MATLAB:",
        feature_cpu.shape
    )


    return feature_cpu.shape