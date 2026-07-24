import scipy.io


scipy.io.savemat(

    "E:/SplitResNet18/matlab_com/config.mat",

    {
        "SNR": 10,
        "CHANNEL": 0
    }

)


print("config.mat generated")