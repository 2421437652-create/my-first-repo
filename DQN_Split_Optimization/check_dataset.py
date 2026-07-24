import h5py
import numpy as np


# ===============================
# Dataset path
# ===============================

DATA_PATH = "./dataset/RL_dataset.mat"


# ===============================
# Load MATLAB v7.3
# ===============================

print("====================")
print("Loading dataset")
print("====================")


with h5py.File(DATA_PATH, "r") as f:

    print("MAT keys:")
    print(list(f.keys()))

    SP = np.array(f["SP"]).flatten()
    SNR = np.array(f["SNR"]).flatten()
    Latency = np.array(f["Latency"]).flatten()
    Energy = np.array(f["Energy"]).flatten()
    Accuracy = np.array(f["Accuracy"]).flatten()


# ===============================
# Basic information
# ===============================

print("\n====================")
print("Basic information")
print("====================")


N = len(SP)

print("Sample number:")
print(N)


print("\nSplit points:")
print(np.unique(SP))


print("\nSNR:")
print(np.unique(SNR))


print("\nLatency:")
print(
    "min:",
    np.min(Latency),
    "max:",
    np.max(Latency),
    "mean:",
    np.mean(Latency)
)


print("\nEnergy:")
print(
    "min:",
    np.min(Energy),
    "max:",
    np.max(Energy),
    "mean:",
    np.mean(Energy)
)


print("\nAccuracy:")
print(
    "min:",
    np.min(Accuracy),
    "max:",
    np.max(Accuracy),
    "mean:",
    np.mean(Accuracy)
)



# ===============================
# First samples
# ===============================

print("\n====================")
print("First 5 samples")
print("====================")


for i in range(5):

    print(
        "SNR:",
        SNR[i],
        "SP:",
        SP[i],
        "Latency:",
        Latency[i],
        "Energy:",
        Energy[i],
        "Accuracy:",
        Accuracy[i]
    )



# ===============================
# Check SNR distribution
# ===============================

print("\n====================")
print("SNR distribution")
print("====================")


snr_list=np.unique(SNR)


for snr in snr_list:

    num=np.sum(SNR==snr)

    print(
        "SNR:",
        snr,
        "samples:",
        num
    )



# ===============================
# Check SP distribution
# ===============================

print("\n====================")
print("SP distribution")
print("====================")


sp_list=np.unique(SP)


for sp in sp_list:

    num=np.sum(SP==sp)

    print(
        "SP:",
        sp,
        "samples:",
        num
    )



# ===============================
# Check SNR-SP combinations
# ===============================


print("\n====================")
print("Checking SNR-SP pairs")
print("====================")


flag=True


for snr in snr_list:

    for sp in sp_list:


        num=np.sum(
            (SNR==snr)
            &
            (SP==sp)
        )


        print(
            "SNR:",
            snr,
            "SP:",
            sp,
            "Number:",
            num
        )


        if num!=10000:

            print(
                "WARNING:",
                "wrong number!"
            )

            flag=False



if flag:

    print("\nALL SNR-SP pairs are correct!")



# ===============================
# Check transition
# ===============================


print("\n====================")
print("Check SP transition")
print("====================")


# 找SP变化位置

change_index=np.where(
    SP[1:]!=SP[:-1]
)[0]


for idx in change_index[:10]:

    print(
        "Index:",
        idx,
        "before:",
        SP[idx],
        "after:",
        SP[idx+1],
        "SNR:",
        SNR[idx]
    )



# ===============================
# Random samples
# ===============================


print("\n====================")
print("Random samples")
print("====================")


index=np.random.randint(
    0,
    N,
    10
)


for i in index:

    print(
        "SNR:",
        SNR[i],
        "SP:",
        SP[i],
        "Latency:",
        Latency[i],
        "Energy:",
        Energy[i],
        "Accuracy:",
        Accuracy[i]
    )



# ===============================
# Finished
# ===============================


print("\n====================")
print("Dataset check finished")
print("====================")