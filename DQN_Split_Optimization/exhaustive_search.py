import h5py
import numpy as np


# =====================
# load dataset
# =====================

data = h5py.File(
    "./dataset/RL_dataset.mat",
    "r"
)


Accuracy = np.array(
    data["Accuracy"]
).flatten()


Latency = np.array(
    data["Latency"]
).flatten()


Energy = np.array(
    data["Energy"]
).flatten()


SNR = np.array(
    data["SNR"]
).flatten()


SP = np.array(
    data["SP"]
).flatten()


# =====================
# normalization
# =====================

lat_norm = (
    Latency - Latency.min()
) / (
    Latency.max()-Latency.min()+1e-9
)


energy_norm = (
    Energy-Energy.min()
) / (
    Energy.max()-Energy.min()+1e-9
)


# reward

Reward = (
    Accuracy
    -0.5*lat_norm
    -0.2*energy_norm
)



# =====================
# exhaustive search
# =====================

snr_list=np.unique(SNR)


print("====================")
print("Exhaustive Search")
print("====================")


best_results={}


for snr in snr_list:


    index=np.where(
        SNR==snr
    )[0]


    reward_snr=Reward[index]
    sp_snr=SP[index]


    best=np.argmax(
        reward_snr
    )


    best_sp=sp_snr[best]
    best_reward=reward_snr[best]


    best_results[snr]=best_sp


    print(
        f"SNR={snr:>5} dB  "
        f"Best SP={int(best_sp)}  "
        f"Reward={best_reward:.4f}"
    )


print("====================")
print("Finished")