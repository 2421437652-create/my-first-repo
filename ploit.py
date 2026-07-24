import numpy as np
import matplotlib.pyplot as plt


snr=[
-5,
0,
5,
10,
15,
20,
25,
30
]


awgn=np.load(
"accuracy_snr.npy"
)


rician=np.load(
"rician_accuracy_snr.npy"
)



plt.figure(figsize=(7,5))


plt.plot(
snr,
awgn,
marker="o",
label="AWGN"
)


plt.plot(
snr,
rician,
marker="s",
label="Rician"
)


plt.xlabel(
"SNR(dB)"
)


plt.ylabel(
"Accuracy(%)"
)


plt.legend()

plt.grid()


plt.savefig(
"AWGN_Rician_Accuracy_SNR.png",
dpi=300
)


plt.show()