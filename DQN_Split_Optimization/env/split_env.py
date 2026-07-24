import numpy as np


class SplitEnv:


    def __init__(self, dataset):


        self.SNR = dataset["SNR"]

        self.SP = dataset["SP"]

        self.Latency = dataset["Latency"]

        self.Energy = dataset["Energy"]

        self.Accuracy = dataset["Accuracy"]



        # 动作空间
        # SP=1~20

        self.action_space = np.arange(
            1,
            21
        )


        self.n_actions = len(
            self.action_space
        )


        self.index = 0


        self.state = None



        # ==========================
        # 归一化参数
        # ==========================

        self.lat_min = np.min(
            self.Latency
        )

        self.lat_max = np.max(
            self.Latency
        )


        self.energy_min = np.min(
            self.Energy
        )

        self.energy_max = np.max(
            self.Energy
        )



    # ==========================
    # reset
    # ==========================

    def reset(self):


        # 随机选择一个信道状态

        self.index = np.random.randint(
            len(self.SNR)
        )


        snr = self.SNR[
            self.index
        ]


        self.state=np.array(
            [
                snr
            ],
            dtype=np.float32
        )


        return self.state



    # ==========================
    # step
    # ==========================

    def step(self, action):


        # action 0~19

        sp = self.action_space[
            action
        ]



        # 找对应样本

        mask=(

            (self.SNR==self.SNR[self.index])

            &

            (self.SP==sp)

        )



        idx=np.where(
            mask
        )[0]



        if len(idx)==0:


            idx=self.index


        else:

            idx=np.random.choice(
                idx
            )



        latency=self.Latency[idx]


        energy=self.Energy[idx]


        accuracy=self.Accuracy[idx]



        # ==============================
        # 归一化
        # ==============================


        lat_norm=(

            latency-self.lat_min

        )/(

            self.lat_max-self.lat_min+1e-9

        )



        energy_norm=(

            energy-self.energy_min

        )/(

            self.energy_max-self.energy_min+1e-9

        )




        # ==============================
        # 新Reward
        # ==============================

        reward = (

                accuracy

                - 0.5 * lat_norm

                - 0.2 * energy_norm

        )


        # ==============================
        # 下一状态
        # ==============================


        next_state=np.array(

            [
                self.SNR[idx]

            ],

            dtype=np.float32

        )



        done=True



        info={


            "SNR":

                self.SNR[idx],


            "SP":

                sp,


            "Latency":

                latency,


            "Energy":

                energy,


            "Accuracy":

                accuracy,


            "Reward":

                reward

        }



        return (

            next_state,

            reward,

            done,

            info

        )