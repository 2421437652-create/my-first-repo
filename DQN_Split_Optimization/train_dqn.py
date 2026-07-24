import h5py
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from env.split_env import SplitEnv
from model.dqn import DQN
from utils.replay_buffer import ReplayBuffer



# ==========================
# 参数
# ==========================


DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


EPISODES = 5000


BATCH_SIZE = 64

LR = 1e-3

GAMMA = 0.9


MEMORY_SIZE = 50000


EPSILON_START = 1.0

EPSILON_END = 0.05

EPSILON_DECAY = 0.995



# ==========================
# 读取MATLAB v7.3
# ==========================


def load_mat73(path):

    data={}

    with h5py.File(path,"r") as f:

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



print("===================")

print("Environment")

print("===================")

print("Actions:",env.n_actions)



# ==========================
# 网络
# ==========================


policy_net=DQN(

    state_dim=1,

    action_dim=env.n_actions

).to(DEVICE)



target_net=DQN(

    state_dim=1,

    action_dim=env.n_actions

).to(DEVICE)



target_net.load_state_dict(

    policy_net.state_dict()

)



target_net.eval()



optimizer=optim.Adam(

    policy_net.parameters(),

    lr=LR

)



loss_fn=nn.MSELoss()



memory=ReplayBuffer(

    MEMORY_SIZE

)



epsilon=EPSILON_START



reward_history=[]



# ==========================
# Training
# ==========================


for episode in range(EPISODES):


    state=env.reset()


    total_reward=0



    done=False



    while not done:


        state_tensor=torch.tensor(

            state,

            dtype=torch.float32

        ).unsqueeze(0).to(DEVICE)



        # epsilon-greedy

        if np.random.rand()<epsilon:


            action=np.random.randint(

                env.n_actions

            )


        else:

            with torch.no_grad():

                q_values=policy_net(

                    state_tensor

                )

                action=torch.argmax(

                    q_values

                ).item()



        next_state,reward,done,info = env.step(action)



        memory.push(

            state,

            action,

            reward,

            next_state,

            done

        )



        state=next_state


        total_reward+=reward



        # ======================
        # 更新网络
        # ======================


        if len(memory)>BATCH_SIZE:


            states,actions,rewards,next_states,dones = memory.sample(

                BATCH_SIZE

            )


            states=torch.tensor(

                np.array(states),

                dtype=torch.float32

            ).to(DEVICE)



            actions=torch.tensor(

                actions,

                dtype=torch.long

            ).to(DEVICE)



            rewards=torch.tensor(

                rewards,

                dtype=torch.float32

            ).to(DEVICE)



            next_states=torch.tensor(

                np.array(next_states),

                dtype=torch.float32

            ).to(DEVICE)



            dones=torch.tensor(

                dones,

                dtype=torch.float32

            ).to(DEVICE)



            q_values=policy_net(states)


            q_values=q_values.gather(

                1,

                actions.unsqueeze(1)

            ).squeeze(1)



            with torch.no_grad():

                next_q=target_net(

                    next_states

                ).max(1)[0]



                target=rewards + GAMMA*next_q*(1-dones)



            loss=loss_fn(

                q_values,

                target

            )


            optimizer.zero_grad()

            loss.backward()

            optimizer.step()



    # epsilon衰减

    epsilon=max(

        EPSILON_END,

        epsilon*EPSILON_DECAY

    )



    reward_history.append(

        total_reward

    )

    if episode % 200 == 0:
        target_net.load_state_dict(
            policy_net.state_dict()
        )



    if episode%100==0:


        print(

            "Episode:",episode,

            "Reward:",round(total_reward,4),

            "Epsilon:",round(epsilon,3),

            "SP:",info["SP"],

            "Latency:",info["Latency"]

        )




# ==========================
# 保存
# ==========================


torch.save(

    policy_net.state_dict(),

    "./model/dqn_split.pth"

)


np.save(

    "./result/reward.npy",

    np.array(reward_history)

)



print("===================")

print("Training Finished")

print("===================")


torch.save(
    policy_net.state_dict(),
    "./result/dqn_split.pth"
)

print("Model saved")