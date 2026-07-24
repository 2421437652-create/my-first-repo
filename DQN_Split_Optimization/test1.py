import h5py
import numpy as np

from env.split_env import SplitEnv

from model.dqn import DQN
import torch




# ==========================
# 读取 MATLAB v7.3 文件
# ==========================

def load_mat73(path):

    data = {}

    with h5py.File(path, "r") as f:


        print("MAT keys:")

        print(list(f.keys()))



        for key in f.keys():

            data[key] = np.array(
                f[key]
            ).squeeze()



    return data




# ==========================
# Load dataset
# ==========================


mat_path = "./dataset/RL_dataset.mat"


data = load_mat73(mat_path)



dataset = {


    "SNR":
        data["SNR"],


    "SP":
        data["SP"],


    "Latency":
        data["Latency"],


    "Energy":
        data["Energy"],


    "Accuracy":
        data["Accuracy"]

}




print("===================")

print("Dataset loaded")

print("===================")


for k,v in dataset.items():

    print(
        k,
        v.shape
    )





# ==========================
# 创建环境
# ==========================


env = SplitEnv(dataset)



print("===================")

print("Action space")

print("===================")


print(env.action_space)



# ==========================
# 测试reset
# ==========================


state = env.reset()


print("===================")

print("Initial state")

print("===================")


print(state)




# ==========================
# 测试step
# ==========================


# 随便选择一个动作

action = 5



next_state,reward,done,info = env.step(action)



print("===================")

print("Step result")

print("===================")


print("Next state:")

print(next_state)



print("Reward:")

print(reward)



print("Done:")

print(done)



print("Info:")

print(info)


# ==========================
# 测试DQN网络
# ==========================

net = DQN(
    state_dim=1,
    action_dim=env.n_actions
)


state_tensor = torch.tensor(
    state,
    dtype=torch.float32
).unsqueeze(0)



with torch.no_grad():

    q_values = net(state_tensor)



print("===================")

print("Q values")

print("===================")

print(q_values)



best_action = torch.argmax(
    q_values,
    dim=1
).item()



print("===================")

print("Best action")

print("===================")

print(best_action)



print("Corresponding SP:")

print(
    env.action_space[best_action]
)



from utils.replay_buffer import ReplayBuffer


buffer=ReplayBuffer(100)


buffer.push(
    state,
    5,
    reward,
    next_state,
    done
)


print(
    "buffer size:",
    len(buffer)
)