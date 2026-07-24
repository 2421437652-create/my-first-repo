# ============================================
# engine/utils.py
# Training utilities
# ============================================


import os
import torch
import json



def create_dir(path):

    if not os.path.exists(path):

        os.makedirs(path)



def save_checkpoint(
        client,
        server,
        optimizer_client,
        optimizer_server,
        epoch,
        accuracy,
        path):


    checkpoint = {

        "epoch": epoch,

        "accuracy": accuracy,

        "client":
            client.state_dict(),

        "server":
            server.state_dict(),

        "optimizer_client":
            optimizer_client.state_dict(),

        "optimizer_server":
            optimizer_server.state_dict()
    }


    torch.save(
        checkpoint,
        path
    )



def load_checkpoint(
        path,
        client,
        server,
        optimizer_client,
        optimizer_server,
        device):


    checkpoint = torch.load(
        path,
        map_location=device
    )


    client.load_state_dict(
        checkpoint["client"]
    )


    server.load_state_dict(
        checkpoint["server"]
    )


    optimizer_client.load_state_dict(
        checkpoint["optimizer_client"]
    )


    optimizer_server.load_state_dict(
        checkpoint["optimizer_server"]
    )


    epoch = checkpoint["epoch"]

    acc = checkpoint["accuracy"]


    return epoch, acc



def save_log(history,path):


    with open(
        path,
        "w"
    ) as f:


        json.dump(
            history,
            f,
            indent=4
        )