# ============================================
# configs/config.py
# Global Configuration
# ============================================

import os


class Config:

    # =====================================
    # Dataset
    # =====================================
    dataset = "CIFAR10"

    data_root = "./data"

    num_classes = 10

    image_size = 224

    num_workers = 4

    # =====================================
    # Training
    # =====================================
    batch_size = 128

    epochs = 20

    learning_rate = 0.001

    momentum = 0.9

    weight_decay = 5e-4

    # =====================================
    # Optimizer
    # =====================================
    optimizer = "SGD"

    # =====================================
    # Split Learning
    # =====================================
    cut_layer = 5

    # =====================================
    # Device
    # =====================================
    device = "cuda"

    # =====================================
    # Save
    # =====================================
    save_dir = "./checkpoints"

    client_weight = os.path.join(
        save_dir,
        "client.pth"
    )

    server_weight = os.path.join(
        save_dir,
        "server.pth"
    )


cfg = Config()

resume=False

checkpoint="./checkpoints/best.pth"