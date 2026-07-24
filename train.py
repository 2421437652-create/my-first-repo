# ============================================
# train.py
# Split Learning Training Entrance
# ============================================


import torch


from configs.config import cfg


from datasets.cifar import get_dataloader


from models.resnet import SplitResNet18


from models.builder import SplitBuilder


from engine.trainer import SplitTrainer



def main():


    # =====================================
    # Device
    # =====================================

    device = torch.device(
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


    print("=" * 60)

    print("Split Learning Training")

    print("Device:",
          device)

    print("=" * 60)



    # =====================================
    # Dataset
    # =====================================

    train_loader, test_loader = get_dataloader(
        cfg
    )



    # =====================================
    # Build Backbone
    # =====================================

    backbone = SplitResNet18(
        num_classes=cfg.num_classes
    )



    # =====================================
    # Split Model
    # =====================================

    builder = SplitBuilder(
        backbone,
        cfg.cut_layer
    )


    client_model, server_model = builder.build()



    # 移动GPU

    client_model = client_model.to(
        device
    )

    server_model = server_model.to(
        device
    )



    # =====================================
    # Trainer
    # =====================================

    trainer = SplitTrainer(

        client=client_model,

        server=server_model,

        train_loader=train_loader,

        test_loader=test_loader,

        cfg=cfg,

        device=device

    )



    # =====================================
    # Train
    # =====================================

    trainer = SplitTrainer(
        client_model,
        server_model,
        train_loader,
        test_loader,
        cfg,
        device
    )

    trainer.train()





if __name__ == "__main__":

    main()


