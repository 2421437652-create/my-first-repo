# ============================================
# engine/trainer.py
# ============================================


import torch
import torch.nn as nn

from engine.tester import Tester

from engine.utils import (
    create_dir,
    save_checkpoint,
    save_log
)



class SplitTrainer:


    def __init__(
            self,
            client,
            server,
            train_loader,
            test_loader,
            cfg,
            device):


        self.client = client

        self.server = server


        self.train_loader=train_loader

        self.device=device


        self.cfg=cfg



        self.criterion = nn.CrossEntropyLoss()



        self.client_optimizer=torch.optim.SGD(
            client.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )


        self.server_optimizer=torch.optim.SGD(
            server.parameters(),
            lr=cfg.learning_rate,
            momentum=cfg.momentum,
            weight_decay=cfg.weight_decay
        )



        self.tester = Tester(
            client,
            server,
            test_loader,
            device
        )



        create_dir(
            cfg.save_dir
        )



        self.history={

            "loss":[],

            "accuracy":[]

        }



        self.best_acc=0



    def train_epoch(self):


        self.client.train()

        self.server.train()


        total_loss=0



        for images,labels in self.train_loader:


            images=images.to(
                self.device
            )

            labels=labels.to(
                self.device
            )


            # Client forward

            feature=self.client(
                images
            )


            feature=feature.detach()

            feature.requires_grad=True





            # Server forward

            output=self.server(
                feature
            )


            loss=self.criterion(
                output,
                labels
            )



            # Server backward


            self.server_optimizer.zero_grad()


            loss.backward()



            grad=feature.grad.clone()



            self.server_optimizer.step()



            # Client backward


            self.client_optimizer.zero_grad()


            feature.backward(
                grad
            )


            self.client_optimizer.step()



            total_loss += loss.item()



        return total_loss / len(
            self.train_loader
        )



    def train(self):


        for epoch in range(
            self.cfg.epochs
        ):


            loss=self.train_epoch()


            acc=self.tester.test()



            self.history["loss"].append(
                loss
            )


            self.history["accuracy"].append(
                acc
            )



            print(
                f"Epoch [{epoch+1}/{self.cfg.epochs}] "
                f"Loss:{loss:.4f} "
                f"Accuracy:{acc:.2f}%"
            )



            # 保存最佳模型

            if acc > self.best_acc:


                self.best_acc=acc


                save_checkpoint(

                    self.client,

                    self.server,

                    self.client_optimizer,

                    self.server_optimizer,

                    epoch,

                    acc,

                    self.cfg.save_dir+
                    "/best.pth"

                )


        save_log(

            self.history,

            self.cfg.save_dir+
            "/history.json"

        )


        print(
            "Best Accuracy:",
            self.best_acc
        )