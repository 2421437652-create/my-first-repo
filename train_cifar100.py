import torch
import torch.nn as nn
import torch.optim as optim

import torchvision
import torchvision.transforms as transforms

from torch.utils.data import DataLoader


from models.resnet18_cifar import ResNet18_CIFAR100



# =====================
# device
# =====================

device=torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)

print(device)



# =====================
# dataset
# =====================


transform_train=transforms.Compose([

    transforms.RandomCrop(
        32,
        padding=4
    ),

    transforms.RandomHorizontalFlip(),

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5071,0.4867,0.4408),
        (0.2675,0.2565,0.2761)
    )

])



transform_test=transforms.Compose([

    transforms.ToTensor(),

    transforms.Normalize(
        (0.5071,0.4867,0.4408),
        (0.2675,0.2565,0.2761)
    )

])



trainset=torchvision.datasets.CIFAR100(

    root="./data",

    train=True,

    download=True,

    transform=transform_train
)



testset=torchvision.datasets.CIFAR100(

    root="./data",

    train=False,

    download=True,

    transform=transform_test
)



trainloader=DataLoader(

    trainset,

    batch_size=128,

    shuffle=True,

    num_workers=0
)



testloader=DataLoader(

    testset,

    batch_size=128,

    shuffle=False,

    num_workers=0
)



# =====================
# model
# =====================


model=ResNet18_CIFAR100()

model=model.to(device)



criterion=nn.CrossEntropyLoss()


optimizer=optim.SGD(

    model.parameters(),

    lr=0.1,

    momentum=0.9,

    weight_decay=5e-4

)



scheduler=optim.lr_scheduler.MultiStepLR(

    optimizer,

    milestones=[100,150],

    gamma=0.1

)

# =====================
# test function
# =====================

def test(model, testloader):

    model.eval()

    correct = 0
    total = 0


    with torch.no_grad():

        for images, labels in testloader:

            images = images.to(device)
            labels = labels.to(device)


            outputs = model(images)


            pred = outputs.argmax(dim=1)


            total += labels.size(0)

            correct += (pred == labels).sum().item()



    acc = 100 * correct / total


    return acc

# =====================
# train
# =====================


epochs=200


# 保存最高测试准确率
best_acc=0



for epoch in range(epochs):


    model.train()


    total_loss=0

    correct=0

    total=0



    for images,labels in trainloader:


        images=images.to(device)

        labels=labels.to(device)



        optimizer.zero_grad()



        outputs=model(images)



        loss=criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()



        total_loss+=loss.item()


        pred=outputs.argmax(1)


        total+=labels.size(0)

        correct+=(pred==labels).sum().item()



    scheduler.step()



    acc=100*correct/total

    test_acc = test(
        model,
        testloader
    )

    # 测试集准确率

    test_acc = test(
        model,
        testloader
    )

    print(
        "Epoch:",
        epoch,
        "Loss:",
        total_loss,
        "Train Acc:",
        acc,
        "Test Acc:",
        test_acc
    )


    # 保存最佳模型

    if test_acc > best_acc:


        best_acc = test_acc

        torch.save(

            model.state_dict(),

            "./checkpoints/best_resnet18_cifar100.pth"

        )


        print(
            "Best model saved!",
            "Accuracy:",
            best_acc
        )


    print("Model saved")