# ============================================
# engine/tester.py
# ============================================


import torch



class Tester:


    def __init__(
            self,
            client,
            server,
            test_loader,
            device):


        self.client = client

        self.server = server

        self.test_loader = test_loader

        self.device = device



    @torch.no_grad()
    def test(self):


        self.client.eval()

        self.server.eval()


        correct = 0

        total = 0



        for images, labels in self.test_loader:


            images = images.to(
                self.device
            )

            labels = labels.to(
                self.device
            )



            feature = self.client(
                images
            )


            output = self.server(
                feature
            )



            prediction = torch.argmax(
                output,
                dim=1
            )


            total += labels.size(0)


            correct += (
                prediction == labels
            ).sum().item()



        acc = 100 * correct / total


        return acc