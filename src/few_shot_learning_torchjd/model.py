import torch
import torch.nn.functional as F
from torch import nn


class MNISTClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(64 * 7 * 7, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=2)
        x = x.view(-1, 64 * 7 * 7)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
    

class JointMNISTClassifier(nn.Module):
    def __init__(self, init_mode: str | None = None):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        #self.fc1 = nn.Linear(10, 128)
        #self.fc11 = nn.Linear(64, 256)
        self.fc2 = nn.Linear(64 * 7 * 7, 128)
        self.fcmu = nn.Linear(256, 32)
        self.fclogvar = nn.Linear(256, 32)
        self.fc4 = nn.Linear(128, 10)
        self.fc5 = nn.Linear(256, 1)
        self.classifier = MNISTClassifier()

        # if init_mode == "zeros":
        #     #nn.init.normal_(self.fc1.weight, std=0.01)
        #     #nn.init.normal_(self.fc11.weight, std=0.01)
        #     nn.init.normal_(self.fc2.weight, std=0.01)
        #     nn.init.normal_(self.fc3.weight, std=0.01)
        #     nn.init.normal_(self.fc4.weight, std=0.01)
        #     #nn.init.zeros_(self.fc1.bias)
        #     ##nn.init.zeros_(self.fc11.bias)
        #     nn.init.zeros_(self.fc2.bias)
        #     nn.init.zeros_(self.fc3.bias)
        #     nn.init.zeros_(self.fc4.bias)
        #     nn.init.normal_(self.conv1.weight, std=0.01)
        #     nn.init.normal_(self.conv2.weight, std=0.01)
        #     nn.init.zeros_(self.conv1.bias)
        #     nn.init.zeros_(self.conv2.bias)

    def forward(self, x: tuple[torch.Tensor, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        image, label = x
        
        #label = F.relu(self.fc1(label))
        #label = F.relu(self.fc11(label))

        # image = F.relu(self.conv1(image))
        # image = F.max_pool2d(image, kernel_size=2)
        # image = F.relu(self.conv2(image))
        # image = F.max_pool2d(image, kernel_size=2)
        # image = image.view(-1, 64 * 7 * 7)
        # image = F.relu(self.fc2(image))
        # image = F.relu(self.fc4(image))

        # join_embedding = torch.dot(image, )
        # joint_embedding = torch.cat((image, label), dim=1)
        # joint_embedding = F.relu(self.fc2(joint_embedding))
        # mu = F.relu(self.fcmu(joint_embedding))
        # logvar = F.relu(self.fclogvar(joint_embedding))

        # z = self.reparameterize(mu, logvar)

        image = self.classifier(image)
        joint_embedding = torch.bmm(image.unsqueeze(1), label.unsqueeze(2))
        joint_embedding = joint_embedding.view(-1, 1)

        #joint_embedding = F.relu(self.fc4(joint_embedding))
        #joint_embedding = F.sigmoid(self.fc5(joint_embedding))
        return joint_embedding, torch.Tensor([0]), torch.Tensor([0])#, mu, logvar
    

    def predict(self, image, n_iterations: int, learning_rate: float) -> torch.Tensor:
        """Given an image, finds the most likely label using gradient descent."""
        self.eval()

        labels_for_grad = torch.zeros((image.shape[0], 10), device=image.device, requires_grad=True)
            
        optimizer = torch.optim.Adam([labels_for_grad], lr=learning_rate)

        for _ in range(n_iterations):
            optimizer.zero_grad()
            output = self((image, labels_for_grad))[0]
            loss = F.mse_loss(output, torch.ones_like(output))
            loss.backward()
            optimizer.step()

        final_labels = labels_for_grad.detach()
        prediction = torch.argmax(final_labels, dim=1).round().type(torch.int64)
        return prediction
    

    def predict_exhaustive(self, image) -> torch.Tensor:
        """Given an image, finds the most likely label using gradient descent."""
        self.eval()

        outputs = []

        with torch.no_grad():
            for i in range(10):
                output = self((image, F.one_hot(torch.tensor(i, device=image.device), num_classes=10).float().repeat(image.shape[0], 1)))[0]
                outputs.append(output)

        outputs = torch.stack(outputs, dim=0)
        prediction = torch.argmax(outputs, dim=0).round().type(torch.int64)
        prediction = prediction.view(-1)
        return prediction

    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var) from
        N(0,1).
        :param mu: (Tensor) Mean of the latent Gaussian [B x D]
        :param logvar: (Tensor) Standard deviation of the latent Gaussian [B x D]
        :return: (Tensor) [B x D]
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return eps * std + mu

def loss_function(self,
                      *args,
                      **kwargs) -> dict:
        r"""
        Computes the VAE loss function.
        KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}
        :param args:
        :param kwargs:
        :return:
        """
        recons = args[0]
        input = args[1]
        mu = args[2]
        log_var = args[3]

        kld_weight = kwargs['M_N'] # Account for the minibatch samples from the dataset
        recons_loss =F.mse_loss(recons, input)


        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim = 1), dim = 0)

        loss = recons_loss + kld_weight * kld_loss
        return {'loss': loss,
                'Reconstruction_Loss':recons_loss,
                'KLD': kld_loss}