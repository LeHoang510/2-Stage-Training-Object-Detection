import torch.nn as nn
import torch.nn.functional as F

class MnistModel(nn.Module):
    def __init__(self):
        super(MnistModel, self).__init__()
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(28 * 28, 128)
        self.fc2 = nn.Linear(128, 64)
        self.fc3 = nn.Linear(64, 10)

        self.relu = nn.ReLU()
        self.log_softmax = nn.LogSoftmax(dim=1)

        self._init_weights()

    def forward(self, image, target=None):
        """        Forward pass of the model.
        Args:
            image (torch.Tensor): Input images.
            target (torch.Tensor, optional): Target labels for training. Defaults to None.
        Returns:
            torch.Tensor: The output logits.
            If target is provided, also returns the loss.
        """
        x = self.flatten(image)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        x = self.log_softmax(x)

        if target is not None:
            loss_fn = nn.NLLLoss()
            loss = loss_fn(x, target)
            return x, {'loss': loss}

        return x, {}

    def _init_weights(self):
        """Initialize weights of the model."""
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_uniform_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
