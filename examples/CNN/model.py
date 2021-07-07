import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    
    def __init__(self, vector_state_size, action_size, seed):
        """Initialize parameters and build model.
        Params
        ======
            vector_state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.vector_state_size = vector_state_size
        self.action_size = action_size
        
        # FC layers
        self.fc1 = nn.Linear(vector_state_size + 576, 128, bias=True)
        self.fc2 = nn.Linear(128, action_size, bias=True)
        
        
        # CNN layers
        # conv layer 1 (sees 19x33x1 image tensor)
        self.conv1 = nn.Conv2d(1, 4, kernel_size=3, stride=1, padding=(1,0))
        # conv layer 2 (sees 10x16x4 image tensor
        self.conv2 = nn.Conv2d(4, 16, 3, stride=1, padding=(1,0))
        # outputs 6x8x16
        # pooling
        self.pool = nn.MaxPool2d(2, stride=2, padding=(1,0))

    def forward(self, state_img, state_vec):
        """Build a network that maps state -> action values."""
        x = F.leaky_relu(self.conv1(state_img))
        x = self.pool(x)
        x = self.pool(F.leaky_relu(self.conv2(x)))
        x = x.view(-1,  576)
        x = torch.cat((x, state_vec), dim=1)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x
