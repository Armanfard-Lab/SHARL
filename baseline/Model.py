import torch
import torch.nn as nn 
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') 
class bi_LSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, num_classes,dropout_rate):
        super(bi_LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.lstm = nn.LSTM(input_size, hidden_size,
                            num_layers, batch_first=True,
                            dropout=dropout_rate, bidirectional=True)
        self.fc = nn.Linear(hidden_size * 2, num_classes)

        self.act = nn.Softmax(dim=1)

    def forward(self, x):
        x = x.float()
        h0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        c0 = torch.zeros(self.num_layers * 2, x.size(0), self.hidden_size).to(device)
        out, _ = self.lstm(x, (h0, c0))
        out = self.fc(torch.cat((out[:, -1, :self.hidden_size], out[:, 0, self.hidden_size:]), 1))
        out = self.act(out)
        return out


class CNN(nn.Module):   
    def __init__(self,in_channels,num_classes):
        super(CNN, self).__init__()

        self.cnn_layers = nn.Sequential(
            # Defining a 2D convolution layer
            nn.Conv2d(in_channels, 32, kernel_size=3),
            # BatchNorm2d(4),
            nn.ReLU(inplace=True),
            # Defining another 2D convolution layer
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            # BatchNorm2d(4),
            nn.ReLU(inplace=True),

            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            # BatchNorm2d(4),
            nn.ReLU(inplace=True),
        )

        self.linear_layers = nn.Sequential(
            nn.Linear(3584, num_classes)
        )

    # Defining the forward pass    
    def forward(self, x):
        x = x.float()
        x = self.cnn_layers(x)
        # x=x.size
        x = x.view(x.size(0), -1)
        x = self.linear_layers(x)
        return x
