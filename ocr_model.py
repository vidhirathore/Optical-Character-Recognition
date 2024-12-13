import torch.nn as nn

class CRNN(nn.Module):
    def __init__(self, vocab_size, cnn_out_dim=128, rnn_hidden_dim=256, num_rnn_layers=2, dropout_rate=0.5):
        super(CRNN, self).__init__()
        
        # CNN Encoder
        self.cnn = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
            
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),
        )
        
        self.fc = nn.Linear(128 * 16 * 4, cnn_out_dim) 
        self.rnn = nn.LSTM(
            input_size=cnn_out_dim,
            hidden_size=rnn_hidden_dim,
            num_layers=num_rnn_layers,
            batch_first=True,
            dropout=dropout_rate,
        )
        
        self.output_layer = nn.Linear(rnn_hidden_dim, vocab_size)
        
    def forward(self, x):
        batch_size = x.size(0)
        
        x = self.cnn(x) 
        x = x.view(batch_size, -1) 
        x = self.fc(x).unsqueeze(1) 
        x, _ = self.rnn(x) 
        x = self.output_layer(x)  
        return x
