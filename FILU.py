class Fish(nn.Module):
    def __init__(self):
        super(Fish, self).__init__()

    def forward(self, x):

        # kilu06
        return x*torch.exp(torch.tanh(0.6*x))
