from torch.nn import Module


class BaseModel(Module):

    def forward(self, *input):
        pass

    def __init__(self):
        super().__init__()
