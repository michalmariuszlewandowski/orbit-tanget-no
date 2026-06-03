import torch

from otno.models import FNO1d, FNO2d


def test_fno1d_shape():
    model = FNO1d(in_channels=1, out_channels=2, width=8, modes=4, depth=2)
    x = torch.randn(3, 32, 1)
    y = model(x)
    assert y.shape == (3, 32, 2)


def test_fno2d_shape():
    model = FNO2d(in_channels=1, out_channels=1, width=8, modes1=4, modes2=4, depth=2)
    x = torch.randn(2, 16, 16, 1)
    y = model(x)
    assert y.shape == (2, 16, 16, 1)
