import torch

from otno.models import FNO1d, FNO2d, MoleculeMLP, ObservableGalileanCanonicalFNO2d


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


def test_observable_galilean_canonical_fno2d_shape():
    model = ObservableGalileanCanonicalFNO2d(
        in_channels=3,
        out_channels=1,
        width=8,
        modes1=4,
        modes2=4,
        depth=2,
        add_grid=False,
    )
    x = torch.randn(2, 16, 16, 3)
    y = model(x)
    assert y.shape == (2, 16, 16, 1)


def test_molecule_mlp_shape():
    model = MoleculeMLP(n_atoms=9, in_channels=4, out_channels=3, hidden=16, depth=2)
    x = torch.randn(5, 9, 4)
    y = model(x)
    assert y.shape == (5, 9, 3)
