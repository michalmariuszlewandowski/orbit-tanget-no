import torch

from otno.models import (
    D4GFNO2d,
    DeepONet2d,
    FNO1d,
    FNO2d,
    MoleculeMLP,
    ObservableGalileanCanonicalFNO2d,
    build_model,
)
from otno.symmetry.transforms import D4Pseudoscalar2D, NavierStokes2DGalilean, TransformSample
from otno.training.losses import orbit_consistency_loss


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


def test_deeponet2d_shape_and_parameter_gradients():
    model = DeepONet2d(
        in_channels=3,
        out_channels=2,
        grid_height=8,
        grid_width=6,
        branch_channels=(4, 8),
        branch_fc_hidden=16,
        trunk_hidden=12,
        trunk_depth=1,
        latent_dim=10,
        coordinate_modes=3,
    )
    x = torch.randn(4, 8, 6, 3)
    y = model(x)
    assert y.shape == (4, 8, 6, 2)
    y.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_deeponet2d_rejects_non_sensor_grid():
    model = DeepONet2d(grid_height=8, grid_width=8, coordinate_modes=2)
    with torch.no_grad():
        try:
            model(torch.randn(1, 7, 8, 1))
        except ValueError as exc:
            assert "fixed sensor grid" in str(exc)
        else:
            raise AssertionError("DeepONet2d accepted an input on the wrong sensor grid")


def test_build_deeponet2d_from_config():
    model = build_model(
        {
            "model": {
                "name": "deeponet2d",
                "in_channels": 3,
                "out_channels": 1,
                "grid_size": 8,
                "branch_channels": [4, 8],
                "branch_fc_hidden": 16,
                "trunk_hidden": 12,
                "trunk_depth": 1,
                "latent_dim": 10,
                "coordinate_modes": 2,
            }
        }
    )
    assert isinstance(model, DeepONet2d)
    assert model(torch.randn(2, 8, 8, 3)).shape == (2, 8, 8, 1)


def test_paper_deeponet2d_parameter_count():
    model = DeepONet2d(in_channels=3, out_channels=1)
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_688_033


def test_deeponet2d_galilean_orbit_loss_backpropagates():
    model = DeepONet2d(
        in_channels=3,
        out_channels=1,
        grid_height=8,
        grid_width=8,
        branch_channels=(4, 8),
        branch_fc_hidden=16,
        trunk_hidden=12,
        trunk_depth=1,
        latent_dim=10,
        coordinate_modes=2,
    )
    transform = NavierStokes2DGalilean(max_boost=0.25, final_time=0.5)
    loss, stats = orbit_consistency_loss(model, torch.randn(3, 8, 8, 3), transform)
    loss.backward()
    assert torch.isfinite(loss)
    assert stats["epsilon_mean"] > 0
    assert any(parameter.grad is not None for parameter in model.branch.parameters())


def test_d4_gfno2d_shape():
    model = D4GFNO2d(in_channels=1, out_channels=1, width=4, modes1=2, modes2=2, depth=1)
    x = torch.randn(2, 16, 16, 1)
    y = model(x)
    assert y.shape == (2, 16, 16, 1)


def test_d4_gfno2d_pseudoscalar_equivariance():
    torch.manual_seed(0)
    model = D4GFNO2d(in_channels=1, out_channels=1, width=4, modes1=2, modes2=2, depth=1)
    transform = D4Pseudoscalar2D()
    x = torch.randn(2, 16, 16, 1)
    sample = TransformSample(
        params={"k": torch.tensor([1, 2]), "flip": torch.tensor([False, True])},
        epsilon=torch.ones(2),
        name="d4_pseudoscalar2d",
    )
    lhs = model(transform.apply_input(x, sample))
    rhs = transform.apply_output(model(x), sample)
    assert torch.allclose(lhs, rhs, atol=1e-4, rtol=1e-4)


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
