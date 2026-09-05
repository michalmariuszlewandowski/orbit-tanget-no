import pytest
import torch

from otno.models import build_model
from otno.models.cno import CNO2d, periodic_resize2d
from otno.symmetry.transforms import NavierStokes2DGalilean
from otno.training.losses import orbit_consistency_loss


def _small_cno(*, n_layers: int = 1) -> CNO2d:
    return CNO2d(
        in_channels=3,
        out_channels=1,
        n_layers=n_layers,
        n_res=1,
        n_res_neck=1,
        channel_multiplier=4,
        lift_project_channels=8,
        resample_halo=4,
    )


@pytest.mark.parametrize("output_size", [(8, 8), (16, 16), (32, 32)])
def test_periodic_resize_preserves_constants(output_size):
    x = torch.full((2, 3, 16, 16), 1.75)
    y = periodic_resize2d(x, output_size, halo=8)
    assert y.shape == (2, 3, *output_size)
    assert torch.allclose(y, torch.full_like(y, 1.75), atol=2e-6, rtol=0.0)


def test_periodic_resize_is_roll_equivariant_across_the_seam():
    torch.manual_seed(0)
    x = torch.randn(1, 2, 16, 16)

    upsampled = periodic_resize2d(x, (32, 32), halo=8)
    shifted_up = periodic_resize2d(torch.roll(x, (1, -2), (-2, -1)), (32, 32), halo=8)
    assert torch.allclose(
        shifted_up,
        torch.roll(upsampled, (2, -4), (-2, -1)),
        atol=2e-5,
        rtol=2e-5,
    )

    downsampled = periodic_resize2d(upsampled, (16, 16), halo=8)
    shifted_down = periodic_resize2d(
        torch.roll(upsampled, (4, -2), (-2, -1)),
        (16, 16),
        halo=8,
    )
    assert torch.allclose(
        shifted_down,
        torch.roll(downsampled, (2, -1), (-2, -1)),
        atol=2e-5,
        rtol=2e-5,
    )


def test_cno2d_shape_and_parameter_gradients():
    torch.manual_seed(1)
    model = _small_cno(n_layers=2)
    x = torch.randn(2, 16, 16, 3)
    y = model(x)
    assert y.shape == (2, 16, 16, 1)
    y.square().mean().backward()
    assert all(parameter.grad is not None for parameter in model.parameters())


def test_cno2d_rejects_incompatible_grid():
    model = _small_cno(n_layers=2)
    with pytest.raises(ValueError, match="divisible"):
        model(torch.randn(1, 14, 16, 3))


def test_cno2d_is_periodic_for_aligned_grid_rolls():
    torch.manual_seed(4)
    model = _small_cno().eval()
    x = torch.randn(1, 8, 8, 3)
    with torch.no_grad():
        output = model(x)
        shifted_output = model(torch.roll(x, (2, -2), (-3, -2)))
    assert torch.allclose(
        shifted_output,
        torch.roll(output, (2, -2), (-3, -2)),
        atol=2e-5,
        rtol=2e-5,
    )


def test_paper_cno2d_parameter_count():
    model = CNO2d(
        in_channels=3,
        out_channels=1,
        n_layers=4,
        n_res=4,
        n_res_neck=3,
        channel_multiplier=20,
        lift_project_channels=64,
        use_batch_norm=False,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) == 2_667_743


def test_build_cno2d_from_all_config_fields():
    model = build_model(
        {
            "model": {
                "name": "cno2d",
                "in_channels": 3,
                "out_channels": 2,
                "n_layers": 2,
                "n_res": 1,
                "n_res_neck": 2,
                "channel_multiplier": 6,
                "lift_project_channels": 10,
                "use_batch_norm": False,
                "resample_halo": 4,
                "negative_slope": 0.2,
            }
        }
    )
    assert isinstance(model, CNO2d)
    assert model.in_channels == 3
    assert model.out_channels == 2
    assert model.n_layers == 2
    assert model.n_res == 1
    assert model.n_res_neck == 2
    assert model.channel_multiplier == 6
    assert model.lift_project_channels == 10
    assert model.use_batch_norm is False
    assert model.resample_halo == 4
    assert model.negative_slope == pytest.approx(0.2)
    assert model(torch.randn(1, 8, 8, 3)).shape == (1, 8, 8, 2)


def test_cno2d_retains_constant_boost_channels():
    torch.manual_seed(2)
    model = _small_cno().eval()
    x = torch.randn(1, 8, 8, 3)
    x[..., 1:] = 0.0
    boosted = x.clone()
    boosted[..., 1] = 0.25
    boosted[..., 2] = -0.125
    with torch.no_grad():
        unboosted_output = model(x)
        boosted_output = model(boosted)
    assert not torch.allclose(unboosted_output, boosted_output, atol=1e-7, rtol=1e-7)

    differentiable_input = boosted.clone().requires_grad_(True)
    model(differentiable_input).square().mean().backward()
    boost_gradient = differentiable_input.grad[..., 1:]
    assert torch.isfinite(boost_gradient).all()
    assert boost_gradient.abs().sum() > 0


def test_cno2d_galilean_orbit_loss_backpropagates():
    torch.manual_seed(3)
    model = _small_cno()
    transform = NavierStokes2DGalilean(max_boost=0.25, final_time=0.5)
    loss, stats = orbit_consistency_loss(model, torch.randn(2, 8, 8, 3), transform)
    loss.backward()
    assert torch.isfinite(loss)
    assert stats["epsilon_mean"] > 0
    assert any(parameter.grad is not None for parameter in model.parameters())
