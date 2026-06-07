import torch

from otno.training.adaptation import _parameter_l2_to_initial, _prediction_preservation_loss


def test_parameter_l2_to_initial_is_real_for_complex_parameters():
    param = torch.nn.Parameter(torch.tensor([1.0 + 2.0j], dtype=torch.complex64))
    initial = [torch.zeros_like(param)]

    reg = _parameter_l2_to_initial([param], initial)

    assert not torch.is_complex(reg)
    assert torch.isclose(reg, torch.tensor(5.0))
    reg.backward()
    assert param.grad is not None


def test_prediction_preservation_loss_is_zero_for_matching_predictions():
    pred = torch.randn(4, 8, 1)
    loss = _prediction_preservation_loss(pred, pred.detach())

    assert torch.isclose(loss, torch.tensor(0.0))


def test_prediction_preservation_loss_is_relative_squared_error():
    pred = torch.tensor([[[2.0], [0.0]]])
    reference = torch.tensor([[[1.0], [1.0]]])

    loss = _prediction_preservation_loss(pred, reference)

    assert torch.isclose(loss, torch.tensor(1.0))
