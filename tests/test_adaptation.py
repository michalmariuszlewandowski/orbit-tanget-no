import torch

from otno.training.adaptation import _parameter_l2_to_initial


def test_parameter_l2_to_initial_is_real_for_complex_parameters():
    param = torch.nn.Parameter(torch.tensor([1.0 + 2.0j], dtype=torch.complex64))
    initial = [torch.zeros_like(param)]

    reg = _parameter_l2_to_initial([param], initial)

    assert not torch.is_complex(reg)
    assert torch.isclose(reg, torch.tensor(5.0))
    reg.backward()
    assert param.grad is not None
