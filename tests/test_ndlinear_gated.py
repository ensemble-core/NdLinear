import torch
import pytest

from typing import get_type_hints, Literal

from NdLinear import NdLinearGated

def test_ndlinear_gated_init():
    input_dims = (8, 16)
    hidden_size = (16, 32)
    
    model = NdLinearGated(input_dims, hidden_size)
    
    assert model.input_dims == input_dims
    assert model.hidden_size == hidden_size
    assert model.num_layers == len(input_dims)
    assert model.transform_outer == True
    assert model.gating_mode == "soft"
    assert model.gating_hidden_dim == 16
    assert model.gated_modes == "all"


def test_ndlinear_gated_forward_soft_all():
    batch_size = 4
    input_dims = (8, 16)
    hidden_size = (16, 32)
    
    model = NdLinearGated(
        input_dims=input_dims,
        hidden_size=hidden_size,
        gating_mode="soft",
        gated_modes="all"
    )
    
    x = torch.randn(batch_size, *input_dims)
    output = model(x)
    
    assert output.shape == (batch_size, *hidden_size)


def test_ndlinear_gated_forward_hard_all():
    batch_size = 4
    input_dims = (8, 16)
    hidden_size = (16, 32)
    
    model = NdLinearGated(
        input_dims=input_dims,
        hidden_size=hidden_size,
        gating_mode="hard",
        gated_modes="all"
    )
    
    x = torch.randn(batch_size, *input_dims)
    output = model(x)
    
    assert output.shape == (batch_size, *hidden_size)


def test_ndlinear_gated_forward_soft_first():
    batch_size = 4
    input_dims = (8, 16)
    hidden_size = (16, 32)
    
    model = NdLinearGated(
        input_dims=input_dims,
        hidden_size=hidden_size,
        gating_mode="soft",
        gated_modes="first"
    )
    
    x = torch.randn(batch_size, *input_dims)
    output = model(x)
    
    assert output.shape == (batch_size, *hidden_size)


def test_ndlinear_gated_forward_topk():
    batch_size = 4
    input_dims = (8, 16)
    hidden_size = (16, 32)
    
    model = NdLinearGated(
        input_dims=input_dims,
        hidden_size=hidden_size,
        gated_modes="topk"
    )
    
    x = torch.randn(batch_size, *input_dims)
    
    assert model.topk_modes is None
    assert model.first_batch_processed is False
    
    output = model(x)
    
    assert model.topk_modes is not None
    assert model.first_batch_processed is True
    assert output.shape == (batch_size, *hidden_size)
    
    # Second call should reuse topk modes
    output2 = model(x)
    assert output2.shape == (batch_size, *hidden_size)


def test_transform_inner_outer():
    batch_size = 4
    input_dims = (8, 16)
    hidden_size = (16, 32)
    
    # Test with transform_outer = True (default)
    model_outer = NdLinearGated(
        input_dims=input_dims,
        hidden_size=hidden_size,
        transform_outer=True
    )
    
    # Test with transform_outer = False
    model_inner = NdLinearGated(
        input_dims=input_dims,
        hidden_size=hidden_size,
        transform_outer=False
    )
    
    x = torch.randn(batch_size, *input_dims)
    
    output_outer = model_outer(x)
    output_inner = model_inner(x)
    
    assert output_outer.shape == (batch_size, *hidden_size)
    assert output_inner.shape == (batch_size, *hidden_size)
    # The outputs should be different due to different transform orders
    assert not torch.allclose(output_outer, output_inner)


def test_invalid_input_dims():
    input_dims = (8, 16)
    hidden_size = (16, 32, 64)  # Mismatch with input_dims
    
    with pytest.raises(Exception):
        NdLinearGated(input_dims, hidden_size)


def test_ndlinear_gated_repr():
    input_dims = (8, 16)
    hidden_size = (16, 32)
    
    model = NdLinearGated(
        input_dims=input_dims,
        hidden_size=hidden_size,
        transform_outer=False,
        gating_mode="hard",
        gating_hidden_dim=32,
        gated_modes="first"
    )
    
    expected_repr = (
        f"NdLinearGated(input_dims={input_dims}, "
        f"hidden_size={hidden_size}, transform_outer=False, "
        f"gating_mode=hard, gating_hidden_dim=32, "
        f"gated_modes=first)"
    )
    
    assert repr(model) == expected_repr


def test_ndlinear_gated_type_hints():
    model = NdLinearGated((8, 16), (16, 32))
    
    init_hints = get_type_hints(model.__init__)
    assert init_hints['input_dims'] == tuple
    assert init_hints['hidden_size'] == tuple
    assert init_hints['transform_outer'] == bool
    assert init_hints['gating_mode'] == Literal["soft", "hard"]
    assert init_hints['gating_hidden_dim'] == int
    assert init_hints['gated_modes'] == Literal["all", "first", "topk"]
    assert init_hints['return'] == type(None)
    
    forward_hints = get_type_hints(model.forward)
    assert forward_hints['X'] == torch.Tensor
    assert forward_hints['return'] == torch.Tensor
    
    topk_hints = get_type_hints(model._compute_topk_modes)
    assert topk_hints['X'] == torch.Tensor
    assert topk_hints['return'] == list
    
    repr_hints = get_type_hints(model.__repr__)
    assert repr_hints['return'] == str 