# interpretation.py
import numpy as np
import pandas as pd
import torch
from captum.attr import Saliency, GradientShap, DeepLift, IntegratedGradients, NoiseTunnel


def _to_device(x, device):
    return x if x.device == device else x.to(device)

def ensure_2d(x):
    if x.ndim != 2:
        raise ValueError("inputs must be 2D (n_samples, n_features)")
    return x

def slice_iter(n, slice_size=None):
    """
    Generate slices for splitting up a batch of 
    size n into smaller slices of size slice_size.
    Yields slice objects.
    """
    if not slice_size or slice_size >= n:
        yield slice(0, n)
        return
    i = 0
    while i < n:
        j = min(n, i + slice_size)
        yield slice(i, j)
        i = j

def norm_scores(raw_attrs):
    """
    Normalize attribution values so scores along the feature axis sum to 1.
    Rescales by dividing individual scores by the total sum along that axis.
    This removes info about overall attribution magnitude, but is useful for 
    visualization, relative importance ranking, or probability-like summaries 
    of feature influence.
    """
    raw_attrs = np.asarray(raw_attrs)
    denom = raw_attrs.sum(axis=-1, keepdims=True)
    denom[denom == 0] = 1.0
    return raw_attrs / denom


# Custom SmoothGrad (non-captum)
def gaussian_noise_torch(x, noise_factor=0.1):
    """
    Add Gaussian noise scaled by (global max - min) as a simple SmoothGrad variant.
    """
    ensure_2d(x)
    scale = (x.max() - x.min()).clamp_min(1e-12)
    return x + torch.randn_like(x) * noise_factor * scale


def saliency_from_scratch(
    model,
    inputs,
    target,
    smoothgrad=False,
    noise_factor=0.1,
    slice_size=None,
    device=None,
):
    """
    Vanilla saliency as ∂logit[target]/∂x.
    
    Processes in mini-batches.
    Optional manual/custom SmoothGrad. 
    returns CPU tensor (n_samples, n_features).
    """
    model.eval()
    ensure_2d(inputs)
    dev = device or next(model.parameters()).device

    out = []
    for sl in slice_iter(inputs.size(0), slice_size):
        x = inputs[sl]
        if smoothgrad:
            x = gaussian_noise_torch(x, noise_factor)
        x = _to_device(x.requires_grad_(True), dev)

        model.zero_grad(set_to_none=True)
        logits = model(x)
        
        if logits.ndim != 2 or target >= logits.size(1):
            raise ValueError("Unexpected logits shape or target out of range")
        
        mask = torch.zeros_like(logits)
        mask[:, target] = 1.0
        grad = torch.autograd.grad(
            logits, x, grad_outputs=mask, create_graph=False
        )[0]
        out.append(grad.detach().cpu())
    return torch.cat(out, dim=0)

# SHAP
def shap_shap(model, inputs, device=None, baselines=None, check_additivity=False):
    """
    SHAP DeepExplainer wrapper.
    Returns CPU tensor (n_samples, n_features).
    """
    try:
        import shap  # lazy import
    except Exception as e:
        raise ImportError("shap is required for shap_shap(); conda install shap") from e

    model.eval()
    ensure_2d(inputs)
    dev = device or next(model.parameters()).device

    x = _to_device(inputs, dev)
    if baselines is None:
        baselines = torch.zeros_like(x[:1])  # single zero baseline
    b = _to_device(baselines, dev)

    explainer = shap.DeepExplainer(model, b)
    sv = explainer.shap_values(x, check_additivity=check_additivity)
    if isinstance(sv, list):
        if len(sv) != 1:
            raise ValueError("Expected single-output model; SHAP returned multiple outputs.")
        sv = sv[0]
    return torch.from_numpy(np.asarray(sv))

# Captum wrappers

def maybe_noise_tunnel(algo, use_tunnel, nt_samples, nt_stdevs):
    if not use_tunnel:
        return algo, {}
    nt = NoiseTunnel(algo)
    return nt, dict(nt_type="smoothgrad_sq", nt_samples=nt_samples, stdevs=nt_stdevs)


def captum_saliency(
    model,
    inputs,
    device=None,
    abs=True,
    target=0,
    smoothgrad=False,
    nt_samples=5,
    nt_stdevs=1.0,
    slice_size=None,
):
    """
    Captum Saliency
    
    Optional SmoothGrad via NoiseTunnel. 
    Returns CPU tensor.
    """
    model.eval()
    ensure_2d(inputs)
    dev = device or next(model.parameters()).device

    algo = Saliency(model)
    algo, nt_kwargs = maybe_noise_tunnel(algo, smoothgrad, nt_samples, nt_stdevs)

    out = []
    for sl in slice_iter(inputs.size(0), slice_size):
        x = _to_device(inputs[sl].requires_grad_(True), dev)
        model.zero_grad(set_to_none=True)
        attr = algo.attribute(x, target=target, abs=abs, **nt_kwargs)
        out.append(attr.detach().cpu())
    return torch.cat(out, dim=0)


def captum_deeplift(
    model,
    inputs,
    device=None,
    mult_by_inputs=True,
    baselines=None,
    target=0,
    smoothgrad=False,
    nt_samples=5,
    nt_stdevs=1.0,
    slice_size=None,
):
    """
    Captum DeepLIFT. 
    
    Optional SmoothGrad via NoiseTunnel. 
    Baseline defaults to a single zero vector. 
    Returns CPU tensor.
    """
    model.eval()
    ensure_2d(inputs)
    dev = device or next(model.parameters()).device

    if baselines is None:
        baselines = torch.zeros_like(inputs[:1])
    b = _to_device(baselines, dev)

    algo = DeepLift(model, multiply_by_inputs=mult_by_inputs)
    algo, nt_kwargs = maybe_noise_tunnel(algo, smoothgrad, 
                                         nt_samples, nt_stdevs)

    out = []
    for sl in slice_iter(inputs.size(0), slice_size):
        x = _to_device(inputs[sl].requires_grad_(True), dev)
        model.zero_grad(set_to_none=True)
        attr = algo.attribute(
            x, target=target, 
            baselines=b, 
            return_convergence_delta=False, 
            **nt_kwargs
        )
        out.append(attr.detach().cpu())
    return torch.cat(out, dim=0)


def captum_gradientshap(
    model,
    inputs,
    device=None,
    mult_by_inputs=True,
    baselines=None,
    target=0,
    smoothgrad=False,
    nt_samples=5,
    nt_stdevs=1.0,
    slice_size=None,
):
    """
    Captum GradientSHAP. Baselines can be multiple samples. 
    Returns CPU tensor.
    """
    model.eval()
    ensure_2d(inputs)
    dev = device or next(model.parameters()).device
    
    if baselines is None:
        baselines = torch.zeros_like(inputs[:8])  # small set by default
    b = _to_device(baselines, dev)

    algo = GradientShap(model, multiply_by_inputs=mult_by_inputs)
    algo, nt_kwargs = maybe_noise_tunnel(algo, smoothgrad, nt_samples, nt_stdevs)

    out = []
    for sl in slice_iter(inputs.size(0), slice_size):
        x = _to_device(inputs[sl].requires_grad_(True), dev)
        model.zero_grad(set_to_none=True)
        attr = algo.attribute(
            x, target=target, 
            baselines=b, 
            return_convergence_delta=False, 
            **nt_kwargs
        )
        out.append(attr.detach().cpu())
    return torch.cat(out, dim=0)


def captum_integratedgradients(
    model,
    inputs,
    device=None,
    mult_by_inputs=True,
    baselines=None,
    target=0,
    smoothgrad=False,
    nt_samples=5,
    nt_stdevs=1.0,
    n_steps=50,
    method="gausslegendre",
    slice_size=None,
    internal_batch_size=None,
):
    """
    Captum Integrated Gradients (+ optional NoiseTunnel). Returns CPU tensor.
    """
    model.eval()
    ensure_2d(inputs)
    dev = device or next(model.parameters()).device

    if baselines is None:
        baselines = torch.zeros_like(inputs[:1])
    b = _to_device(baselines, dev)

    algo = IntegratedGradients(model, multiply_by_inputs=mult_by_inputs)
    algo, nt_kwargs = maybe_noise_tunnel(algo, smoothgrad, nt_samples, nt_stdevs)

    out = []
    for sl in slice_iter(inputs.size(0), slice_size):
        x = _to_device(inputs[sl].requires_grad_(True), dev)
        model.zero_grad(set_to_none=True)
        attr = algo.attribute(
            x,
            target=target,
            baselines=b,
            n_steps=n_steps,
            method=method,
            internal_batch_size=internal_batch_size,
            return_convergence_delta=False,
            **nt_kwargs,
        )
        out.append(attr.detach().cpu())
    return torch.cat(out, dim=0)
