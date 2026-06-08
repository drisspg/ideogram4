from __future__ import annotations

import os
import warnings

import bitsandbytes as bnb
import torch
import torch.nn as nn
import torch.nn.functional as F


_BNB_SIBLING_SUFFIXES = (
  ".absmax",
  ".quant_map",
  ".nested_absmax",
  ".nested_quant_map",
)

# Largest magnitude representable by the e4m3 float8 format. Per-row weight
# scales map each row's max abs value onto this so we use the full range.
FP8_E4M3_MAX = 448.0
FP8_WEIGHT_DTYPE = torch.float8_e4m3fn
FP8_SCALE_SUFFIX = ".weight_scale"
# Marker written into the text encoder's config.json so the loader knows to take
# the custom weight-only FP8 path instead of transformers' from_pretrained.
FP8_TEXT_ENCODER_CONFIG_FLAG = "ideogram_fp8_weight_only"
_TORCHAO_BLOCK_SIZES = {
  "nvfp4": 16,
  "mxfp8": 32,
  "mxfp8_floor": 32,
  "mxfp8_even": 32,
  "mxfp8_ceil": 32,
  "mxfp8_mlp_padded": 32,
  "fp8": None,
  "fp8_row": None,
}


def is_bnb4bit_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  """True if any key looks like a bnb 4-bit quant_state sibling."""
  return any(".quant_state.bitsandbytes__" in k for k in state_dict)


def swap_linears_to_bnb4bit(
  module: nn.Module,
  compute_dtype: torch.dtype,
  *,
  quant_type: str = "nf4",
  compress_statistics: bool = False,
) -> None:
  for name, child in list(module.named_children()):
    if isinstance(child, nn.Linear):
      new_linear = bnb.nn.Linear4bit(
        child.in_features,
        child.out_features,
        bias=child.bias is not None,
        compute_dtype=compute_dtype,
        compress_statistics=compress_statistics,
        quant_type=quant_type,
      )
      setattr(module, name, new_linear)
    else:
      swap_linears_to_bnb4bit(
        child,
        compute_dtype,
        quant_type=quant_type,
        compress_statistics=compress_statistics,
      )


def load_bnb4bit_state_dict(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
) -> None:
  """Load a bnb 4-bit checkpoint by assigning prepared tensors into the model."""
  consumed: set[str] = set()
  for full_name, tensor in state_dict.items():
    if ".quant_state." in full_name or full_name.endswith(_BNB_SIBLING_SUFFIXES):
      continue
    parent_path, _, param_name = full_name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    current = parent._parameters.get(param_name)
    if not isinstance(current, bnb.nn.Params4bit):
      continue
    prefix = full_name + "."
    quantized_stats = {
      name: stat for name, stat in state_dict.items() if name.startswith(prefix)
    }
    # bnb's from_prequantized may mutate the stats dict, so snapshot names first.
    consumed.add(full_name)
    consumed.update(quantized_stats.keys())
    parent._parameters[param_name] = bnb.nn.Params4bit.from_prequantized(
      data=tensor,
      quantized_stats=quantized_stats,
      requires_grad=False,
      device=device,
    )

  prepared_remaining = {}
  for name, tensor in state_dict.items():
    if name in consumed:
      continue
    if tensor.is_floating_point():
      prepared_remaining[name] = tensor.to(device=device, dtype=dtype)
    else:
      prepared_remaining[name] = tensor.to(device=device)

  missing, unexpected = model.load_state_dict(
    prepared_remaining, strict=False, assign=True
  )
  # Quantized weights are loaded via from_prequantized above, so they appear in
  # `missing` from load_state_dict's perspective — filter those out.
  real_missing = [m for m in missing if m not in consumed]
  if real_missing:
    raise RuntimeError(f"missing keys after quantized load: {real_missing[:10]}")
  if unexpected:
    raise RuntimeError(f"unexpected keys after quantized load: {unexpected[:10]}")

  for name, buffer in list(model.named_buffers()):
    parent_path, _, leaf = name.rpartition(".")
    parent = model.get_submodule(parent_path) if parent_path else model
    persistent = leaf not in parent._non_persistent_buffers_set
    if persistent and buffer.is_floating_point():
      moved = buffer.to(device=device, dtype=dtype)
    else:
      moved = buffer.to(device=device)
    if moved is not buffer:
      parent.register_buffer(leaf, moved, persistent=persistent)


# ---------------------------------------------------------------------------
# Weight-only FP8 (e4m3)
#
# Activations stay in the compute dtype (e.g. bfloat16); only Linear weights are
# stored as float8 with a per-output-channel (per-row) float32 scale. At forward
# time the weight is dequantized back to the compute dtype and a normal bf16
# matmul runs, so this needs no FP8 tensor-core hardware and works on any device
# that can store float8 (CPU included). The win is ~2x smaller Linear weights.
# ---------------------------------------------------------------------------


def quantize_weight_to_fp8(
  weight: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
  """Quantize a 2-D Linear weight to e4m3 float8 with per-row scales.

  Returns ``(weight_fp8, scale)`` where ``weight_fp8`` has shape ``(out, in)``
  in ``float8_e4m3fn`` and ``scale`` has shape ``(out,)`` in float32 such that
  ``weight ≈ weight_fp8.to(dtype) * scale[:, None]``.
  """
  w = weight.detach().to(torch.float32)
  amax = w.abs().amax(dim=1, keepdim=True).clamp(min=1e-12)
  scale = amax / FP8_E4M3_MAX
  q = (w / scale).clamp(-FP8_E4M3_MAX, FP8_E4M3_MAX).to(FP8_WEIGHT_DTYPE)
  return q, scale.squeeze(1).to(torch.float32)


def is_fp8_state_dict(state_dict: dict[str, torch.Tensor]) -> bool:
  """True if the checkpoint carries weight-only FP8 Linear weights."""
  return any(k.endswith(FP8_SCALE_SUFFIX) for k in state_dict) or any(
    v.dtype == FP8_WEIGHT_DTYPE for v in state_dict.values()
  )


def dequantize_fp8_state_dict(
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
) -> dict[str, torch.Tensor]:
  """Materialize a weight-only FP8 checkpoint as regular floating tensors."""
  prepared: dict[str, torch.Tensor] = {}
  for key, tensor in state_dict.items():
    if key.endswith(FP8_SCALE_SUFFIX):
      continue
    scale = state_dict.get(f"{key}_scale")
    if tensor.dtype == FP8_WEIGHT_DTYPE and scale is not None:
      prepared[key] = tensor.to(device=device, dtype=dtype) * scale.to(
        device=device, dtype=dtype
      ).unsqueeze(1)
    elif tensor.is_floating_point():
      prepared[key] = tensor.to(device=device, dtype=dtype)
    else:
      prepared[key] = tensor.to(device=device)
  return prepared


class Fp8Linear(nn.Module):
  """Linear layer holding an e4m3 float8 weight + per-row float32 scale.

  The weight and scale are registered as buffers (not parameters) so they load
  via ``load_state_dict`` and are excluded from optimizer/grad machinery. The
  dequantized matmul runs in ``compute_dtype``.
  """

  weight: torch.Tensor
  weight_scale: torch.Tensor
  bias: torch.Tensor | None

  def __init__(
    self,
    in_features: int,
    out_features: int,
    bias: bool,
    compute_dtype: torch.dtype,
  ) -> None:
    super().__init__()
    self.in_features = in_features
    self.out_features = out_features
    self.compute_dtype = compute_dtype
    self.register_buffer(
      "weight",
      torch.empty(out_features, in_features, dtype=FP8_WEIGHT_DTYPE),
    )
    self.register_buffer("weight_scale", torch.empty(out_features, dtype=torch.float32))
    if bias:
      self.register_buffer("bias", torch.empty(out_features, dtype=compute_dtype))
    else:
      self.bias = None

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    w = self.weight.to(x.dtype) * self.weight_scale.to(x.dtype).unsqueeze(1)
    bias = self.bias.to(x.dtype) if self.bias is not None else None
    return F.linear(x, w, bias)


def swap_linears_to_fp8(
  module: nn.Module,
  state_dict: dict[str, torch.Tensor],
  compute_dtype: torch.dtype,
  *,
  prefix: str = "",
) -> None:
  """Replace each ``nn.Linear`` that has a saved FP8 scale with an ``Fp8Linear``.

  Gating on the presence of ``<name>.weight_scale`` means only layers that were
  actually quantized at save time are swapped; everything else loads normally in
  the compute dtype.
  """
  for name, child in list(module.named_children()):
    child_prefix = f"{prefix}{name}"
    if (
      isinstance(child, nn.Linear) and f"{child_prefix}{FP8_SCALE_SUFFIX}" in state_dict
    ):
      setattr(
        module,
        name,
        Fp8Linear(
          child.in_features,
          child.out_features,
          bias=child.bias is not None,
          compute_dtype=compute_dtype,
        ),
      )
    else:
      swap_linears_to_fp8(child, state_dict, compute_dtype, prefix=f"{child_prefix}.")


def load_fp8_state_dict(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
  *,
  assign: bool = False,
  strict: bool = True,
) -> None:
  """Load a weight-only FP8 checkpoint into ``model``.

  ``model`` must already have its FP8 Linear layers swapped in (see
  ``swap_linears_to_fp8``). FP8 weights are kept as float8, scales stay float32,
  and every other floating tensor is cast to ``dtype``.

  ``assign=True`` replaces the module's tensors with the prepared ones rather than
  copying into them. Use it when the model was built with ``from_config`` so the
  non-quantized params take the loaded dtype directly and computed non-persistent
  buffers (e.g. rotary caches) are left untouched. With ``assign=False`` (default),
  the caller must have already put the unquantized params in ``dtype``.

  ``strict=False`` downgrades missing keys to a warning (e.g. tied weights that a
  ``transformers`` model resolves itself); unexpected keys always raise.
  """
  prepared: dict[str, torch.Tensor] = {}
  for k, v in state_dict.items():
    if v.dtype == FP8_WEIGHT_DTYPE:
      prepared[k] = v.to(device=device)
    elif k.endswith(FP8_SCALE_SUFFIX):
      prepared[k] = v.to(device=device, dtype=torch.float32)
    elif v.is_floating_point():
      prepared[k] = v.to(device=device, dtype=dtype)
    else:
      prepared[k] = v.to(device=device)

  missing, unexpected = model.load_state_dict(prepared, strict=False, assign=assign)
  if unexpected:
    raise RuntimeError(f"unexpected keys after fp8 load: {unexpected[:10]}")
  if missing:
    if strict:
      raise RuntimeError(f"missing keys after fp8 load: {missing[:10]}")
    warnings.warn(f"missing keys after fp8 load: {missing[:10]}", stacklevel=2)

  model.to(device)


def load_fp8_state_dict_as_bf16(
  model: nn.Module,
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
  *,
  assign: bool = True,
  strict: bool = True,
) -> None:
  """Load a weight-only FP8 checkpoint into ordinary Linear modules."""
  missing, unexpected = model.load_state_dict(
    dequantize_fp8_state_dict(state_dict, device, dtype), strict=False, assign=assign
  )
  if unexpected:
    raise RuntimeError(f"unexpected keys after fp8 dequant load: {unexpected[:10]}")
  if missing:
    if strict:
      raise RuntimeError(f"missing keys after fp8 dequant load: {missing[:10]}")
    warnings.warn(f"missing keys after fp8 dequant load: {missing[:10]}", stacklevel=2)
  model.to(device)


class OutputPaddedLinear(nn.Module):
  def __init__(self, linear: nn.Linear, padded_out_features: int) -> None:
    super().__init__()
    self.out_features = linear.out_features
    self.linear = nn.Linear(
      linear.in_features,
      padded_out_features,
      bias=linear.bias is not None,
      device=linear.weight.device,
      dtype=linear.weight.dtype,
    )
    with torch.no_grad():
      self.linear.weight.zero_()
      self.linear.weight[: linear.out_features].copy_(linear.weight)
      if linear.bias is not None:
        self.linear.bias.zero_()
        self.linear.bias[: linear.out_features].copy_(linear.bias)

  def forward(self, x: torch.Tensor) -> torch.Tensor:
    return self.linear(x)[..., : self.out_features]


def _base_torchao_quantization(quantization: str) -> str:
  if quantization == "mxfp8_mlp_padded":
    return "mxfp8"
  for suffix in ("_mlp_up", "_mlp_down", "_mlp"):
    if quantization.endswith(suffix):
      return quantization.removesuffix(suffix)
  return quantization


def _is_mlp_padded_target(fqn: str, module: nn.Linear) -> bool:
  return (
    (".feed_forward.w1" in fqn or ".feed_forward.w3" in fqn)
    and module.out_features == 12288
  )


def _pad_mxfp8_mlp_linears(module: nn.Module) -> int:
  converted = 0
  for fqn, child in list(module.named_modules()):
    if not isinstance(child, nn.Linear) or not _is_mlp_padded_target(fqn, child):
      continue
    parent_path, _, child_name = fqn.rpartition(".")
    parent = module.get_submodule(parent_path) if parent_path else module
    setattr(parent, child_name, OutputPaddedLinear(child, 14336))
    converted += 1
  return converted


def _torchao_linear_filter(
  module: nn.Module, fqn: str, block_size: int | None, quantization: str
) -> bool:
  if not isinstance(module, nn.Linear):
    return False
  if quantization in ("mxfp8_mlp_padded",) and ".feed_forward." not in fqn:
    return False
  if quantization.endswith("_mlp") and ".feed_forward." not in fqn:
    return False
  if quantization.endswith("_mlp_up") and not (
    ".feed_forward.w1" in fqn or ".feed_forward.w3" in fqn
  ):
    return False
  if quantization.endswith("_mlp_down") and ".feed_forward.w2" not in fqn:
    return False
  if any(filtered in fqn for filtered in ("embedder", "embed", "embedding")):
    return False
  out_features, in_features = module.weight.shape
  if block_size is not None and in_features % block_size != 0:
    return False
  if block_size == 16 and out_features % 16 != 0:
    return False
  if out_features <= 64:
    return False
  return not (in_features <= 1024 and out_features <= 1024)


def _torchao_nvfp4_config(use_triton_kernel: bool):
  if use_triton_kernel:
    os.environ.setdefault("TRITON_ALLOW_NON_CONSTEXPR_GLOBALS", "1")
  try:
    from torchao.prototype.mx_formats.inference_workflow import (  # type: ignore[import-not-found]
      NVFP4DynamicActivationNVFP4WeightConfig,
    )
  except ImportError:
    from torchao.prototype.mx_formats.mx_subclass import (  # type: ignore[import-not-found,no-redef]
      NVFP4InferenceConfig as NVFP4DynamicActivationNVFP4WeightConfig,
    )
  return NVFP4DynamicActivationNVFP4WeightConfig(
    use_triton_kernel=use_triton_kernel,
    use_dynamic_per_tensor_scale=True,
  )


def _torchao_config(quantization: str, use_triton_kernel: bool):
  base_quantization = _base_torchao_quantization(quantization)
  match base_quantization:
    case "nvfp4":
      return _torchao_nvfp4_config(use_triton_kernel)
    case "mxfp8" | "mxfp8_floor" | "mxfp8_even" | "mxfp8_ceil":
      from torchao.prototype.mx_formats import (  # type: ignore[import-not-found]
        MXDynamicActivationMXWeightConfig,
      )
      from torchao.prototype.mx_formats.mx_tensor import (  # type: ignore[import-not-found]
        ScaleCalculationMode,
      )

      scaling_modes = {
        "mxfp8": ScaleCalculationMode.RCEIL,
        "mxfp8_floor": ScaleCalculationMode.FLOOR,
        "mxfp8_even": ScaleCalculationMode.EVEN,
        "mxfp8_ceil": ScaleCalculationMode.CEIL,
      }
      return MXDynamicActivationMXWeightConfig(
        block_size=32,
        activation_dtype=torch.float8_e4m3fn,
        weight_dtype=torch.float8_e4m3fn,
        scaling_mode=scaling_modes[base_quantization],
      )
    case "fp8":
      from torchao.quantization import (  # type: ignore[import-not-found]
        Float8DynamicActivationFloat8WeightConfig,
        PerTensor,
      )

      return Float8DynamicActivationFloat8WeightConfig(granularity=PerTensor())
    case "fp8_row":
      from torchao.quantization import (  # type: ignore[import-not-found]
        Float8DynamicActivationFloat8WeightConfig,
        PerRow,
      )
      from torchao.quantization.quantize_.common.kernel_preference import (  # type: ignore[import-not-found]
        KernelPreference,
      )

      return Float8DynamicActivationFloat8WeightConfig(
        granularity=PerRow(), kernel_preference=KernelPreference.TORCH
      )
    case _:
      raise ValueError(f"unsupported torchao quantization: {quantization}")


def apply_torchao_quantization(
  module: nn.Module,
  quantization: str,
  *,
  use_triton_kernel: bool = True,
) -> int:
  """Apply optional torchao Linear quantization and return converted layer count."""
  if quantization == "mxfp8_mlp_padded":
    _pad_mxfp8_mlp_linears(module)
  base_quantization = _base_torchao_quantization(quantization)
  if base_quantization not in _TORCHAO_BLOCK_SIZES:
    raise ValueError(f"unsupported torchao quantization: {quantization}")
  block_size = _TORCHAO_BLOCK_SIZES[base_quantization]
  try:
    from torchao.quantization import quantize_  # type: ignore[import-not-found]
  except ImportError as error:
    raise RuntimeError(
      f"torchao {quantization} requested but torchao is not installed in this environment"
    ) from error

  def filter_fn(child: nn.Module, fqn: str) -> bool:
    return _torchao_linear_filter(child, fqn, block_size, quantization)

  converted = sum(1 for fqn, child in module.named_modules() if filter_fn(child, fqn))
  if converted == 0:
    raise RuntimeError(
      f"torchao {quantization} found no eligible nn.Linear layers; BNB NF4 and "
      "custom FP8 Linear modules must be materialized back to ordinary Linear weights first"
    )
  quantize_(module, config=_torchao_config(quantization, use_triton_kernel), filter_fn=filter_fn)
  return converted
