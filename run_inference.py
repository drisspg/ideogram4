"""Minimal example: text-to-image inference with Ideogram 4."""

from __future__ import annotations

import argparse
import os
import sys

import torch

from ideogram4 import (
  DEFAULT_MAGIC_PROMPT,
  MAGIC_PROMPTS,
  PRESETS,
  Ideogram4Pipeline,
  Ideogram4PipelineConfig,
  aspect_ratio_from_size,
  moderate_image,
  moderate_prompt,
)

QUANTIZATION_REPOS = {
  "nf4": "ideogram-ai/ideogram-4-nf4",
  "fp8": "ideogram-ai/ideogram-4-fp8",
}


def _default_device() -> str:
  """Pick the best available device: cuda, then mps (Apple Silicon), then cpu."""
  if torch.cuda.is_available():
    return "cuda"
  if torch.backends.mps.is_available():
    return "mps"
  return "cpu"


def _default_quantization() -> str:
  """nf4 (bitsandbytes 4-bit) is CUDA-only; fall back to fp8 on mps/cpu."""
  return "nf4" if torch.cuda.is_available() else "fp8"


def _print_flags(label: str, flags: list[tuple[str, float]]) -> None:
  print(f"{label}:", file=sys.stderr)
  for name, score in sorted(flags, key=lambda x: -x[1]):
    print(f"  {name}: {score:.3f}", file=sys.stderr)


def main() -> None:
  parser = argparse.ArgumentParser()
  parser.add_argument("--prompt", required=True)
  parser.add_argument("--output", default="output.png")
  parser.add_argument("--height", type=int, default=1024)
  parser.add_argument("--width", type=int, default=1024)
  parser.add_argument(
    "--sampler-preset",
    choices=sorted(PRESETS.keys()),
    default="V4_QUALITY_48",
    help=(
      "Named sampler preset that bundles num_steps, per-step CFG schedule, "
      "mu, and std. To use a different configuration, add a new entry to "
      "ideogram4.sampler_configs.PRESETS. Available: "
      + ", ".join(sorted(PRESETS.keys()))
    ),
  )
  parser.add_argument("--seed", type=int, default=0)
  parser.add_argument("--device", default=_default_device())
  parser.add_argument(
    "--compile-model",
    action="store_true",
    help="Compile repeated inference modules with torch.compile; first generation pays compilation cost.",
  )
  parser.add_argument(
    "--torchao-linear",
    choices=(
      "bf16",
      "fp8",
      "fp8_row",
      "fp8_row_mlp",
      "mxfp8",
      "mxfp8_floor",
      "mxfp8_even",
      "mxfp8_ceil",
      "mxfp8_mlp",
      "nvfp4",
      "nvfp4_mlp",
      "nvfp4_mlp_up",
      "nvfp4_mlp_down",
    ),
    help=(
      "Materialize fp8 weights to regular Linear modules, then optionally "
      "convert them to a torchao/PyTorch scaled_mm Linear engine. Use bf16 for "
      "a materialized full-bf16 baseline, fp8 for per-tensor FP8, fp8_row for "
      "per-row FP8, fp8_row_mlp for per-row FP8 transformer MLPs only, mxfp8 "
      "for MXFP8 block-32, or nvfp4 for NVFP4 block-16. "
      "Requires --quantization fp8."
    ),
  )
  parser.add_argument(
    "--quantization",
    choices=sorted(QUANTIZATION_REPOS.keys()),
    default=_default_quantization(),
    help=(
      "Weight precision. 'nf4' loads bitsandbytes 4-bit pre-quantized weights "
      f"from {QUANTIZATION_REPOS['nf4']} (CUDA only). 'fp8' loads weight-only "
      f"e4m3 float8 transformer weights from {QUANTIZATION_REPOS['fp8']}; "
      "activations stay bf16, so it runs on any device (no FP8 hardware needed). "
      "Defaults to nf4 when CUDA is available, otherwise fp8."
    ),
  )
  parser.add_argument(
    "--magic-prompt",
    action=argparse.BooleanOptionalAction,
    default=True,
    help=(
      "Expand the plain --prompt into the structured JSON caption the model "
      "was trained on, using a magic-prompt LLM (ON by default; pass "
      "--no-magic-prompt to feed --prompt to the model verbatim, e.g. when you "
      "already have a structured caption)."
    ),
  )
  parser.add_argument(
    "--magic-prompt-model",
    choices=sorted(MAGIC_PROMPTS),
    default=DEFAULT_MAGIC_PROMPT,
    help=(
      "Which magic-prompt configuration to use (a model + system-prompt "
      f"version). Default: {DEFAULT_MAGIC_PROMPT}."
    ),
  )
  parser.add_argument(
    "--magic-prompt-key",
    default=os.environ.get("MAGIC_PROMPT_API_KEY")
    or os.environ.get("IDEOGRAM_API_KEY"),
    help=(
      "API key for the magic-prompt model (or set MAGIC_PROMPT_API_KEY). "
      "Required unless --no-magic-prompt. The claude-* configurations call "
      "OpenRouter (get a key at https://openrouter.ai); the ideogram-4-v1 "
      "configuration calls Ideogram's free hosted magic-prompt API and reads "
      "IDEOGRAM_API_KEY by default."
    ),
  )
  parser.add_argument(
    "--warn-on-caption-issues",
    action="store_true",
    help=(
      "If set, caption verifier issues are emitted as warnings instead of "
      "aborting with an error."
    ),
  )
  parser.add_argument(
    "--hive-text-key",
    default=os.environ.get("HIVE_TEXT_MODERATION_KEY"),
    help=(
      "Hive Text Moderation API key (or set HIVE_TEXT_MODERATION_KEY). "
      "Screens the prompt for NSFW / hate / violence / self-harm / bullying "
      "before generation. We strongly discourage running this inference code "
      "without a Hive key configured: doing so disables prompt safety "
      "screening entirely. Sign up at https://thehive.ai and create a Text "
      "Moderation project for the key."
    ),
  )
  parser.add_argument(
    "--hive-visual-key",
    default=os.environ.get("HIVE_VISUAL_MODERATION_KEY"),
    help=(
      "Hive Visual Content Moderation API key (or set "
      "HIVE_VISUAL_MODERATION_KEY). Screens the generated image for NSFW / "
      "gore / drugs / weapons / hate symbols. We strongly discourage running "
      "this inference code without a Hive key configured: doing so disables "
      "output safety screening entirely. Sign up at https://thehive.ai and "
      "create a Visual Content Moderation project."
    ),
  )
  args = parser.parse_args()

  if args.torchao_linear and args.quantization == "nf4":
    print(
      "ERROR: --torchao-linear cannot consume the bnb NF4 checkpoint directly. "
      "Use --quantization fp8 to materialize regular weights before converting "
      "to a PyTorch scaled_mm low-precision engine.",
      file=sys.stderr,
    )
    sys.exit(2)

  if args.hive_text_key:
    flags = moderate_prompt(args.prompt, args.hive_text_key)
    if flags:
      _print_flags("Prompt rejected by Hive text moderation", flags)
      sys.exit(2)
  else:
    print(
      "WARNING: no Hive text moderation key configured -- prompt safety "
      "screening is DISABLED. We strongly discourage using this inference code "
      "without a Hive key. Set HIVE_TEXT_MODERATION_KEY or pass "
      "--hive-text-key to enable.",
      file=sys.stderr,
    )

  prompt = args.prompt
  if args.magic_prompt:
    if not args.magic_prompt_key:
      print(
        "ERROR: magic prompt is enabled but no API key was found. Set "
        "MAGIC_PROMPT_API_KEY, pass --magic-prompt-key, or disable expansion "
        "with --no-magic-prompt.",
        file=sys.stderr,
      )
      sys.exit(2)
    aspect_ratio = aspect_ratio_from_size(args.width, args.height)
    magic = MAGIC_PROMPTS[args.magic_prompt_model](api_key=args.magic_prompt_key)  # type: ignore[call-arg]
    print(
      f"Magic prompt ({args.magic_prompt_model}): "
      f"expanding prompt for {aspect_ratio}...",
      file=sys.stderr,
    )
    prompt = magic.expand(args.prompt, aspect_ratio=aspect_ratio)
    print(f"Expanded caption:\n{prompt}", file=sys.stderr)

  preset = PRESETS[args.sampler_preset]

  pipe = Ideogram4Pipeline.from_pretrained(
    config=Ideogram4PipelineConfig(
      weights_repo=QUANTIZATION_REPOS[args.quantization],
      compile_model=args.compile_model,
      torchao_quantization=args.torchao_linear,
    ),
    device=args.device,
    dtype=torch.bfloat16,
  )
  images = pipe(
    prompt,
    height=args.height,
    width=args.width,
    num_steps=preset.num_steps,
    guidance_schedule=preset.guidance_schedule,
    mu=preset.mu,
    std=preset.std,
    seed=args.seed,
    raise_on_caption_issues=not args.warn_on_caption_issues,
  )

  if args.hive_visual_key:
    flags = moderate_image(images[0], args.hive_visual_key)
    if flags:
      _print_flags("Generated image rejected by Hive visual moderation", flags)
      sys.exit(2)
  else:
    print(
      "WARNING: no Hive visual moderation key configured -- output safety "
      "screening is DISABLED. We strongly discourage using this inference code "
      "without a Hive key. Set HIVE_VISUAL_MODERATION_KEY or pass "
      "--hive-visual-key to enable.",
      file=sys.stderr,
    )

  images[0].save(args.output)
  print(f"Saved {args.output}")


if __name__ == "__main__":
  main()
