"""
LLM Class Exercise: GPU timing and softmax winner-take-all concentration.

Usage:
    python topk_explore.py D [--gpu]

Example:
    python topk_explore.py 128
    python topk_explore.py 256 --gpu
"""

import argparse
import time
import torch
import torch.nn.functional as F


# ── Memory warnings ────────────────────────────────────────────────────────────
CPU_SAFE_D = 256
GPU_SAFE_D = 512


def warn_if_large(D: int, use_gpu: bool) -> None:
    limit = GPU_SAFE_D if use_gpu else CPU_SAFE_D
    if D > limit:
        # Each of T2d, T3d, scores, probs is float32.
        # Rough total: D^2 * 4 bytes  +  3 * D^3 * 4 bytes
        bytes_est = 4 * (D ** 2 + 3 * D ** 3)
        gb_est = bytes_est / 1024 ** 3
        device_str = "GPU" if use_gpu else "CPU"
        print(
            f"WARNING: D={D} exceeds the recommended limit of {limit} for {device_str}.\n"
            f"  Estimated tensor memory: {gb_est:.2f} GB. "
            "You may hit OOM errors.\n"
        )


# ── Timing helpers ─────────────────────────────────────────────────────────────
def sync(device: torch.device) -> None:
    """Barrier so CUDA work is complete before we read the clock."""
    if device.type == "cuda":
        torch.cuda.synchronize(device)


def timed_op(fn, device: torch.device):
    """
    Run fn(), bracketed by synchronize() calls on each side.
    Returns (result, elapsed_ms).
    """
    sync(device)
    t0 = time.perf_counter()
    result = fn()
    sync(device)
    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return result, elapsed_ms


# ── Main ───────────────────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Explore GPU timing and softmax concentration with large random tensors."
    )
    parser.add_argument(
        "D",
        type=int,
        help="Dimension D. Creates tensors of shape (D, D) and (D, D, D).",
    )
    parser.add_argument(
        "--gpu",
        action="store_true",
        help="Run on GPU (CUDA). Falls back to CPU with a warning if CUDA is unavailable.",
    )
    args = parser.parse_args()

    D = args.D
    if D <= 0:
        parser.error("D must be a positive integer.")

    # ── Device selection ──────────────────────────────────────────────────────
    if args.gpu:
        if not torch.cuda.is_available():
            print("WARNING: --gpu requested but CUDA is not available. Falling back to CPU.\n")
            device = torch.device("cpu")
        else:
            device = torch.device("cuda")
    else:
        device = torch.device("cpu")

    print(f"Device : {device}")
    print(f"D      : {D}")
    print()

    warn_if_large(D, device.type == "cuda")

    # ── Tensor creation ───────────────────────────────────────────────────────
    # Normal(mean=0, variance=4) → std=2, so multiply randn (std=1) by 2.
    T2d = torch.randn(D, D, device=device) * 2          # shape: (D, D)
    T3d = torch.randn(D, D, D, device=device) * 2       # shape: (D, D, D)

    # ── Warmup pass (results discarded) ───────────────────────────────────────
    # This lets CUDA JIT-compile kernels and avoids counting one-time
    # overhead in the timed measurements.
    print("Running warmup pass...")
    _ = torch.matmul(T3d, T2d)
    sync(device)
    _ = F.softmax(torch.matmul(T3d, T2d), dim=-1)
    sync(device)
    print("Warmup complete.\n")

    # ── Timed: batched matrix multiplication ──────────────────────────────────
    # T3d is treated as a batch of D matrices each of shape (D, D).
    # torch.matmul broadcasts: (D, D, D) @ (D, D) → (D, D, D).
    scores, ms_matmul = timed_op(lambda: torch.matmul(T3d, T2d), device)

    # ── Timed: softmax over final dimension ───────────────────────────────────
    probs, ms_softmax = timed_op(lambda: F.softmax(scores, dim=-1), device)

    # ── Report timing ─────────────────────────────────────────────────────────
    print(f"Batched matmul  (D×D×D) @ (D×D) → (D×D×D) : {ms_matmul:10.3f} ms")
    print(f"Softmax over final dim               : {ms_softmax:10.3f} ms")
    print()

    # ── Report softmax concentration ──────────────────────────────────────────
    # probs has shape (D, D, D). The last dimension is the distribution.
    # For each of the D*D distributions, find the maximum probability.
    # Then average those maxima.
    max_probs = probs.max(dim=-1).values          # shape: (D, D)
    mean_max = max_probs.mean().item()

    print(f"Softmax distributions   : {D * D:,}  (one per row of the D×D batch)")
    print(f"Distribution dimension  : {D}  (vocabulary size analogue)")
    print(f"Mean maximum probability: {mean_max:.6f}")
    print()
    print(
        "As D grows, the dot products inside 'scores' grow in magnitude\n"
        "(their std scales as sqrt(D)), so one logit tends to dominate\n"
        "the softmax exponential, concentrating probability on a single\n"
        "winner — the same effect seen in LLM token distributions at\n"
        "low temperature or high-dimensional embeddings."
    )


if __name__ == "__main__":
    main()
