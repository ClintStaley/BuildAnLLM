"""
Python Fundamentals via Projectile Motion  —  File 5 of 5
===========================================================
CONCEPTS COVERED (3-D PyTorch Tensors)
  • A 3-D tensor of shape (S, N, 3): S independent firing ranges,
    N shots each, 3 values per shot (t, x, y)
  • Generating all S×N shots in one vectorised pass
  • Per-set reductions along dim=1: mean, min, max → shape (S, 3)
  • Per-set closest pair using a 4-D broadcast: (S, N, N, 2)
  • Cross-set closest pair: flatten to (S*N, 2) then mask same-set pairs
  • argmin() on a flat index, reconstructing (set, point) from it
  • The connection between (S, N, features) and the LLM training
    shape (batch_size, seq_len, embedding_dim)

Shape convention used throughout:
    dim 0 — S, the set (analogous to batch_size in LLM code)
    dim 1 — N, the point within a set (analogous to seq_len)
    dim 2 — 3, the feature per point: t, x, y (analogous to embedding_dim)
"""

import time
import functools
import torch
from collections import namedtuple


# ── shared infrastructure ─────────────────────────────────────────

G = torch.tensor(9.81)

def timer(func):
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        start  = time.perf_counter()
        result = func(*args, **kwargs)
        ms     = (time.perf_counter() - start) * 1000
        print(f"  [{func.__name__}]  {ms:.3f} ms")
        return result
    return wrapper

LaunchConfig = namedtuple('LaunchConfig', ['az_limit', 'speed_min', 'speed_max'])


# ─────────────────────────────────────────────────────────────────
# SECTION 1  Building the 3-D Tensor
# ─────────────────────────────────────────────────────────────────
# Instead of generating one set of N shots (shape N, 3), we generate
# S independent sets simultaneously.  The random tensors become (S, N)
# instead of (N,), and the physics equations still work element-wise
# across both dimensions at once.

print("=== Section 1: building a (S, N, 3) tensor ===\n")

torch.manual_seed(42)
cfg = LaunchConfig(az_limit=360.0, speed_min=50.0, speed_max=200.0)

S, N = 6, 300    # 6 independent firing ranges, 300 shots each

# All random values in one call, shape (S, N):
az_deg  = torch.rand(S, N) * cfg.az_limit
el_deg  = torch.rand(S, N) * 88.0 + 1.0
speeds  = torch.rand(S, N) * (cfg.speed_max - cfg.speed_min) + cfg.speed_min

print(f"  az_deg.shape = {az_deg.shape}   (one value per set per shot)")

ar = torch.deg2rad(az_deg)    # (S, N)
er = torch.deg2rad(el_deg)    # (S, N)

t = 2.0 * speeds * torch.sin(er) / G          # (S, N)
x = speeds * torch.cos(er) * torch.cos(ar) * t  # (S, N)
y = speeds * torch.cos(er) * torch.sin(ar) * t  # (S, N)

# torch.stack([t, x, y], dim=2) stacks along a NEW third axis:
data = torch.stack([t, x, y], dim=2)   # (S, N, 3)

print(f"  t.shape      = {t.shape}")
print(f"  data.shape   = {data.shape}   ← the 3-D tensor")
print(f"\n  data[0, 0]   = {data[0, 0].tolist()}   (set 0, shot 0: [t, x, y])")
print(f"  data[0, :, 1].shape = {data[0, :, 1].shape}  "
      f"(all x-coords in set 0)")
print(f"\n  Analogy to LLM shapes:")
print(f"    dim 0 = {S}  ↔  batch_size  (independent examples)")
print(f"    dim 1 = {N}  ↔  seq_len     (tokens / points per example)")
print(f"    dim 2 = 3   ↔  embed_dim   (features per token / point)")


# ─────────────────────────────────────────────────────────────────
# SECTION 2  Per-Set Reductions (dim=1)
# ─────────────────────────────────────────────────────────────────
# Reducing along dim=1 collapses the N dimension and gives one
# output per set, so the result has shape (S, 3).
#
# This is the same idea as computing one loss value per batch item
# in a training loop — you collapse the sequence dimension while
# keeping the batch dimension.

print("\n=== Section 2: per-set reductions along dim=1 ===\n")

# Per-set centroid — mean of each column across the N shots:
centroids = data.mean(dim=1)         # (S, 3) — one [t_mean, cx, cy] per set
print(f"  data.shape       = {data.shape}")
print(f"  centroids.shape  = {centroids.shape}   (dim=1 collapsed)")
print(f"\n  centroids (cx, cy) per set:")
for s in range(S):
    print(f"    set {s}: cx={centroids[s, 1].item():7.1f} m  "
          f"cy={centroids[s, 2].item():7.1f} m")

# Per-set bounding box — min and max along dim=1:
x_data   = data[:, :, 1]                             # (S, N) — x only
x_mins   = x_data.min(dim=1).values                  # (S,)
x_maxs   = x_data.max(dim=1).values                  # (S,)
x_ranges = x_maxs - x_mins                           # (S,)

print(f"\n  x-range per set (max x − min x):")
for s in range(S):
    print(f"    set {s}: [{x_mins[s].item():.0f}, {x_maxs[s].item():.0f}]  "
          f"range = {x_ranges[s].item():.0f} m")


# ─────────────────────────────────────────────────────────────────
# SECTION 3  Per-Set Closest Pair (4-D broadcast)
# ─────────────────────────────────────────────────────────────────
# For a single set (N, 2) the pairwise diff tensor was (N, N, 2).
# For S sets (S, N, 2) the diff tensor is (S, N, N, 2) — we add one
# dimension in front and broadcast the same way.
#
# Shape trace:
#   xy              (S, N,  2)
#   xy[:,  :, None, :]   →  (S, N,  1, 2)
#   xy[:, None,  :, :]   →  (S, 1,  N, 2)
#   diff = subtract →  (S, N,  N, 2)   ← broadcast expands the 1s
#   dist_sq = sum(diff²) over dim=3  →  (S, N, N)
#   min over dim=2 then dim=1  →  (S,)  one min-distance per set

print("\n=== Section 3: per-set closest pair (4-D broadcasting) ===\n")

@timer
def per_set_closest(data_sn3: torch.Tensor) -> torch.Tensor:
    """Return (S,) tensor of minimum pairwise distances, one per set.

    Args:
        data_sn3: shape (S, N, 3), column layout [t, x, y]

    Returns:
        min_dists: shape (S,), minimum distance between any two shots
                   in each set.
    """
    S, N, _ = data_sn3.shape
    xy      = data_sn3[:, :, 1:]                    # (S, N, 2)

    # Broadcasting across both the S and N dimensions:
    diff    = xy[:, :, None, :] - xy[:, None, :, :]  # (S, N, N, 2)
    dist_sq = (diff ** 2).sum(dim=3)                  # (S, N, N)

    # Mask the diagonal (same-point distance = 0) with infinity:
    eye_mask = torch.eye(N, dtype=torch.bool).unsqueeze(0)  # (1, N, N)
    dist_sq.masked_fill_(eye_mask, float('inf'))

    # Two successive min reductions:
    per_point_min = dist_sq.min(dim=2).values         # (S, N)
    per_set_min   = per_point_min.min(dim=1).values   # (S,)
    return per_set_min.sqrt()                         # convert sq-dist → dist

min_dists = per_set_closest(data)
print(f"\n  minimum pairwise distance per set:")
for s in range(S):
    print(f"    set {s}: {min_dists[s].item():.1f} m")

# Which set is most "crowded" (smallest minimum distance)?
densest_set = min_dists.argmin().item()
print(f"\n  most crowded set: set {densest_set} "
      f"(min distance = {min_dists[densest_set].item():.1f} m)")


# ─────────────────────────────────────────────────────────────────
# SECTION 4  Cross-Set Closest Pair
# ─────────────────────────────────────────────────────────────────
# Now we want the closest pair where the two points come from
# DIFFERENT sets — e.g. "which two opposing batteries are closest?"
#
# Approach:
#   1. Reshape (S, N, 2) → (S*N, 2) — flatten all sets into one long list.
#   2. Build the full (S*N, S*N) pairwise distance matrix.
#   3. Mask out pairs from the SAME set (we only want cross-set pairs).
#   4. Take argmin to find the winning pair.
#   5. Recover (set, point) indices from the flat index.

print("\n=== Section 4: cross-set closest pair ===\n")

@timer
def cross_set_closest(data_sn3: torch.Tensor):
    """Find the closest pair of points that come from DIFFERENT sets.

    Returns:
        (set_a, point_a, set_b, point_b, distance_m)
    """
    S, N, _ = data_sn3.shape
    xy_flat  = data_sn3[:, :, 1:].reshape(S * N, 2)   # (S*N, 2)

    # Full pairwise distance matrix:
    diff    = xy_flat.unsqueeze(1) - xy_flat.unsqueeze(0)  # (S*N, S*N, 2)
    dist_sq = (diff ** 2).sum(dim=2)                        # (S*N, S*N)

    # Build a same-set mask: set_ids[k] = which set point k belongs to.
    set_ids   = torch.arange(S).repeat_interleave(N)         # (S*N,)
    same_mask = set_ids.unsqueeze(0) == set_ids.unsqueeze(1) # (S*N, S*N)
    dist_sq.masked_fill_(same_mask, float('inf'))

    # Find the minimum over all cross-set pairs:
    flat_idx = dist_sq.argmin()
    ia = flat_idx // (S * N)
    ib = flat_idx %  (S * N)

    set_a,  pt_a  = (ia // N).item(), (ia % N).item()
    set_b,  pt_b  = (ib // N).item(), (ib % N).item()
    dist          = dist_sq[ia, ib].sqrt().item()
    return set_a, pt_a, set_b, pt_b, dist

sa, pa, sb, pb, dist = cross_set_closest(data)
print(f"\n  cross-set closest pair:")
print(f"    set {sa}, shot {pa}  ↔  set {sb}, shot {pb}")
print(f"    distance = {dist:.1f} m")
print(f"\n  coords of the two points:")
print(f"    set {sa}[{pa}]: x={data[sa,pa,1].item():.1f}  y={data[sa,pa,2].item():.1f}")
print(f"    set {sb}[{pb}]: x={data[sb,pb,1].item():.1f}  y={data[sb,pb,2].item():.1f}")


# ─────────────────────────────────────────────────────────────────
# SECTION 5  Wrapping It All in TargetRanges
# ─────────────────────────────────────────────────────────────────
# We can package the whole 3-D workflow into a class with the same
# interface pattern as TorchTargetRange in file 4.

class TargetRanges:
    """Multiple independent firing ranges in a single 3-D tensor.

    Internal tensor shape:  (n_sets, n_points, 3)
      dim 0 — which set (firing range)
      dim 1 — which shot within that set
      dim 2 — [flight_time, x, y]
    """

    def __init__(self, config: LaunchConfig,
                 n_sets: int, n_points: int,
                 el_range: tuple = None, seed: int = None):
        self.config    = config
        self.n_sets    = n_sets
        self.n_points  = n_points
        self.el_range  = el_range if el_range is not None else (1.0, 89.0)
        self.seed      = seed
        self._tensor   = None    # (n_sets, n_points, 3)

    def __repr__(self) -> str:
        shape = tuple(self._tensor.shape) if self._tensor is not None else "—"
        return f"TargetRanges(sets={self.n_sets}, points={self.n_points}, shape={shape})"

    @timer
    def generate(self) -> None:
        """Populate the (n_sets, n_points, 3) tensor in one vectorised pass."""
        if self.seed is not None:
            torch.manual_seed(self.seed)
        S, N = self.n_sets, self.n_points
        el_min, el_max = self.el_range

        az_d  = torch.rand(S, N) * self.config.az_limit
        el_d  = torch.rand(S, N) * (el_max - el_min) + el_min
        spds  = (torch.rand(S, N)
                 * (self.config.speed_max - self.config.speed_min)
                 + self.config.speed_min)

        ar = torch.deg2rad(az_d);  er = torch.deg2rad(el_d)
        tv = 2.0 * spds * torch.sin(er) / G
        xv = spds * torch.cos(er) * torch.cos(ar) * tv
        yv = spds * torch.cos(er) * torch.sin(ar) * tv
        self._tensor = torch.stack([tv, xv, yv], dim=2)

    @timer
    def per_set_centroids(self) -> torch.Tensor:
        """Return (n_sets, 3) tensor — one [mean_t, mean_x, mean_y] per set."""
        return self._tensor.mean(dim=1)

    @timer
    def per_set_min_distances(self) -> torch.Tensor:
        """Return (n_sets,) tensor — minimum pairwise distance per set."""
        return per_set_closest(self._tensor)

    @timer
    def cross_set_closest(self):
        """Return (set_a, pt_a, set_b, pt_b, distance) across different sets."""
        return cross_set_closest(self._tensor)


print("\n=== Section 5: TargetRanges class ===\n")

ranges = TargetRanges(cfg, n_sets=6, n_points=300, seed=42)
print(f"  {ranges!r}")
print()
ranges.generate()
print(f"\n  {ranges!r}")

cents = ranges.per_set_centroids()
print(f"\n  per_set_centroids().shape = {cents.shape}")
print(f"  set 0 centroid: cx={cents[0,1].item():.1f}  cy={cents[0,2].item():.1f}")

mdists = ranges.per_set_min_distances()
print(f"\n  per_set_min_distances(): {[round(d,1) for d in mdists.tolist()]}")

sa, pa, sb, pb, d = ranges.cross_set_closest()
print(f"\n  cross-set closest: set {sa}[{pa}] ↔ set {sb}[{pb}]  dist={d:.1f} m")

print("\n[v5 complete]")
print()
print("── End of series ──────────────────────────────────────────")
print("  v1  Python scalars, lists, slicing, copy/alias")
print("  v2  Functions, decorator, namedtuple, input parsing,")
print("      enumerate/zip, stats, closest-pair, None, exceptions")
print("  v3  Class design, dunder methods, property, inheritance")
print("  v4  PyTorch: tensors, vectorised physics, timing")
print("  v5  PyTorch: 3-D tensors, per-set and cross-set ops")
