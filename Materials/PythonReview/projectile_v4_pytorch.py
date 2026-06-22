"""
Python Fundamentals via Projectile Motion  —  File 4 of 5
===========================================================
CONCEPTS COVERED (PyTorch)
  • torch.rand, torch.deg2rad, element-wise tensor operations
  • torch.stack to assemble a 2-D (N, 3) tensor from column vectors
  • Tensor column selection with  tensor[:, col]
  • Reduction methods: .min(), .max(), .mean(), .argmin()
  • .item() to extract a Python scalar from a 0-dim tensor
  • Pairwise distance matrix via broadcasting (no Python loop)
  • fill_diagonal_() for in-place masking
  • .clone() as the tensor equivalent of deepcopy
  • Aliasing with tensors — same gotcha as with lists
  • Timing comparison: Python nested loops vs vectorised PyTorch

This file re-implements the TargetRange class from file 3 using
PyTorch tensors instead of Python lists.  The class interface is
identical — generate(), stats(), closest_pair() — so side-by-side
timing comparisons are straightforward.
"""

import math
import random
import time
import functools
import torch
from collections import namedtuple


# ── shared infrastructure ─────────────────────────────────────────

G = torch.tensor(9.81)          # gravity as a PyTorch scalar tensor

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
LandingStats = namedtuple('LandingStats', ['x_min', 'x_max',
                                           'y_min', 'y_max',
                                           'centroid_x', 'centroid_y'])

# Python-loop version (from file 3) kept for timing comparison:
def _py_generate(config, n, seed=42):
    random.seed(seed)
    def _one(az, el, spd):
        ar = math.radians(az); er = math.radians(el)
        t = 2*spd*math.sin(er)/9.81
        return [t, spd*math.cos(er)*math.cos(ar)*t, spd*math.cos(er)*math.sin(ar)*t]
    return [_one(random.uniform(0,config.az_limit), random.uniform(1,89),
                 random.uniform(config.speed_min,config.speed_max)) for _ in range(n)]

def _py_closest(shots):
    best=float('inf'); bi=bj=-1
    for i in range(len(shots)):
        for j in range(i+1,len(shots)):
            d=math.hypot(shots[i][1]-shots[j][1],shots[i][2]-shots[j][2])
            if d<best: best=d; bi,bj=i,j
    return bi,bj,best


# ─────────────────────────────────────────────────────────────────
# SECTION 1  Torch Warmup — Tensors and Element-wise Operations
# ─────────────────────────────────────────────────────────────────
# Before porting the whole class, let's see the key torch idioms
# that replace Python loops in the physics calculation.

print("=== Section 1: torch tensor basics ===\n")

torch.manual_seed(42)
N = 8   # tiny, so output is readable

# Generate all random values at once — one tensor operation each:
az_deg  = torch.rand(N) * 360.0          # shape (N,)   azimuth degrees
el_deg  = torch.rand(N) * 88.0 + 1.0    # shape (N,)   elevation 1°–89°
speeds  = torch.rand(N) * 150.0 + 50.0  # shape (N,)   speed m/s

print(f"  az_deg.shape  = {az_deg.shape}    (one value per shot)")
print(f"  az_deg[:4]    = {az_deg[:4].round(decimals=1).tolist()}")

# torch.deg2rad converts a whole tensor in one call:
az_rad  = torch.deg2rad(az_deg)
el_rad  = torch.deg2rad(el_deg)

# The physics equations work identically on tensors — element-wise:
t_vec   = 2.0 * speeds * torch.sin(el_rad) / G          # (N,)
x_vec   = speeds * torch.cos(el_rad) * torch.cos(az_rad) * t_vec   # (N,)
y_vec   = speeds * torch.cos(el_rad) * torch.sin(az_rad) * t_vec   # (N,)

# torch.stack([a, b, c], dim=1) lines up three (N,) column vectors
# side-by-side into one (N, 3) matrix:
data = torch.stack([t_vec, x_vec, y_vec], dim=1)   # (N, 3)
print(f"\n  t_vec.shape   = {t_vec.shape}")
print(f"  data.shape    = {data.shape}   (N rows, 3 columns: t, x, y)")
print(f"  data[:3]:\n{data[:3].round(decimals=2)}")

# Column selection with [:, col]:
print(f"\n  data[:, 0]    = t column:  {data[:,0].round(decimals=2).tolist()}")
print(f"  data[:, 1]    = x column:  {data[:,1].round(decimals=1).tolist()}")


# ─────────────────────────────────────────────────────────────────
# SECTION 2  Tensor Reductions
# ─────────────────────────────────────────────────────────────────
# .min(), .max(), .mean(), .sum() collapse a tensor to a scalar.
# They return a 0-dimensional tensor; call .item() to get a Python float.

print("\n=== Section 2: reductions ===\n")
x_col = data[:, 1]
print(f"  x_col.min()           = {x_col.min().item():.1f} m")
print(f"  x_col.max()           = {x_col.max().item():.1f} m")
print(f"  x_col.mean()          = {x_col.mean().item():.1f} m")
print(f"  x_col.min().item()    type: {type(x_col.min().item()).__name__}  "
      f"(Python float — safe for arithmetic outside torch)")


# ─────────────────────────────────────────────────────────────────
# SECTION 3  Tensor Aliasing and .clone()
# ─────────────────────────────────────────────────────────────────
# Tensors have the same aliasing behaviour as Python lists.
# Simple assignment creates a second name for the SAME storage.
# Use .clone() for a fully independent copy — the tensor equivalent
# of copy.deepcopy().

print("\n=== Section 3: tensor aliasing and .clone() ===\n")

original = torch.tensor([1.0, 2.0, 3.0])
alias    = original          # NOT a copy — same underlying storage
alias[0] = 999.0
print(f"  after alias[0] = 999.0:")
print(f"    alias[0]    = {alias[0].item()}")
print(f"    original[0] = {original[0].item()}  ← SAME storage; both changed")

original[0] = 1.0            # restore

deep_copy = original.clone()
deep_copy[0] = 777.0
print(f"\n  after deep_copy[0] = 777.0:")
print(f"    deep_copy[0] = {deep_copy[0].item()}")
print(f"    original[0]  = {original[0].item()}  ← .clone() is independent")


# ─────────────────────────────────────────────────────────────────
# SECTION 4  TorchTargetRange Class
# ─────────────────────────────────────────────────────────────────
# The class interface is identical to TargetRange in file 3.
# The only change is what lives inside generate(), stats(), and
# closest_pair() — lists replaced by tensors, Python loops replaced
# by vectorised operations.

class TorchTargetRange:
    """A TargetRange backed by a PyTorch (N, 3) tensor.

    Tensor layout:  column 0 = t (flight time, s)
                    column 1 = x (east,  m)
                    column 2 = y (north, m)
    """

    def __init__(self, config: LaunchConfig, n: int,
                 el_range: tuple = None, seed: int = None):
        self.config   = config
        self.n        = n
        self.el_range = el_range if el_range is not None else (1.0, 89.0)
        self.seed     = seed
        self._tensor  = None    # (N, 3) tensor, filled by generate()

    def __repr__(self) -> str:
        status = (f"tensor {tuple(self._tensor.shape)}"
                  if self._tensor is not None else "not yet generated")
        return f"TorchTargetRange(n={self.n}, {status})"

    def __len__(self) -> int:
        return self._tensor.shape[0] if self._tensor is not None else 0

    def __bool__(self) -> bool:
        return self._tensor is not None and self._tensor.shape[0] > 0

    @property
    def tensor(self) -> torch.Tensor:
        """The raw (N, 3) tensor.  Requires generate() first."""
        if self._tensor is None:
            raise RuntimeError(f"{self!r}: call generate() first")
        return self._tensor

    @timer
    def generate(self) -> None:
        """Generate N shots using vectorised tensor operations — no Python loop."""
        if self.seed is not None:
            torch.manual_seed(self.seed)

        el_min, el_max = self.el_range

        # All N random values generated in single tensor operations:
        az_deg = torch.rand(self.n) * self.config.az_limit
        el_deg = torch.rand(self.n) * (el_max - el_min) + el_min
        speeds = (torch.rand(self.n)
                  * (self.config.speed_max - self.config.speed_min)
                  + self.config.speed_min)

        ar = torch.deg2rad(az_deg)
        er = torch.deg2rad(el_deg)

        t = 2.0 * speeds * torch.sin(er) / G
        x = speeds * torch.cos(er) * torch.cos(ar) * t
        y = speeds * torch.cos(er) * torch.sin(ar) * t

        self._tensor = torch.stack([t, x, y], dim=1)   # (N, 3)

    @timer
    def stats(self) -> LandingStats:
        """Compute bounding box and centroid using tensor reductions."""
        if not self:
            raise RuntimeError(f"{self!r}: call generate() first")
        xs = self._tensor[:, 1]    # (N,) — all x values
        ys = self._tensor[:, 2]    # (N,) — all y values
        return LandingStats(
            x_min=xs.min().item(),  x_max=xs.max().item(),
            y_min=ys.min().item(),  y_max=ys.max().item(),
            centroid_x=xs.mean().item(),
            centroid_y=ys.mean().item(),
        )

    @timer
    def closest_pair(self) -> tuple:
        """Find the nearest pair of landing points using broadcasting.

        Key idea:
          xy has shape (N, 2).
          xy.unsqueeze(1) has shape (N, 1, 2) — a column of row-vectors.
          xy.unsqueeze(0) has shape (1, N, 2) — a row of row-vectors.
          Subtracting broadcasts to (N, N, 2): diff[i, j] = xy[i] - xy[j].
          Squaring and summing dim=2 gives the (N, N) pairwise distance² matrix.

        Memory note: the diff tensor is N*N*2 floats — about 2 MB at N=500.
        For very large N you would process in chunks; here N ≤ 500 is fine.
        """
        if not self:
            raise RuntimeError(f"{self!r}: call generate() first")

        xy      = self._tensor[:, 1:]              # (N, 2) — drop the time column
        diff    = xy.unsqueeze(1) - xy.unsqueeze(0)  # (N, N, 2)
        dist_sq = (diff ** 2).sum(dim=2)             # (N, N)

        # The diagonal is i==j (distance from a point to itself = 0).
        # We fill it with infinity so argmin skips it.
        dist_sq.fill_diagonal_(float('inf'))

        # argmin on a 2-D tensor returns a flat index into the flattened array.
        # Dividing / taking remainder by N recovers (row, col).
        flat_idx = dist_sq.argmin()
        i = flat_idx // self.n
        j = flat_idx %  self.n
        dist = dist_sq[i, j].sqrt().item()
        return i.item(), j.item(), dist


# ─────────────────────────────────────────────────────────────────
# SECTION 5  Demo and Timing Comparison
# ─────────────────────────────────────────────────────────────────

print("\n=== Section 4–5: TorchTargetRange demo and timing ===\n")

cfg = LaunchConfig(az_limit=360.0, speed_min=50.0, speed_max=200.0)

# Warmup: the very first torch call on a cold process pays JIT overhead.
# A tiny warmup call makes the subsequent timings representative.
_warmup = TorchTargetRange(cfg, 10, seed=0)
_warmup.generate()

ttr = TorchTargetRange(cfg, n=500, seed=42)
print(f"  {ttr!r}\n")

ttr.generate()
print(f"  {ttr!r}")
print(f"  first row (t, x, y): {ttr.tensor[0].tolist()}")

st = ttr.stats()
print(f"\n  stats: x∈[{st.x_min:.0f}, {st.x_max:.0f}]  "
      f"y∈[{st.y_min:.0f}, {st.y_max:.0f}]")
print(f"  centroid: ({st.centroid_x:.1f}, {st.centroid_y:.1f}) m")

bi, bj, bd = ttr.closest_pair()
print(f"\n  closest pair: shots[{bi}] and shots[{bj}], distance = {bd:.1f} m")

# ── head-to-head timing comparison ───────────────────────────────
print("\n--- timing comparison (Python loops vs PyTorch) ---\n")

for N in [500, 5_000, 50_000]:
    # Python loop generate:
    t0 = time.perf_counter()
    py_shots = _py_generate(cfg, N)
    py_gen_ms = (time.perf_counter() - t0) * 1000

    # PyTorch generate:
    ttr2 = TorchTargetRange(cfg, N, seed=42)
    t0 = time.perf_counter()
    ttr2._tensor = None   # reset without printing
    torch.manual_seed(42)
    el_min, el_max = (1.0, 89.0)
    az_d = torch.rand(N)*cfg.az_limit; el_d=torch.rand(N)*88+1; spds=torch.rand(N)*150+50
    ar=torch.deg2rad(az_d); er=torch.deg2rad(el_d)
    tv=2*spds*torch.sin(er)/G; xv=spds*torch.cos(er)*torch.cos(ar)*tv; yv=spds*torch.cos(er)*torch.sin(ar)*tv
    tsr = torch.stack([tv,xv,yv],dim=1)
    th_gen_ms = (time.perf_counter() - t0) * 1000

    print(f"  N={N:>6}  generate:  Python={py_gen_ms:7.2f} ms  "
          f"torch={th_gen_ms:5.2f} ms  speedup={py_gen_ms/max(th_gen_ms,0.01):4.0f}×")

# Closest-pair timing (N=500 is the sweet spot for the broadcasting approach):
N_cp = 500
py_shots_cp = _py_generate(cfg, N_cp)
ttr_cp = TorchTargetRange(cfg, N_cp, seed=42)
ttr_cp.generate()

t0 = time.perf_counter(); _py_closest(py_shots_cp); py_cp_ms = (time.perf_counter()-t0)*1000
t0 = time.perf_counter(); ttr_cp.closest_pair();    th_cp_ms = (time.perf_counter()-t0)*1000

print(f"\n  N={N_cp}  closest_pair: Python={py_cp_ms:.1f} ms  "
      f"torch={th_cp_ms:.1f} ms")
print(f"  (for N≫500 the (N,N,2) diff tensor becomes large — see file 5 "
      f"for strategies)")

print("\n[v4 complete]")
