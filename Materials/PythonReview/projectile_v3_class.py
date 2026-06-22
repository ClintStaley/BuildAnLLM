"""
Python Fundamentals via Projectile Motion  —  File 3 of 5
===========================================================
CONCEPTS COVERED
  • Class definition: __init__, instance variables, __repr__
  • Special (dunder) methods: __len__, __bool__, __str__
  • Instance methods and self
  • @property — controlled access to internal state
  • if / elif / else
  • while loop
  • Raising exceptions from within a class
  • Implementation inheritance: super().__init__(), overriding a method
  • isinstance()

The @timer decorator and helper functions from file 2 are
reproduced here so this file runs standalone.
"""

import math
import random
import time
import functools
from collections import namedtuple


# ── shared infrastructure (also in file 2) ───────────────────────

g = 9.81

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
Landing      = namedtuple('Landing',      ['t', 'x', 'y'])
LandingStats = namedtuple('LandingStats', ['x_min', 'x_max',
                                           'y_min', 'y_max',
                                           'centroid_x', 'centroid_y'])

def single_shot(az_deg, el_deg, speed):
    ar = math.radians(az_deg); er = math.radians(el_deg)
    t  = 2.0 * speed * math.sin(er) / g
    return Landing(t,
                   speed * math.cos(er) * math.cos(ar) * t,
                   speed * math.cos(er) * math.sin(ar) * t)


# ─────────────────────────────────────────────────────────────────
# SECTION 1  Class Basics
# ─────────────────────────────────────────────────────────────────
# A class bundles related data (instance variables) and the functions
# that operate on that data (methods) into a single named unit.
#
# __init__ is the constructor.  Python calls it automatically when you
# write TargetRange(cfg, 500).  The first parameter is always 'self' —
# a reference to the newly created object.
#
# Assigning to self.something inside __init__ creates an instance
# variable: each object gets its own copy.
#
# __repr__ should return an unambiguous string representation of the
# object, useful in the REPL and in debugging.  By convention it looks
# like a constructor call.

class TargetRange:
    """A lazily-generated set of projectile landing points.

    Call generate() to create the point data; all analysis methods
    require generate() to have been called first.
    """

    def __init__(self, config: LaunchConfig, n: int,
                 el_range: tuple = None, seed: int = None):
        """
        Args:
            config:   LaunchConfig controlling azimuth and speed ranges
            n:        number of shots to generate when generate() is called
            el_range: optional (el_min, el_max) in degrees; default (1, 89)
            seed:     optional random seed for reproducible output
        """
        self.config   = config
        self.n        = n
        self.el_range = el_range if el_range is not None else (1.0, 89.0)
        self.seed     = seed
        self._shots   = None    # private by convention (single underscore)
                                # generate() will fill this in

    # ── dunder methods ──────────────────────────────────────────

    def __repr__(self) -> str:
        """Unambiguous developer-facing description."""
        status = (f"{len(self._shots)} shots generated"
                  if self._shots is not None else "not yet generated")
        return f"TargetRange(n={self.n}, {status})"

    def __str__(self) -> str:
        """Readable user-facing description (used by print())."""
        if self._shots is None:
            return f"TargetRange — {self.n} shots pending"
        st = self._compute_stats_internal()
        return (f"TargetRange — {len(self._shots)} shots  "
                f"centroid=({st.centroid_x:.0f}, {st.centroid_y:.0f}) m")

    def __len__(self) -> int:
        """len(tr) returns the number of shots generated so far."""
        return len(self._shots) if self._shots is not None else 0

    def __bool__(self) -> bool:
        """bool(tr) is True only when shots have been generated.
        This allows:  if tr: ...  and  if not tr: ...
        """
        return self._shots is not None and len(self._shots) > 0


# ─────────────────────────────────────────────────────────────────
# SECTION 2  Instance Methods and @property
# ─────────────────────────────────────────────────────────────────
# A @property turns a method into an attribute-style accessor.
# Callers write  tr.shots  instead of  tr.shots(), but the
# property can enforce invariants (here: generate() must come first).
#
# Methods that logically belong to the class but shouldn't be part
# of the public API get a leading underscore by convention.

    @property
    def shots(self) -> list:
        """Read-only access to the generated shots.
        Raises RuntimeError if generate() has not been called.
        """
        if self._shots is None:
            raise RuntimeError(
                f"{self!r}: call generate() before accessing shots"
            )
        return self._shots

    @timer
    def generate(self) -> None:
        """Generate self.n random shots and store them in self._shots."""
        if self.seed is not None:
            random.seed(self.seed)

        el_min, el_max = self.el_range
        self._shots = [
            single_shot(
                random.uniform(0.0, self.config.az_limit),
                random.uniform(el_min, el_max),
                random.uniform(self.config.speed_min, self.config.speed_max),
            )
            for _ in range(self.n)
        ]

    def _compute_stats_internal(self) -> LandingStats:
        """Private helper — computes stats without timing overhead."""
        xs = [s.x for s in self._shots]
        ys = [s.y for s in self._shots]
        return LandingStats(min(xs), max(xs), min(ys), max(ys),
                            sum(xs) / len(xs), sum(ys) / len(ys))

    @timer
    def stats(self) -> LandingStats:
        """Return bounding box and centroid.  Requires generate() first."""
        if not self:    # uses __bool__; False means no shots yet
            raise RuntimeError(f"{self!r}: no shots — call generate() first")
        return self._compute_stats_internal()

    @timer
    def closest_pair(self) -> tuple:
        """Return (i, j, distance_m) of the nearest pair of landing points."""
        if not self:
            raise RuntimeError(f"{self!r}: no shots — call generate() first")
        best = float('inf')
        bi = bj = -1
        shots = self._shots
        for i in range(len(shots)):
            for j in range(i + 1, len(shots)):
                d = math.hypot(shots[i].x - shots[j].x,
                               shots[i].y - shots[j].y)
                if d < best:
                    best = d
                    bi, bj = i, j
        return bi, bj, best


# ─────────────────────────────────────────────────────────────────
# SECTION 3  if / elif / else  and  while
# ─────────────────────────────────────────────────────────────────
# These appear as standalone examples first, then as natural
# method behaviour further down.

    def classify_density(self) -> str:
        """Classify how densely packed the landing points are.

        Demonstrates if / elif / else.
        """
        if not self:
            return "no data"

        st    = self._compute_stats_internal()
        area  = (st.x_max - st.x_min) * (st.y_max - st.y_min)   # m²
        density = len(self._shots) / max(area, 1.0)               # shots/m²

        # if / elif / else — exactly one branch runs:
        if density > 1e-4:
            category = "dense"
        elif density > 1e-5:
            category = "moderate"
        elif density > 1e-6:
            category = "sparse"
        else:
            category = "very sparse"

        return f"{category} ({density:.2e} shots/m²)"

    def shots_in_radius(self, cx: float, cy: float, radius: float) -> list:
        """Return all shots within `radius` metres of (cx, cy).

        Demonstrates a while loop with an explicit index counter —
        sometimes clearer than for-loop when the exit condition is complex.
        """
        if not self:
            raise RuntimeError(f"{self!r}: call generate() first")

        result = []
        i = 0
        while i < len(self._shots):
            s = self._shots[i]
            if math.hypot(s.x - cx, s.y - cy) <= radius:
                result.append(s)
            i += 1
        return result


# ─────────────────────────────────────────────────────────────────
# SECTION 4  Demonstration of TargetRange
# ─────────────────────────────────────────────────────────────────

print("=== Section 1-3: TargetRange class ===\n")

cfg = LaunchConfig(az_limit=360.0, speed_min=50.0, speed_max=200.0)
tr  = TargetRange(cfg, n=600, seed=42)

print(f"  repr before generate: {tr!r}")
print(f"  str  before generate: {tr!s}")
print(f"  bool before generate: {bool(tr)}")
print(f"  len  before generate: {len(tr)}")

# @property guard — accessing shots before generate() raises RuntimeError:
try:
    _ = tr.shots
except RuntimeError as e:
    print(f"\n  property guard: {e}")

# Generate and re-inspect:
print()
tr.generate()
print(f"\n  repr after generate:  {tr!r}")
print(f"  str  after generate:  {tr!s}")
print(f"  bool after generate:  {bool(tr)}")
print(f"  len  after generate:  {len(tr)}")

# Access the stats:
st = tr.stats()
print(f"\n  stats:    x∈[{st.x_min:.0f}, {st.x_max:.0f}]  "
      f"y∈[{st.y_min:.0f}, {st.y_max:.0f}]")
print(f"  centroid: ({st.centroid_x:.1f}, {st.centroid_y:.1f}) m")

# if / elif / else demo:
print(f"\n  density classification: {tr.classify_density()}")

# while loop demo:
near = tr.shots_in_radius(cx=0.0, cy=0.0, radius=500.0)
print(f"  shots within 500 m of origin: {len(near)}")

# Closest pair:
bi, bj, bd = tr.closest_pair()
print(f"\n  closest pair: shots[{bi}] and shots[{bj}], "
      f"distance = {bd:.1f} m")


# ─────────────────────────────────────────────────────────────────
# SECTION 5  Inheritance
# ─────────────────────────────────────────────────────────────────
# Implementation inheritance lets a subclass reuse all of the parent's
# code and override only the parts that differ.
#
# super().__init__(...) calls the parent's constructor so you don't
# have to duplicate that setup work.
#
# Overriding generate() below calls super().generate() first — the
# parent fills self._shots with the full set — then the subclass
# filters out shots that fall outside the keep zone.
#
# The key rule: a subclass IS-A parent class.  isinstance(obj, Parent)
# returns True for both direct instances and subclass instances.

class BoundedTargetRange(TargetRange):
    """Like TargetRange, but discards any shots outside a rectangular
    bounding box after generation.

    This models a range with impassable terrain or restricted airspace
    on two sides.
    """

    def __init__(self, config: LaunchConfig, n: int,
                 x_bounds: tuple, y_bounds: tuple,
                 el_range: tuple = None, seed: int = None):
        """
        Args:
            x_bounds: (x_min, x_max) keep zone in metres
            y_bounds: (y_min, y_max) keep zone in metres
            All other args forwarded to TargetRange.__init__.
        """
        super().__init__(config, n, el_range, seed)   # parent sets up self.config, self.n, etc.
        self.x_bounds = x_bounds
        self.y_bounds = y_bounds

    def __repr__(self) -> str:
        status = (f"{len(self._shots)} kept"
                  if self._shots is not None else "not yet generated")
        return (f"BoundedTargetRange(n={self.n}, "
                f"x={self.x_bounds}, y={self.y_bounds}, {status})")

    @timer
    def generate(self) -> None:
        """Generate shots, then discard those outside the keep zone."""
        super().generate()          # parent fills self._shots with self.n entries
        xlo, xhi = self.x_bounds
        ylo, yhi = self.y_bounds
        before = len(self._shots)

        # Filter in-place using a list comprehension:
        self._shots = [
            s for s in self._shots
            if xlo <= s.x <= xhi and ylo <= s.y <= yhi
        ]
        print(f"    kept {len(self._shots)} of {before} shots inside the keep zone")


print("\n=== Section 5: BoundedTargetRange inheritance ===\n")

btr = BoundedTargetRange(
    cfg, n=800,
    x_bounds=(-1000, 1000),
    y_bounds=(-1000, 1000),
    seed=42,
)

print(f"  repr: {btr!r}")
print()
btr.generate()

print(f"\n  repr after generate: {btr!r}")
print(f"  isinstance(btr, TargetRange):        {isinstance(btr, TargetRange)}")
print(f"  isinstance(btr, BoundedTargetRange): {isinstance(btr, BoundedTargetRange)}")

# All parent methods (stats, closest_pair, etc.) work unchanged
# because self._shots is populated the same way:
st2 = btr.stats()
print(f"\n  stats: x∈[{st2.x_min:.0f}, {st2.x_max:.0f}]  "
      f"y∈[{st2.y_min:.0f}, {st2.y_max:.0f}]")
print(f"  density: {btr.classify_density()}")

print("\n[v3 complete]")
