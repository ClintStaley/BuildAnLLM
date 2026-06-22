"""
Python Fundamentals via Projectile Motion  —  File 2 of 5
===========================================================
CONCEPTS COVERED
  • The @timer decorator — and why it introduces *args / **kwargs
  • functools.wraps
  • namedtuple: defining, accessing fields, _asdict, _replace
  • Functions with type hints and default arguments
  • *args for a variadic optional argument
  • User input: input(), str.split(), int(), float(), tuple unpacking
  • enumerate() and zip()
  • A function returning a namedtuple; tuple unpacking on the call side
  • Closest-pair via O(N²) nested loops — and measuring why it hurts
  • None and truthiness; 'is None' vs 'not x'
  • Exception handling: try / except / else / finally; raising

Same physical scenario as file 1:
  t = 2 * speed * sin(el_rad) / g
  x = speed * cos(el_rad) * cos(az_rad) * t
  y = speed * cos(el_rad) * sin(az_rad) * t
"""

import math
import random
import time
import functools
from collections import namedtuple

g = 9.81    # m/s²


# ─────────────────────────────────────────────────────────────────
# SECTION 1  The @timer Decorator
# ─────────────────────────────────────────────────────────────────
# A decorator is a function that wraps another function, adding
# behaviour before and/or after it runs.  The @ syntax is just
# shorthand for:   generate_shots = timer(generate_shots)
#
# To wrap ANY function (regardless of how many arguments it takes),
# the wrapper must accept *args (any positional arguments, collected
# into a tuple) and **kwargs (any keyword arguments, collected into
# a dict).  It then forwards them unchanged to the original function.
#
# functools.wraps copies the original function's __name__, __doc__,
# etc. onto the wrapper — without it, every decorated function would
# appear to be called "wrapper" in error messages and help text.

def timer(func):
    """Decorator: prints elapsed time in ms after each call."""
    @functools.wraps(func)          # preserve __name__, __doc__, etc.
    def wrapper(*args, **kwargs):   # accept any signature
        start  = time.perf_counter()
        result = func(*args, **kwargs)   # call the real function
        ms     = (time.perf_counter() - start) * 1000
        print(f"  [{func.__name__}]  {ms:.3f} ms")
        return result
    return wrapper

# Quick demonstration of *args and **kwargs before we use @timer:
def show_args(*args, **kwargs):
    print(f"  positional: {args}")
    print(f"  keyword:    {kwargs}")

print("=== Section 1: *args / **kwargs / @timer ===")
show_args(1, "hello", True, x=10, label="test")


# ─────────────────────────────────────────────────────────────────
# SECTION 2  namedtuple
# ─────────────────────────────────────────────────────────────────
# A namedtuple is a lightweight, immutable record type — like a tiny
# class with no methods, only named fields.  Access fields by name
# (like an object attribute) or by integer index (like a tuple).
# It costs almost nothing in memory compared to a regular class.

LaunchConfig = namedtuple('LaunchConfig', ['az_limit', 'speed_min', 'speed_max'])
Landing      = namedtuple('Landing',      ['t', 'x', 'y'])
LandingStats = namedtuple('LandingStats', ['x_min', 'x_max',
                                           'y_min', 'y_max',
                                           'centroid_x', 'centroid_y'])

print("\n=== Section 2: namedtuple ===")

cfg  = LaunchConfig(az_limit=360.0, speed_min=50.0, speed_max=200.0)
shot = Landing(t=5.1, x=312.4, y=-88.7)

print(f"  cfg             = {cfg}")
print(f"  shot            = {shot}")
print(f"  shot.t          = {shot.t}    (field access by name)")
print(f"  shot[0]         = {shot[0]}    (field access by index — same value)")
print(f"  shot._asdict()  = {shot._asdict()}")

# _replace returns a NEW namedtuple with one field changed (they're immutable).
corrected = shot._replace(t=5.2)
print(f"  corrected       = {corrected}   (original shot unchanged: {shot.t})")

# namedtuples support unpacking just like plain tuples:
t_val, x_val, y_val = shot
print(f"  unpacked: t={t_val}  x={x_val}  y={y_val}")


# ─────────────────────────────────────────────────────────────────
# SECTION 3  Core Functions
# ─────────────────────────────────────────────────────────────────
# Python type hints (e.g. az_deg: float) are optional annotations —
# they do NOT enforce types at runtime, but they document intent and
# are checked by IDE tools and type checkers like mypy.

def single_shot(az_deg: float, el_deg: float, speed: float) -> Landing:
    """Compute the landing point for one projectile.

    Args:
        az_deg: azimuth in degrees (counter-clockwise from x-axis)
        el_deg: elevation in degrees above horizontal (1–89)
        speed:  launch speed in m/s

    Returns:
        Landing namedtuple with fields t (s), x (m), y (m)
    """
    ar = math.radians(az_deg)
    er = math.radians(el_deg)
    t  = 2.0 * speed * math.sin(er) / g
    x  = speed * math.cos(er) * math.cos(ar) * t
    y  = speed * math.cos(er) * math.sin(ar) * t
    return Landing(t, x, y)

print("\n=== Section 3: functions ===")
result = single_shot(45.0, 30.0, 120.0)
print(f"  single_shot(45°, 30°, 120 m/s) = {result}")
print(f"  range = {math.hypot(result.x, result.y):.1f} m")

# Default arguments: the parameter gets its default if the caller
# omits it.  Default values are evaluated ONCE at definition time.
def describe_shot(landing: Landing, label: str = "shot") -> str:
    return (f"{label}: t={landing.t:.2f}s  "
            f"({landing.x:.1f}, {landing.y:.1f}) m  "
            f"range={math.hypot(landing.x, landing.y):.1f} m")

print(f"\n  {describe_shot(result)}")
print(f"  {describe_shot(result, label='Alpha-1')}")


# ─────────────────────────────────────────────────────────────────
# SECTION 4  *args for a Variadic Optional Argument
# ─────────────────────────────────────────────────────────────────
# *args inside a function definition collects ALL extra positional
# arguments into a tuple.  Here we use it to make the elevation
# range optional: callers who don't need to change it omit it,
# while callers who do pass a single (el_min, el_max) tuple.

@timer
def generate_shots(config: LaunchConfig, n: int, *args) -> list:
    """Generate n random Landing results.

    Args:
        config: LaunchConfig with azimuth and speed limits
        n:      number of shots to generate
        *args:  optional single argument — a (el_min, el_max) tuple.
                Defaults to (1.0, 89.0) if omitted.

    Returns:
        list of Landing namedtuples
    """
    el_min, el_max = args[0] if args else (1.0, 89.0)
    return [
        single_shot(
            random.uniform(0.0, config.az_limit),
            random.uniform(el_min, el_max),
            random.uniform(config.speed_min, config.speed_max)
        )
        for _ in range(n)
    ]

print("\n=== Section 4: *args for optional elevation range ===")
random.seed(42)
shots_default = generate_shots(cfg, 5)               # default el range 1°–89°
shots_low     = generate_shots(cfg, 5, (1.0, 20.0))  # force low-angle shots

print(f"  default el range — first shot: {shots_default[0]}")
print(f"  low el range     — first shot: {shots_low[0]}")
print(f"  (low-angle shots fly fast and flat — shorter t, longer range)")


# ─────────────────────────────────────────────────────────────────
# SECTION 5  User Input and String Parsing
# ─────────────────────────────────────────────────────────────────
# input() in an interactive session blocks until the user presses Enter,
# then returns the raw text as a string.  str.split() breaks a string
# on whitespace (by default) and returns a list of substrings.
#
# For this demonstration we assign the string directly so the file
# runs without waiting for keyboard input — but the parsing code is
# identical to what you'd write after a real input() call.

print("\n=== Section 5: user input parsing ===")

# In a live session you would write:
#   raw = input("Enter: n  az_limit  speed_min  speed_max\n> ")
# Here we simulate it:
raw = "1000  360  50.0  200.0"

print(f"  simulated input string: {raw!r}")

parts = raw.split()          # ['1000', '360', '50.0', '200.0']
print(f"  after .split()        : {parts}   (all still strings)")

# Convert each part to the appropriate numeric type:
n_input   = int(parts[0])
az_input  = float(parts[1])
sp_min    = float(parts[2])
sp_max    = float(parts[3])

# Tuple unpacking lets you do the conversion in one line:
n2, az2, sp_min2, sp_max2 = int(parts[0]), float(parts[1]), float(parts[2]), float(parts[3])
print(f"  unpacked: n={n2}  az_limit={az2}  speed {sp_min2}–{sp_max2} m/s")

# str.split() with a separator argument is useful for comma-separated input:
csv_line = "Alpha,45.0,30.0,120.0"
label, az_s, el_s, spd_s = csv_line.split(',')
print(f"\n  CSV parse: label={label!r}  az={az_s}  el={el_s}  spd={spd_s}")

# strip() removes leading/trailing whitespace (important after split on
# lines that may have trailing spaces or newlines):
padded = "  120.5  \n"
print(f"  {padded!r}.strip() → {padded.strip()!r}")


# ─────────────────────────────────────────────────────────────────
# SECTION 6  enumerate() and zip()
# ─────────────────────────────────────────────────────────────────
# enumerate(iterable) yields (index, value) pairs — you get the loop
# counter for free, without maintaining a separate counter variable.
#
# zip(a, b) pairs up elements from two iterables.  It stops at the
# shortest one.  zip(*lists) is the transpose — columns become rows.

print("\n=== Section 6: enumerate and zip ===")
random.seed(42)
sample = generate_shots(cfg, 5)

print("  enumerate:")
for i, s in enumerate(sample):
    print(f"    [{i}]  t={s.t:.2f}s  range={math.hypot(s.x, s.y):.1f}m")

# Compare two independent sets of shots element-by-element:
random.seed(99)
sample_b = generate_shots(cfg, 5)

print("\n  zip — pairing shots from two runs and computing distance between them:")
for i, (a, b) in enumerate(zip(sample, sample_b)):
    d = math.hypot(a.x - b.x, a.y - b.y)
    print(f"    pair [{i}]:  distance = {d:.1f} m")


# ─────────────────────────────────────────────────────────────────
# SECTION 7  Stats Function — returning a namedtuple, unpacking it
# ─────────────────────────────────────────────────────────────────

@timer
def compute_stats(shots: list) -> LandingStats:
    """Compute bounding box and centroid for a list of Landing records."""
    xs = [s.x for s in shots]
    ys = [s.y for s in shots]
    return LandingStats(
        x_min=min(xs), x_max=max(xs),
        y_min=min(ys), y_max=max(ys),
        centroid_x=sum(xs) / len(xs),
        centroid_y=sum(ys) / len(ys),
    )

print("\n=== Section 7: stats function and tuple unpacking ===")

random.seed(42)
big_shots = generate_shots(cfg, n_input)   # 1 000 shots from Section 5

# Access the result as a namedtuple:
stats = compute_stats(big_shots)
print(f"  stats = {stats}")
print(f"  x bounding box: [{stats.x_min:.1f}, {stats.x_max:.1f}]")
print(f"  centroid: ({stats.centroid_x:.1f}, {stats.centroid_y:.1f})")

# The caller can also unpack the return value directly:
xlo, xhi, ylo, yhi, cx, cy = compute_stats(big_shots)
print(f"\n  unpacked centroid: cx={cx:.1f}  cy={cy:.1f}")


# ─────────────────────────────────────────────────────────────────
# SECTION 8  Closest Pair — Nested Loops and the O(N²) Cost
# ─────────────────────────────────────────────────────────────────
# The naive algorithm compares every pair of points exactly once.
# With N points there are N*(N-1)/2 pairs.
# For N=500  that is ~125 000 comparisons    — fast.
# For N=5000 that is ~12 500 000 comparisons — noticeably slow.
# File 4 will replace this with a PyTorch broadcasting approach.

@timer
def closest_pair(shots: list) -> tuple:
    """Find the two closest landing points by O(N²) exhaustive search.

    Returns:
        (index_i, index_j, distance_m) of the nearest pair.
    """
    best_dist = float('inf')    # +infinity — any real distance will be smaller
    best_i = best_j = -1

    for i in range(len(shots)):
        for j in range(i + 1, len(shots)):    # j > i avoids duplicate pairs
            dx   = shots[i].x - shots[j].x
            dy   = shots[i].y - shots[j].y
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_i, best_j = i, j

    return best_i, best_j, best_dist

print("\n=== Section 8: closest pair (nested loops) ===")

for trial_n in [500, 2000]:
    random.seed(42)
    trial_shots = generate_shots(cfg, trial_n)
    bi, bj, bd = closest_pair(trial_shots)
    print(f"  N={trial_n}: closest pair is shots [{bi}] and [{bj}], "
          f"distance = {bd:.1f} m")
    # Notice how the time grows much faster than N — it's O(N²).


# ─────────────────────────────────────────────────────────────────
# SECTION 9  None and Truthiness
# ─────────────────────────────────────────────────────────────────
# In Python every object has a truth value.  The following are ALL
# "falsey" (they evaluate to False in an if-statement):
#   None, 0, 0.0, 0j,  ""  (empty string),
#   []   (empty list), {}  (empty dict), ()  (empty tuple)
# Everything else is truthy.
#
# None in particular is used as a sentinel — "not yet assigned" or
# "not applicable".

print("\n=== Section 9: None and truthiness ===")

falsey_values = [None, 0, 0.0, "", [], {}, ()]
for v in falsey_values:
    print(f"  bool({v!r:12}) = {bool(v)}")

# The important Python idiom:
cache = None    # signals "nothing computed yet"

if cache is None:               # CORRECT for None checks
    print("\n  cache is None — computing...")
    cache = compute_stats(big_shots)

if cache:                       # truthy — a populated namedtuple is truthy
    print(f"  cache hit: centroid = ({cache.centroid_x:.1f}, {cache.centroid_y:.1f})")

# Contrast: 'if not cache' would be wrong here if cache were an empty list,
# because an empty list is falsey even though it isn't None.
# Use 'is None' / 'is not None' when you specifically mean None.

# PyTorch preview: you will see this exact pattern constantly —
#   if self._tensor is None:
#       self._generate()


# ─────────────────────────────────────────────────────────────────
# SECTION 10  Exception Handling
# ─────────────────────────────────────────────────────────────────
# Exceptions are Python's mechanism for signalling errors.
# try    — the code that might fail
# except — what to do if a specific exception is raised
# else   — runs only if NO exception occurred
# finally — runs ALWAYS, whether or not an exception occurred
#           (used to close files, release resources, etc.)

print("\n=== Section 10: exception handling ===")

def parse_launch_string(raw: str) -> LaunchConfig:
    """Parse 'az_limit speed_min speed_max' from a whitespace-separated string.
    Raises ValueError with a descriptive message on bad input.
    """
    parts = raw.strip().split()
    if len(parts) != 3:
        raise ValueError(
            f"Expected 3 values (az_limit speed_min speed_max), got {len(parts)}"
        )
    az_lim, sp_lo, sp_hi = float(parts[0]), float(parts[1]), float(parts[2])
    if sp_lo >= sp_hi:
        raise ValueError(
            f"speed_min ({sp_lo}) must be less than speed_max ({sp_hi})"
        )
    return LaunchConfig(az_limit=az_lim, speed_min=sp_lo, speed_max=sp_hi)

# Good input:
try:
    good_cfg = parse_launch_string("360  50  200")
    print(f"  parsed ok: {good_cfg}")
except ValueError as e:
    print(f"  parse failed: {e}")
else:
    print(f"  (else block: parsing succeeded — LaunchConfig is ready to use)")
finally:
    print(f"  (finally block: always runs)")

# Too few fields:
print()
try:
    bad_cfg = parse_launch_string("360 50")
except ValueError as e:
    print(f"  too few fields → ValueError: {e}")

# Speed range inverted:
print()
try:
    bad_cfg = parse_launch_string("360 200 50")
except ValueError as e:
    print(f"  inverted speeds → ValueError: {e}")

# Non-numeric value:
print()
try:
    bad_cfg = parse_launch_string("360 abc 200")
except ValueError as e:
    print(f"  non-numeric → ValueError: {e}")

# Catching multiple exception types:
print()
for bad_raw in ["", None]:
    try:
        parse_launch_string(bad_raw)
    except (ValueError, AttributeError, TypeError) as e:
        print(f"  {type(e).__name__}: {e}")

print("\n[v2 complete]")
