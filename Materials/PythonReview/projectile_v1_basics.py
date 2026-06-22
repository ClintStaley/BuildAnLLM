"""
Python Fundamentals via Projectile Motion  —  File 1 of 5
===========================================================
CONCEPTS COVERED
  • Variables: str, int, float; the type() function
  • Basic arithmetic operators and the math library
  • Printing with f-strings
  • 1-D lists: creation, indexing, mutation
  • 2-D lists: for-loop version, then list-comprehension version
  • Sequence indexing and slicing
  • Aliasing, shallow copy, and deep copy

THE PHYSICAL SCENARIO
  A projectile is launched from the origin (0, 0) with a given
  azimuth (angle counter-clockwise from the x-axis, in degrees),
  elevation angle (degrees above horizontal), and launch speed (m/s).
  Air resistance is ignored.

  Landing equations  (all angles converted to radians first):
      t  = 2 * speed * sin(elevation) / g          flight time
      x  = speed * cos(elevation) * cos(azimuth) * t
      y  = speed * cos(elevation) * sin(azimuth) * t

  g = 9.81 m/s²   (standard gravity)
"""

import math
import random
import copy

random.seed(42)     # fix the seed — every run produces identical output
g = 9.81            # m/s²; a module-level constant by convention (not enforced)


# ─────────────────────────────────────────────────────────────────
# SECTION 1  Variables and Types
# ─────────────────────────────────────────────────────────────────
# Python infers the type from the assigned value — no declaration needed.
# The three most common scalar types are str, int, and float.

mission_label = "Alpha Battery, Fire Mission 1"   # str  — text, any quotes
n_rounds      = 6                                  # int  — whole number
launch_speed  = 120.0                              # float — decimal number

# type() returns the type object; __name__ gives its readable string.
print("=== Section 1: Types ===")
print(f"  {type(mission_label).__name__:8s}  {mission_label}")
print(f"  {type(n_rounds).__name__:8s}  {n_rounds}")
print(f"  {type(launch_speed).__name__:8s}  {launch_speed}")

# Strings support methods; split() will reappear in the functions file.
print(f"\n  mission_label.upper()  → {mission_label.upper()}")
print(f"  mission_label.split()  → {mission_label.split()}")


# ─────────────────────────────────────────────────────────────────
# SECTION 2  Arithmetic and the math Library
# ─────────────────────────────────────────────────────────────────
# Built-in operators:  +  -  *  /  //  %  **
#   /   — true division (always returns float, even for integers)
#   //  — floor division (truncates toward negative infinity)
#   %   — modulo (remainder)
#   **  — exponentiation

print("\n=== Section 2: Arithmetic and math library ===")
print(f"  7 / 2   = {7 / 2}")       # 3.5
print(f"  7 // 2  = {7 // 2}")      # 3
print(f"  7 % 2   = {7 % 2}")       # 1
print(f"  2 ** 10 = {2 ** 10}")     # 1024

# The math library supplies trig, constants, rounding, etc.
# IMPORTANT: all trig functions in Python work in RADIANS.
# math.radians() converts degrees → radians.
az_deg = 45.0
el_deg = 30.0
az_rad = math.radians(az_deg)       # π/4  ≈ 0.7854
el_rad = math.radians(el_deg)       # π/6  ≈ 0.5236

print(f"\n  math.pi              = {math.pi:.6f}")
print(f"  math.radians(45)     = {az_rad:.6f}   (π/4)")
print(f"  math.cos(45°)        = {math.cos(az_rad):.6f}   (≈ 0.707107 = 1/√2)")
print(f"  math.sin(30°)        = {math.sin(el_rad):.6f}   (exactly 0.5)")
print(f"  math.sqrt(2)         = {math.sqrt(2):.6f}")
print(f"  math.hypot(3, 4)     = {math.hypot(3, 4):.6f}   (√(3²+4²), Euclidean distance)")


# ─────────────────────────────────────────────────────────────────
# SECTION 3  Single Projectile Calculation
# ─────────────────────────────────────────────────────────────────
# Using the canonical shot (az=45°, el=30°, speed=120 m/s).
# Maximum range for a given speed occurs at el=45°; 30° gives a shorter,
# lower trajectory — a common artillery angle.

speed   = 120.0   # m/s

t_land  = 2.0 * speed * math.sin(el_rad) / g
x_land  = speed * math.cos(el_rad) * math.cos(az_rad) * t_land
y_land  = speed * math.cos(el_rad) * math.sin(az_rad) * t_land
range2d = math.hypot(x_land, y_land)   # straight-line distance from launch

# (We'll print these in Section 4 using f-strings.)


# ─────────────────────────────────────────────────────────────────
# SECTION 4  f-Strings
# ─────────────────────────────────────────────────────────────────
# An f-string is prefixed with f (or F).  Any Python expression inside
# curly braces { } is evaluated and inserted into the string.
#
# Format specifiers follow a colon inside the braces:
#   .3f   — fixed-point, 3 decimal places
#   .1f   — fixed-point, 1 decimal place
#   >8.1f — right-align in a field of width 8, 1 decimal place
#   ,     — thousands separator
#
# The conversion flags !r (repr) and !s (str) change how the value
# is formatted before the format spec is applied.

print("\n=== Section 4: f-strings ===")
print(f"  azimuth {az_deg}°, elevation {el_deg}°, speed {speed} m/s")
print(f"  flight time : {t_land:>8.3f} s")
print(f"  landing x   : {x_land:>8.1f} m")
print(f"  landing y   : {y_land:>8.1f} m")
print(f"  range       : {range2d:>8.1f} m")

big_number = 1_234_567.89              # underscores in literals are allowed
print(f"\n  thousands separator : {big_number:,.2f}")

label = "shot_001"
print(f"  plain string : {label}    with !r : {label!r}")  # !r adds quotes


# ─────────────────────────────────────────────────────────────────
# SECTION 5  1-D Lists
# ─────────────────────────────────────────────────────────────────
# A list is an ordered, mutable sequence of any values — you can
# mix types freely, though in practice lists usually hold one type.

print("\n=== Section 5: 1-D lists ===")

landing = [t_land, x_land, y_land]    # 3-element list from the shot above

print(f"  landing              = {[round(v, 2) for v in landing]}")
print(f"  landing[0]           = {landing[0]:.3f}   (first element; indices start at 0)")
print(f"  landing[2]           = {landing[2]:.1f}   (third element)")
print(f"  landing[-1]          = {landing[-1]:.1f}   (last element; -1 counts from end)")
print(f"  len(landing)         = {len(landing)}")

# Lists are mutable — you can reassign any element in place.
landing[0] = 0.0
print(f"\n  after landing[0] = 0.0  → {[round(v,2) for v in landing]}")
landing[0] = t_land              # restore

# Lists can hold mixed types — even other lists.
meta = [mission_label, n_rounds, launch_speed, True, None]
print(f"\n  mixed-type list: {meta}")

# Useful list methods:
nums = [3, 1, 4, 1, 5, 9, 2, 6]
nums.append(7)                          # add one element at the end
nums.sort()                             # sort in place
print(f"\n  after append and sort: {nums}")
print(f"  nums.index(5)        = {nums.index(5)}   (position of value 5)")


# ─────────────────────────────────────────────────────────────────
# SECTION 6  2-D Lists: for-loop Version
# ─────────────────────────────────────────────────────────────────
# A 2-D list is a list whose elements are themselves lists.
# Here each row is one [t, x, y] landing result.
# We generate N random shots and collect the results.

print("\n=== Section 6: 2-D list via for-loop ===")

N = 12    # keep small so output is readable

shots = []       # start empty; build by appending inside the loop

for _ in range(N):
    # _ is conventional for a loop variable you don't use.
    az  = random.uniform(0.0, 360.0)    # degrees, full circle
    el  = random.uniform(1.0,  89.0)    # degrees, avoid flat (0°) and straight-up (90°)
    spd = random.uniform(50.0, 200.0)   # m/s

    ar = math.radians(az)
    er = math.radians(el)
    t  = 2.0 * spd * math.sin(er) / g
    x  = spd * math.cos(er) * math.cos(ar) * t
    y  = spd * math.cos(er) * math.sin(ar) * t

    shots.append([t, x, y])     # each row is a 3-element list

# Access patterns for a 2-D list:
print(f"  shots[0]       = {[round(v,2) for v in shots[0]]}   (first row)")
print(f"  shots[-1]      = {[round(v,2) for v in shots[-1]]}   (last row)")
print(f"  shots[2][1]    = {shots[2][1]:.2f}   (row 2, col 1 → x of the 3rd shot)")

print(f"\n  First 3 rows:")
for row in shots[:3]:
    print(f"    {[round(v, 2) for v in row]}")


# ─────────────────────────────────────────────────────────────────
# SECTION 7  2-D Lists: List Comprehension Version
# ─────────────────────────────────────────────────────────────────
# A list comprehension collapses the create-empty / loop / append
# pattern into one readable expression.
#
# General form:
#   [expression  for  variable  in  iterable]
#
# It can also include a filter:
#   [expression  for  variable  in  iterable  if  condition]
#
# To produce the exact same random shots as the for-loop above,
# we re-seed and pre-generate the raw (az, el, speed) triples.

random.seed(42)    # re-seed to reproduce the same sequence

# Step 1 — generate the random input triples:
params = [
    (random.uniform(0.0, 360.0),
     random.uniform(1.0,  89.0),
     random.uniform(50.0, 200.0))
    for _ in range(N)
]

# Step 2 — a helper that converts one triple to [t, x, y].
# (Functions are covered fully in file 2; this one is just a
#  short helper to keep the comprehension below readable.)
def _one_shot(az_deg, el_deg, spd):
    ar = math.radians(az_deg)
    er = math.radians(el_deg)
    t  = 2.0 * spd * math.sin(er) / g
    x  = spd * math.cos(er) * math.cos(ar) * t
    y  = spd * math.cos(er) * math.sin(ar) * t
    return [t, x, y]

# Step 3 — the comprehension itself:
shots_c = [_one_shot(az, el, spd) for az, el, spd in params]

print("\n=== Section 7: list comprehension ===")
print(f"  shots_c[0]     = {[round(v,2) for v in shots_c[0]]}")
print(f"  matches for-loop? {shots_c == shots}   (same seed → identical values)")

# Filtered comprehension — keep only short-flight shots:
fast = [row for row in shots_c if row[0] < 10.0]
print(f"\n  shots with flight time < 10 s: {len(fast)} of {N}")

# Nested comprehension — extract x-coordinates for all shots:
x_coords = [row[1] for row in shots_c]
print(f"  x-coords: {[round(v, 1) for v in x_coords]}")


# ─────────────────────────────────────────────────────────────────
# SECTION 8  Indexing and Slicing
# ─────────────────────────────────────────────────────────────────
# Slicing syntax:  sequence[start : stop : step]
#   start — inclusive; defaults to 0
#   stop  — exclusive; defaults to len(sequence)
#   step  — defaults to 1; use -1 to reverse
#
# Negative indices count from the right: -1 is the last element,
# -2 is second-to-last, etc.

print("\n=== Section 8: indexing and slicing ===")
data = [round(v, 1) for v in x_coords]   # keep it readable
print(f"  data          = {data}")
print(f"  data[:4]      = {data[:4]}      (first 4)")
print(f"  data[-3:]     = {data[-3:]}        (last 3)")
print(f"  data[2:8]     = {data[2:8]}  (indices 2 through 7)")
print(f"  data[::3]     = {data[::3]}     (every 3rd element)")
print(f"  data[::-1]    = {data[::-1]}  (reversed)")

# Slicing a 2-D list gives a sub-list of rows:
print(f"\n  shots_c[1:4]  (rows 1–3):")
for row in shots_c[1:4]:
    print(f"    {[round(v, 2) for v in row]}")


# ─────────────────────────────────────────────────────────────────
# SECTION 9  Aliasing, Shallow Copy, and Deep Copy
# ─────────────────────────────────────────────────────────────────
# This is one of the most common sources of bugs for Python beginners.
#
# ALIASING — assignment does NOT copy a list.
# It creates a second variable that points to the SAME object.

print("\n=== Section 9: aliasing, shallow copy, deep copy ===")

alias = shots_c                     # alias and shots_c are the same object in RAM
alias[0][0] = 9999.0                # mutating through alias also mutates shots_c

print("ALIASING:")
print(f"  after alias[0][0] = 9999.0:")
print(f"    alias[0][0]      = {alias[0][0]}")
print(f"    shots_c[0][0]    = {shots_c[0][0]}  ← SAME object; both changed")
print(f"    alias is shots_c → {alias is shots_c}")

shots_c[0][0] = shots[0][0]        # restore using original shots list

# SHALLOW COPY — copy.copy() creates a new outer list, but the inner
# lists are still shared between the copy and the original.

shallow = copy.copy(shots_c)
shallow[0][0] = 8888.0             # inner list [0] is shared — shots_c[0][0] also changes

print("\nSHALLOW COPY (copy.copy):")
print(f"  after shallow[0][0] = 8888.0:")
print(f"    shallow[0][0]    = {shallow[0][0]}")
print(f"    shots_c[0][0]    = {shots_c[0][0]}  ← inner lists still shared")
print(f"    shallow is shots_c     → {shallow is shots_c}   (different outer list)")
print(f"    shallow[0] is shots_c[0] → {shallow[0] is shots_c[0]}  (same inner list)")

shots_c[0][0] = shots[0][0]        # restore

# DEEP COPY — copy.deepcopy() recursively duplicates every level.
# The result is fully independent of the original.

deep = copy.deepcopy(shots_c)
deep[0][0] = 7777.0                # shots_c is NOT affected

print("\nDEEP COPY (copy.deepcopy):")
print(f"  after deep[0][0] = 7777.0:")
print(f"    deep[0][0]       = {deep[0][0]}")
print(f"    shots_c[0][0]    = {shots_c[0][0]:.3f}  ← independent; unchanged")
print(f"    deep[0] is shots_c[0] → {deep[0] is shots_c[0]}  (different inner list)")

# PyTorch note (preview — covered in detail in files 4 & 5):
#   b = a           # alias — same tensor storage
#   b = a.clone()   # deep copy — new independent tensor
#   The same bug you just saw with lists appears with tensors.

print("\n[v1 complete]")
