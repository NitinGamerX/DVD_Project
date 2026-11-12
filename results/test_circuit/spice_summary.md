# 3-bit CLA: SPICE reference (Part C, step 6)

39 cells, 186 transistors; 128 input combinations x 200 corners.

| Leakage | Median | Min | Max | Lowest-leakage input (median) | Highest-leakage input (median) |
|---|---|---|---|---|---|
| total from every source at VDD | 14.083 µA | 3.053 µA | 46.565 µA | A=0 B=0 cin=0: 12.621 µA | A=7 B=0 cin=1: 15.997 µA |
| VDD supply only | 13.079 µA | 2.580 µA | 44.532 µA | A=7 B=7 cin=1: 11.782 µA | A=0 B=7 cin=1: 14.649 µA |

Worst internal-net offset from its rail: 1.403 % of VDD (15.42 mV).
