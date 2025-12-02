import numpy as np
from typing import Optional, Tuple


import numpy as np
from typing import Optional, Tuple


def footprint_polygon(
    boundary: dict | None,
) -> Optional[Tuple[list[float], list[float], float, float, float, float]]:
    """
    Build a survey footprint polygon from boundary metadata.

    If 'corners' are present in boundary['boundary'], uses them directly.
    Otherwise falls back to an axis-aligned box from x_range/y_range.

    Returns
    -------
    xs, ys, x_min, x_max, y_min, y_max
    where xs, ys are lists of polygon vertices (closed: first == last).
    """
    if not boundary:
        return None

    b = boundary.get("boundary", boundary)

    # 1) Preferred: explicit 4 corners
    corners = b.get("corners", None)
    if corners and len(corners) >= 4:
        # Ensure we have exactly 4 and close the polygon
        cs = corners[:4]
        xs = [c[0] for c in cs] + [cs[0][0]]
        ys = [c[1] for c in cs] + [cs[0][1]]
        return xs, ys, min(xs), max(xs), min(ys), max(ys)

    # 2) Fallback: simple axis-aligned box
    try:
        x_min, x_max = b["x_range"]
        y_min, y_max = b["y_range"]
    except Exception:
        return None

    x_min = float(x_min)
    x_max = float(x_max)
    y_min = float(y_min)
    y_max = float(y_max)

    xs = [x_min, x_max, x_max, x_min, x_min]
    ys = [y_min, y_min, y_max, y_max, y_min]

    return xs, ys, x_min, x_max, y_min, y_max
