#!/usr/bin/env python3
"""Generate print-ready AprilTag sheets for the cell's stations.

    python make_apriltags.py                      # the three route stations
    python make_apriltags.py --ids 0 1 2 3 4      # specific ids
    python make_apriltags.py --size-mm 200        # smaller tags

Then print at **100% / "Actual size"** -- never "Fit to page". Every sheet
carries a 100 mm reference line; measure it with a ruler before mounting. If
it is not 100.0 mm the printer rescaled, and every distance the robot
estimates will be wrong by that same factor.

Why tag36h11
------------
It is the AprilTag family with the best false-positive rejection, and 587
unique ids -- far more than this cell will ever need. Smaller families
(16h5) pack more ids into less area but misdetect far more often, which on a
walking robot means confidently driving toward a tag that was never there.

Why the size matters more than anything else
--------------------------------------------
Detection needs roughly 25-30 pixels across the tag in the image. The R1's
camera is 1280x720 across a ~160 deg fisheye -- about 8 px per degree near
the centre, and worse toward the edges. So a tag of side S seen from distance
D spans about `2*atan(S/2D)` degrees:

    | tag side | 2 m    | 3 m    | 5 m    |
    |----------|--------|--------|--------|
    | 150 mm   | 34 px  | 23 px  | 14 px  |
    | 200 mm   | 46 px  | 30 px  | 18 px  |
    | 300 mm   | 68 px  | 46 px  | 27 px  |

Anything under ~25 px is unreliable. 300 mm is the sensible default for
station markers you want seen from across the cell; 200 mm is fine for a tag
the robot walks right up to.

The number pose estimation needs
--------------------------------
`--size-mm` is the side of the **black square's outer edge** -- not the paper,
not including the white margin. That is the length AprilTag/OpenCV pose
estimation expects, and it is printed on each sheet so it cannot be lost.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from r1vision.venv_guard import ensure_venv  # noqa: E402

ensure_venv(("cv2",))

import cv2  # noqa: E402
import numpy as np  # noqa: E402

DPI = 300
PAGES_MM = {"A4": (210.0, 297.0), "A3": (297.0, 420.0)}

# The route this is for: one tag per station, so the robot always has a metric
# reference in view at the points where being wrong actually costs something.
STATIONS = {0: "LARA", 1: "DOOR", 2: "MAIRA"}


def mm_to_px(mm: float, dpi: int = DPI) -> int:
    return int(round(mm / 25.4 * dpi))


def build_sheet(tag_id: int, size_mm: float, label: str, dpi: int = DPI):
    """One page: tag at exact physical size, quiet zone, label, scale bar."""
    adict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_APRILTAG_36H11)

    tag_px = mm_to_px(size_mm, dpi)
    tag = cv2.aruco.generateImageMarker(adict, tag_id, tag_px)
    tag = cv2.cvtColor(tag, cv2.COLOR_GRAY2BGR)

    # AprilTag needs white space around the black border or the detector cannot
    # find the edge. One module is the documented minimum, and that is all this
    # reserves -- the rest of the page is white anyway, so the effective quiet
    # zone is far larger than the spec asks for.
    modules = adict.markerSize + 2          # 6 data + 1 border each side
    quiet_px = int(round(tag_px / modules))

    # Pick the smallest page the tag plus its quiet zone actually fits on,
    # rather than silently cropping a 300 mm tag onto A4.
    quiet_mm = size_mm / modules
    need_mm = size_mm + 2 * quiet_mm + 30.0   # + margin for text
    page = "A4" if need_mm <= PAGES_MM["A4"][0] else "A3"
    pw_mm, ph_mm = PAGES_MM[page]
    if need_mm > pw_mm:
        # Invert need_mm = size*(1 + 2/modules) + 30 for an honest suggestion.
        fits = int((pw_mm - 30.0) / (1 + 2.0 / modules))
        raise SystemExit(
            f"a {size_mm:g} mm tag does not fit on {page} "
            f"({pw_mm:g} mm wide). Largest that fits: --size-mm {fits}. "
            f"For anything bigger, print tiled across sheets or use a plotter.")

    W, H = mm_to_px(pw_mm, dpi), mm_to_px(ph_mm, dpi)
    sheet = np.full((H, W, 3), 255, np.uint8)

    block = tag_px + 2 * quiet_px
    x0 = (W - block) // 2
    y0 = mm_to_px(35.0, dpi)
    sheet[y0 + quiet_px:y0 + quiet_px + tag_px,
          x0 + quiet_px:x0 + quiet_px + tag_px] = tag

    # Corner ticks marking the black square's true extent, so the printed size
    # can be verified with a ruler against the number below.
    tx0, ty0 = x0 + quiet_px, y0 + quiet_px
    tick = mm_to_px(4.0, dpi)
    gap = mm_to_px(3.0, dpi)
    for cx, cy, dx, dy in ((tx0, ty0, -1, -1), (tx0 + tag_px, ty0, 1, -1),
                           (tx0, ty0 + tag_px, -1, 1),
                           (tx0 + tag_px, ty0 + tag_px, 1, 1)):
        cv2.line(sheet, (cx + dx * gap, cy), (cx + dx * (gap + tick), cy),
                 (0, 0, 0), 2)
        cv2.line(sheet, (cx, cy + dy * gap), (cx, cy + dy * (gap + tick)),
                 (0, 0, 0), 2)

    def text(s, y_mm, scale=1.1, thick=2):
        (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
        cv2.putText(sheet, s, ((W - tw) // 2, mm_to_px(y_mm, dpi)),
                    cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick,
                    cv2.LINE_AA)

    text(f"{label}", 18.0, 1.6, 3)
    text(f"tag36h11   id={tag_id}   size={size_mm:g} mm "
         f"(black square, corner ticks)", 27.0, 0.9, 2)

    # Print-scale check. A printer set to "fit to page" silently shrinks
    # everything by a few percent; every distance the robot computes would
    # then be wrong by that factor, consistently, in a way that looks like a
    # calibration error rather than a printing mistake.
    bar_mm = 100.0
    bar_px = mm_to_px(bar_mm, dpi)
    by = y0 + block + mm_to_px(18.0, dpi)
    bx = (W - bar_px) // 2
    cv2.line(sheet, (bx, by), (bx + bar_px, by), (0, 0, 0), 3)
    for ex in (bx, bx + bar_px):
        cv2.line(sheet, (ex, by - mm_to_px(3, dpi)),
                 (ex, by + mm_to_px(3, dpi)), (0, 0, 0), 3)
    s = "this line must measure exactly 100 mm - if not, the printer rescaled"
    (tw, _), _ = cv2.getTextSize(s, cv2.FONT_HERSHEY_SIMPLEX, 0.75, 2)
    cv2.putText(sheet, s, ((W - tw) // 2, by + mm_to_px(12, dpi)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 0, 0), 2, cv2.LINE_AA)

    return sheet, page


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--ids", type=int, nargs="+", default=sorted(STATIONS),
                    help="tag ids to generate (default: the route stations)")
    ap.add_argument("--size-mm", type=float, default=200.0,
                    help="side of the black square in mm (default 200, the "
                         "largest that fits A3 comfortably)")
    ap.add_argument("--out", default="tags", help="output directory")
    ap.add_argument("--dpi", type=int, default=DPI)
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)

    for tag_id in args.ids:
        label = STATIONS.get(tag_id, f"TAG {tag_id}")
        sheet, page = build_sheet(tag_id, args.size_mm, label, args.dpi)
        path = out / f"tag36h11_id{tag_id:02d}_{args.size_mm:g}mm_{page}.png"
        cv2.imwrite(str(path), sheet)
        print(f"  {path}   [{page}, print at 100%]")

    print(f"\n  {len(args.ids)} sheets -> {out.resolve()}")
    print("  Print at 100% / Actual size, then measure the 100 mm line.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
