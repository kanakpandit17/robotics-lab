"""Staircase 3 of the ARENA2036 shopfloor as a MuJoCo scene (scene_staircase3.xml), with the R1.

    python stairs-r1/staircase3.py            # writes scene_staircase3.xml

Measured from imported_cad/arena_full.usd (shopfloor frame, floor top at Z = 0.133):
  16 tread plates, 1.05 m wide (Y -8.762..-7.712), front edges every 22.35 cm toward -X starting
  at X = -31.722, tops from Z 0.317 to 2.876 (17.05 cm rise); landing Part 546 at Z 3.113,
  X -36.986..-35.386; platform Part 889, 7.6 x 3.0 m, north of the flight, same height.

Two deliberate departures from the CAD, both stated where a reader will see them:
  * the CAD treads are 8.8 cm deep plates (open risers, a modelling shortcut); no foot fits on
    that, so each step here is a SOLID block spanning the full 22.35 cm going, at the CAD height;
  * nothing else -- the uneven last riser (2.876 -> 3.113 = 23.7 cm, vs 17.05 cm below) is kept.

Local frame: +x = up the stairs, x = 0 at the first step's front edge, y = 0 on the flight's
centre line, z = 0 on the floor. Shopfloor <- local:  X = -31.722 - x,  Y = -8.237 - y,  Z = z + 0.133
(a 180 deg turn about z, so the flight still climbs toward -X in the shopfloor).
"""
from pathlib import Path

HERE = Path(__file__).resolve().parent
FLOOR_Z = 0.133

# CAD tread tops (shopfloor Z), bottom to top
TREAD_TOPS = [0.317, 0.487, 0.658, 0.829, 0.999, 1.170, 1.340, 1.511,
              1.682, 1.852, 2.023, 2.193, 2.364, 2.535, 2.705, 2.876]
GOING = 0.2235
WIDTH = 1.05
LANDING_TOP = 3.113
LANDING_LEN, LANDING_W = 1.6, 1.15            # Part 546
# Part 889, X -38.872..-31.278, Y -7.662..-4.662. Shopfloor north (larger Y) is NEGATIVE local y.
PLATFORM = dict(x0=-0.444, x1=7.150, y0=-3.575, y1=-0.575)

STEP_TOPS = [z - FLOOR_Z for z in TREAD_TOPS]               # local
TOP_Z = LANDING_TOP - FLOOR_Z                               # 2.980
STAIR_LEN = GOING * len(STEP_TOPS)                          # 3.576
START = (-1.0, 0.0)                                         # robot spawn, facing +x


def box(name, x0, x1, y0, y1, z0, z1, rgba="0.62 0.64 0.68 1", collide=True):
    c = f'contype="1" conaffinity="1"' if collide else 'contype="0" conaffinity="0"'
    return (f'    <geom name="{name}" type="box" pos="{(x0 + x1) / 2:.4f} {(y0 + y1) / 2:.4f} {(z0 + z1) / 2:.4f}" '
            f'size="{(x1 - x0) / 2:.4f} {(y1 - y0) / 2:.4f} {(z1 - z0) / 2:.4f}" rgba="{rgba}" {c}/>')


def world_geoms():
    g = []
    hw = WIDTH / 2
    for i, top in enumerate(STEP_TOPS):
        shade = "0.93 0.48 0.12 1" if i % 2 == 0 else "0.85 0.42 0.10 1"
        g.append(box(f"step{i:02d}", i * GOING, (i + 1) * GOING, -hw, hw, 0.0, top, shade))
    # landing (Part 546): 1.15 m wide, starts where the last going ends
    g.append(box("landing", STAIR_LEN, STAIR_LEN + LANDING_LEN, -0.575, 0.575, TOP_Z - 0.10, TOP_Z, "0.55 0.57 0.62 1"))
    g.append(box("landing_support", STAIR_LEN, STAIR_LEN + LANDING_LEN, -0.575, 0.575, 0.0, TOP_Z - 0.10, "0.40 0.42 0.46 1"))
    # platform (Part 889), north of the flight
    g.append(box("platform", PLATFORM["x0"], PLATFORM["x1"], PLATFORM["y0"], PLATFORM["y1"], TOP_Z - 0.20, TOP_Z, "0.55 0.57 0.62 1"))
    # wall of the room under the platform, right beside the flight (Parts 1830/1648/724...)
    g.append(box("north_wall", PLATFORM["x0"], STAIR_LEN, -0.646, -0.575, 0.0, TOP_Z - 0.20, "0.75 0.76 0.80 1"))
    # south handrail (balusters ~0.93 m) -- visual only, it's at arm height and arms don't collide
    # fromto, not euler: r1.xml (from the URDF) sets compiler angle="radian", so degrees would be misread
    y, rail = hw + 0.06, 0.93
    g.append(f'    <geom name="handrail" type="capsule" size="0.025" '
             f'fromto="{GOING / 2:.4f} {y:.4f} {STEP_TOPS[0] + rail:.4f} {STAIR_LEN - GOING / 2:.4f} {y:.4f} {STEP_TOPS[-1] + rail:.4f}" '
             f'rgba="0.95 0.8 0.2 1" contype="0" conaffinity="0" group="2"/>')   # group 2: height-map rays skip it
    return "\n".join(g)


def scene_xml(robot="r1.xml"):
    return f"""<mujoco model="r1_staircase3">
  <include file="{robot}"/>
  <statistic center="1.5 0 1.2" extent="5"/>
  <visual><global offwidth="1280" offheight="720"/></visual>
  <visual>
    <headlight diffuse="0.6 0.6 0.6" ambient="0.35 0.35 0.35" specular="0 0 0"/>
    <global azimuth="-130" elevation="-20"/>
  </visual>
  <asset>
    <texture name="grid" type="2d" builtin="checker" width="512" height="512" rgb1="0.22 0.24 0.27" rgb2="0.28 0.30 0.33"/>
    <material name="grid" texture="grid" texrepeat="1 1" texuniform="true"/>
  </asset>
  <worldbody>
    <light pos="0 0 6" dir="0 0 -1" directional="true"/>
    <geom name="floor" type="plane" size="0 0 0.05" material="grid" contype="1" conaffinity="1"/>
{world_geoms()}
  </worldbody>
</mujoco>
"""


if __name__ == "__main__":
    out = HERE / "scene_staircase3.xml"
    out.write_text(scene_xml())
    print(f"wrote {out}: {len(STEP_TOPS)} steps, rise {STEP_TOPS[1] - STEP_TOPS[0]:.4f} m, going {GOING} m, "
          f"top riser {TOP_Z - STEP_TOPS[-1]:.3f} m, landing at {TOP_Z:.3f} m")
