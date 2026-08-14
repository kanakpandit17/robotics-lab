"""Arm kinematics for the Unitree R1, read from the robot's own URDF.

    python -m r1vision.arm_kinematics      # self-check, no robot required

Everything here comes from `unitree_r1/R1.urdf` -- link offsets, joint axes,
mounting tilts and joint limits are parsed at import rather than hand-entered,
so there is nothing to keep in sync and nothing guessed.

Why this replaces the earlier closed-form version
--------------------------------------------------
The first version of this module assumed a spherical shoulder (three axes
meeting at a point) and split the datasheet's combined arm reach evenly into
two 210 mm segments. The URDF shows both assumptions are wrong:

  * the shoulder axes are offset, not intersecting -- pitch to roll is 64 mm,
    roll to yaw another 114 mm -- and the pitch joint carries a -15 deg
    mounting tilt (`rpy=-0.26187`)
  * the real segments are 88.5 mm (yaw->elbow) and 116.7 mm (elbow->wrist),
    roughly half what was assumed

A closed-form spherical solve cannot represent that arm, so position IK is
now numerical: damped least squares on the analytic Jacobian, which handles
the offset axes, honours the real joint limits, and degrades gracefully near
singularities instead of jumping.

The end effector is the wrist-roll joint origin -- the last link in the URDF,
since this robot has no hand fitted. Wrist roll therefore spins about a point
and does not move the end effector at all: position is controlled by the four
joints before it, and wrist roll is passed straight through as orientation.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np

URDF_PATH = Path(__file__).resolve().parent.parent / "unitree_r1" / "R1.urdf"

# The chain, shoulder outward. Order matches r1_arm.py's joint index lists.
CHAIN = ("shoulder_pitch", "shoulder_roll", "shoulder_yaw", "elbow", "wrist_roll")

# Wrist roll spins about the end-effector point, so it cannot contribute to
# position. Only the first four joints enter the IK.
N_POSITION_JOINTS = 4


def _rpy(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """URDF fixed-axis roll-pitch-yaw -> rotation matrix (R = Rz @ Ry @ Rx)."""
    cr, sr = math.cos(roll), math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw), math.sin(yaw)
    return np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp,     cp * sr,                cp * cr],
    ])


def _axis_angle(axis: np.ndarray, q: float) -> np.ndarray:
    """Rodrigues rotation about a unit axis."""
    k = axis / np.linalg.norm(axis)
    K = np.array([[0, -k[2], k[1]], [k[2], 0, -k[0]], [-k[1], k[0], 0]])
    return np.eye(3) + math.sin(q) * K + (1 - math.cos(q)) * (K @ K)


class ArmChain:
    """One arm's kinematics, parsed from the URDF."""

    def __init__(self, side: str = "right", urdf: Path | str = URDF_PATH):
        if side not in ("left", "right"):
            raise ValueError(f"side must be left or right, got {side!r}")
        urdf = Path(urdf)
        if not urdf.exists():
            raise FileNotFoundError(
                f"{urdf} not found -- the R1 URDF ships in the repo under "
                "unitree_r1/. Without it there is nothing to derive kinematics "
                "from, and hand-entered numbers are what this module exists to "
                "avoid.")
        root = ET.parse(urdf).getroot()
        joints = {j.get("name"): j for j in root.findall("joint")}

        self.side = side
        self.names, self.origins, self.rots, self.axes = [], [], [], []
        lo, hi = [], []
        for part in CHAIN:
            name = f"{side}_{part}_joint"
            j = joints.get(name)
            if j is None:
                raise KeyError(f"{name} missing from {urdf}")
            o = j.find("origin")
            xyz = [float(v) for v in (o.get("xyz") or "0 0 0").split()]
            rpy = [float(v) for v in (o.get("rpy") or "0 0 0").split()]
            ax = [float(v) for v in j.find("axis").get("xyz").split()]
            lim = j.find("limit")
            self.names.append(name)
            self.origins.append(np.array(xyz))
            self.rots.append(_rpy(*rpy))
            self.axes.append(np.array(ax))
            lo.append(float(lim.get("lower")))
            hi.append(float(lim.get("upper")))
        self.lower = np.array(lo)
        self.upper = np.array(hi)

        # A relaxed pose well inside the limits: arm hanging, slight elbow
        # bend. Used to seed IK and to resolve the redundant DOF, so repeated
        # solves stay near a natural posture instead of wandering.
        self.rest = np.clip(np.array([0.0, -0.2, 0.0, 0.3, 0.0]),
                           self.lower, self.upper)
        self._reach: float | None = None    # filled lazily by reach()

    # ------------------------------------------------------------------ FK
    def frames(self, q) -> list[tuple[np.ndarray, np.ndarray]]:
        """(origin, rotation) of each joint frame, base -> tip, in the
        shoulder-parent (waist) frame."""
        q = np.asarray(q, dtype=float)
        out = []
        p = np.zeros(3)
        R = np.eye(3)
        for i in range(len(self.names)):
            p = p + R @ self.origins[i]
            R = R @ self.rots[i]
            out.append((p.copy(), R.copy()))
            R = R @ _axis_angle(self.axes[i], q[i])
        return out

    def fk(self, q) -> np.ndarray:
        """Joint angles (rad) -> end-effector position, in the waist frame."""
        return self.frames(q)[-1][0]

    def jacobian(self, q) -> np.ndarray:
        """3 x N_POSITION_JOINTS position Jacobian, analytic."""
        fr = self.frames(q)
        p_ee = fr[-1][0]
        J = np.zeros((3, N_POSITION_JOINTS))
        for i in range(N_POSITION_JOINTS):
            p_i, R_i = fr[i]
            axis_w = R_i @ self.axes[i]
            J[:, i] = np.cross(axis_w, p_ee - p_i)
        return J

    def estimate_force(self, q, tau) -> np.ndarray:
        """Estimate the 3D force applied at the end-effector, from joint
        torques -- no force/torque sensor exists on this robot; this is what
        stands in for one.

        The relation is tau = J^T @ F, the standard static-equilibrium
        mapping between joint torque and an external force at the tip.
        N_POSITION_JOINTS (4) > 3, so this is over-determined and solved by
        least squares, not inverted directly. `tau` must be ordered to match
        `jacobian()` -- this arm's first N_POSITION_JOINTS joints; wrist roll
        is excluded because it does not move the end-effector's position, so
        its torque carries no information about a force applied there.

        This is a STATIC relation: it assumes near-zero joint velocity and
        acceleration. It reads a held or slowly-lifted object well; during
        fast motion, joint torque is also fighting inertia and friction, not
        only an external load, and this will overestimate the force.

        It also does not separate "the object" from "the arm's own weight" --
        tau already includes what it costs the arm to hold itself up at this
        pose. The caller gets an object-only estimate by taking two readings
        at the same pose (empty hand, then holding the object) and
        subtracting the tau before calling this, not by calling it twice and
        subtracting the resulting forces -- J varies with q, so subtracting
        force estimates from two different poses does not cancel the same
        way subtracting torques at ONE pose does.
        """
        J = self.jacobian(q)
        tau = np.asarray(tau, dtype=float)[:N_POSITION_JOINTS]
        force, *_ = np.linalg.lstsq(J.T, tau, rcond=None)
        return force

    # ------------------------------------------------------------------ IK
    def _solve_from(self, target, q0, iters: int, tol: float):
        """One damped-least-squares descent from a given seed."""
        q = q0.copy()
        damping = 0.20      # start heavily damped: stable but slow
        best_q, best_err = q.copy(), float(np.linalg.norm(target - self.fk(q)))
        for _ in range(iters):
            err_v = target - self.fk(q)
            err = float(np.linalg.norm(err_v))
            if err < best_err:
                best_q, best_err = q.copy(), err
            if err < tol:
                break
            J = self.jacobian(q)
            JT = J.T
            dq = JT @ np.linalg.solve(J @ JT + (damping ** 2) * np.eye(3), err_v)
            q[:N_POSITION_JOINTS] = np.clip(
                q[:N_POSITION_JOINTS] + dq, self.lower[:N_POSITION_JOINTS],
                self.upper[:N_POSITION_JOINTS])
            # Levenberg-Marquardt style: ease off the damping as the error
            # falls, so the tail of the solve converges tightly instead of
            # crawling. Heavy damping only earns its keep far from the target.
            damping = max(0.01, damping * 0.85)
        return best_q, best_err

    def ik(self, target, q_init=None, wrist_roll: float = 0.0,
          iters: int = 100, tol: float = 1e-4, restarts: int = 4):
        """Target position -> joint angles, by damped least squares.

        Returns (q, position_error_m). The error is reported rather than
        raised on: an unreachable target yields the closest pose the arm can
        strike, which is the sane thing for teleoperation -- the operator sees
        the arm stretch toward their hand and stop, instead of the solver
        refusing and the arm freezing.

        Seeds from `q_init` (the arm's current pose during teleop, which makes
        successive solves both fast and continuous), falling back to the rest
        pose. Extra random restarts are only spent if that seed stalls: a
        4-DOF arm with asymmetric limits has local minima that a single
        descent can sit in, and a stalled solve during teleop reads as the arm
        mysteriously refusing to follow your hand.
        """
        target = np.asarray(target, dtype=float)
        seed = (self.rest.copy() if q_init is None
                else np.clip(np.asarray(q_init, dtype=float).copy(),
                             self.lower, self.upper))

        best_q, best_err = self._solve_from(target, seed, iters, tol)
        if best_err > tol:
            rng = np.random.default_rng(0)
            for _ in range(restarts):
                q0 = seed.copy()
                q0[:N_POSITION_JOINTS] = rng.uniform(
                    self.lower[:N_POSITION_JOINTS],
                    self.upper[:N_POSITION_JOINTS])
                q, err = self._solve_from(target, q0, iters, tol)
                if err < best_err:
                    best_q, best_err = q, err
                if best_err < tol:
                    break

        best_q[4] = float(np.clip(wrist_roll, self.lower[4], self.upper[4]))
        return best_q, best_err

    # ----------------------------------------------------------- workspace
    def reach(self) -> float:
        """Farthest the end effector gets from the shoulder-pitch origin.

        Sampled once and cached: it is a fixed property of the geometry, and
        the 4000-sample sweep costs ~180 ms. Recomputing it per control cycle
        (which an earlier version did, via clamp_reachable) dropped the 100 Hz
        loop to 4.5 Hz.
        """
        if self._reach is None:
            best = 0.0
            rng = np.random.default_rng(0)
            for _ in range(4000):
                q = rng.uniform(self.lower, self.upper)
                best = max(best, float(np.linalg.norm(self.fk(q))))
            self._reach = best
        return self._reach

    def clamp_reachable(self, target, margin: float = 0.02) -> np.ndarray:
        """Pull a target inside the arm's maximum reach radius.

        This is a coarse guard against absurd targets (a hand waved metres
        away), NOT a reachability test. The reachable set is not a sphere --
        joint limits carve large holes out of it, so plenty of targets inside
        this radius still cannot be struck. The honest signal for "the arm
        cannot go where you are pointing" is the residual `ik` returns, which
        the teleop HUD shows as `ik_err_mm`.
        """
        p = np.asarray(target, dtype=float)
        d = float(np.linalg.norm(p))
        hi = self.reach() - margin
        if d < 1e-9:
            return p
        return p * (hi / d) if d > hi else p


_CACHE: dict[str, ArmChain] = {}


def chain(side: str = "right") -> ArmChain:
    """Cached ArmChain -- parsing the URDF on every control cycle would be
    wasteful, and the geometry never changes at runtime."""
    if side not in _CACHE:
        _CACHE[side] = ArmChain(side)
    return _CACHE[side]


def selftest(verbose: bool = True) -> bool:
    """IK should recover poses that FK says are reachable.

    Targets are generated by running FK on random valid joint configurations,
    so every one is reachable by construction -- a failure means the solver is
    wrong, not that the target was impossible.
    """
    ok = True
    for side in ("right", "left"):
        arm = chain(side)
        rng = np.random.default_rng(1)
        errs = []
        for _ in range(300):
            q_true = rng.uniform(arm.lower, arm.upper)
            target = arm.fk(q_true)
            q_sol, err = arm.ik(target)
            errs.append(err)
            # Joint limits must hold no matter what was asked for.
            if np.any(q_sol < arm.lower - 1e-9) or np.any(q_sol > arm.upper + 1e-9):
                ok = False
                if verbose:
                    print(f"  FAIL {side}: solution outside joint limits")
        errs = np.array(errs)
        solved = float((errs < 1e-3).mean())
        if verbose:
            print(f"  {side:5s}  reach={arm.reach():.3f} m   "
                 f"median err={np.median(errs) * 1000:.3f} mm   "
                 f"95th={np.percentile(errs, 95) * 1000:.2f} mm   "
                 f"solved<1mm: {solved * 100:.0f}%")
        if solved < 0.95:
            ok = False

        # estimate_force round-trip: pick a random joint pose and a random
        # "true" force, compute the torques that force WOULD produce
        # (tau = J^T @ F, the physically correct forward direction -- this
        # is not circular, it is the same relation run the other way), then
        # check estimate_force recovers the force it started from.
        force_errs = []
        for _ in range(300):
            q = rng.uniform(arm.lower, arm.upper)
            f_true = rng.uniform(-20.0, 20.0, size=3)   # newtons
            tau = arm.jacobian(q).T @ f_true
            f_est = arm.estimate_force(q, tau)
            force_errs.append(float(np.linalg.norm(f_est - f_true)))
        force_errs = np.array(force_errs)
        if verbose:
            print(f"  {side:5s}  estimate_force round-trip: "
                 f"median err={np.median(force_errs):.4f} N   "
                 f"max={np.max(force_errs):.4f} N")
        if np.max(force_errs) > 1e-6:
            ok = False
            if verbose:
                print(f"  FAIL {side}: estimate_force did not recover a "
                     "known force from its own torques")
    return ok


if __name__ == "__main__":
    import sys

    print(f"arm_kinematics selftest -- from {URDF_PATH.name}\n")
    arm = chain("right")
    print("  right arm chain (from URDF):")
    for n, o, l, u in zip(arm.names, arm.origins, arm.lower, arm.upper):
        print(f"    {n:32s} offset={np.round(o, 4)}  "
             f"limit={math.degrees(l):+7.1f}..{math.degrees(u):+7.1f} deg")
    print()
    good = selftest()
    print("\n  " + ("PASS" if good else "FAIL"))
    sys.exit(0 if good else 1)
