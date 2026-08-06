"""NEURA MAiRA, driven through the NEURA SDK on the workstation.

This is the only file in the project that imports anything NEURA, and it runs
only on the machine with the ethernet cable in it. If NEURA's stack starts ROS
internally, it starts here and goes no further -- the laptop's dependency list
never learns that ROS exists.

Bound against neurapy 5.1.85 / maira7M, from the API dump in the notes below.

    Robot()                        defaults come from robot_overview.yaml
    move_joint(target, speed=, acceleration=, enable_safety=)
                                   target is radians OR the name of a taught
                                   point, so MoveHome is move_joint("Home")
    get_point(name, representation="Joint"|"Cartesian")
    pause() / unpause()            halt and resume motion
    stop()                         see "Why Stop pauses" below
    motion_status()                'NOT_RUNNING' | 'RUNNING' | 'PAUSED'
    get_errors() / reset_errors()
    switch_to_automatic_mode(), is_robot_in_automatic_mode()

Why Stop pauses rather than calling stop()
------------------------------------------
neurapy's `stop()` is documented as "stop the robot's motion **and terminate the
script execution** ... needs to be added at the end of the script (only once)".
That is a teardown call, not an emergency halt: using it for the fleet's Stop
would end the SDK session the agent depends on, and the next command would find
a dead handle.

So Stop calls `pause()`, which halts motion and is reversible, and latches the
agent into ESTOP so nothing moves until ResetError. `stop()` is reserved for
disconnect, which is exactly what the docstring describes. If a genuinely
destructive halt is ever wanted, that belongs on the physical e-stop, not here.

Cancellation, honestly
----------------------
`move_joint` blocks until the motion finishes and offers no progress callback,
so a CancelToken cannot interrupt one mid-flight. Stop works by calling
`pause()` from the transport thread, which acts on the robot directly. Between
steps of a multi-step routine the token is checked as usual. This is a property
of the vendor API, not something the fleet layer can paper over.
"""

from __future__ import annotations

import contextlib
import io
import logging
import os
from typing import Any, Callable

from fleetcore.errors import ErrorCode, RobotError

from .base import CancelToken, RobotAdapter

log = logging.getLogger(__name__)

# neurapy maps its own failures onto these names. Matching by name rather than
# by imported class keeps this file importable when the SDK is absent, which is
# what lets the simulator and the unit tests run anywhere.
_ERROR_MAP = {
    "WrongMode": (ErrorCode.NOT_READY,
                  "robot is in Teach mode; switch it to Automatic"),
    "UnfeasibleMotion": (ErrorCode.MOTION_FAILED,
                         "motion is not reachable with these parameters"),
    "ConnectionError": (ErrorCode.SDK_UNAVAILABLE,
                        "lost the connection to the robot"),
    "InterruptedError": (ErrorCode.CANCELLED, "motion was interrupted"),
}


@contextlib.contextmanager
def _quiet_import():
    """Silence neurapy's import-time chatter.

    Importing neurapy prints robot_overview details and starts a socket.io
    watchdog thread that tries to reach the teach-pendant GUI, throwing a
    ConnectionError traceback when the GUI is not running. None of that is a
    fault, but in the agent's log it reads exactly like one.
    """
    buf = io.StringIO()
    try:
        with contextlib.redirect_stdout(buf):
            yield
    finally:
        text = buf.getvalue().strip()
        if text:
            log.debug("neurapy import output: %s", text.replace("\n", " | ")[:400])


class MairaAdapter(RobotAdapter):
    model = "NEURA MAiRA"

    def __init__(self, robot_id: str, config: dict | None = None):
        super().__init__(robot_id, config)
        self._sdk: Any = None
        self._simulate = bool(self.config.get("simulate", False))
        self._estopped = False
        self._skills: dict[str, Any] = {}
        # Why each telemetry field is missing, when it is.
        self._read_errors: dict[str, str] = {}
        # Telemetry and commands travel by different transports on this cell:
        # the SDK's CORBA port is closed, rosbridge is not.
        self._ros = None

    # ------------------------------------------------------------- lifecycle
    def _start_ros(self) -> None:
        cfg = self.config.get("ros_bridge")
        if not cfg or self._ros is not None:
            return
        from agent.ros_telemetry import RosBridgeTelemetry
        self._ros = RosBridgeTelemetry(
            cfg["host"], int(cfg.get("port", 9090)),
            throttle_ms=int(cfg.get("throttle_ms", 100)))
        self._ros.start()
        log.info("%s telemetry via rosbridge %s:%s", self.robot_id,
                 cfg["host"], cfg.get("port", 9090))

    def connect(self) -> None:
        self._start_ros()
        if self._simulate:
            from .simulator import StubArm
            self._sdk = StubArm(self.model)
            self._connected = True
            log.warning("%s running SIMULATED -- no hardware is being commanded",
                        self.robot_id)
            return

        try:
            with _quiet_import():
                from neurapy.robot import Robot

                # Positional args are only accepted on Ubuntu 18/20; elsewhere
                # the constructor reads robot_overview.yaml. Passing them only
                # when configured keeps both cases working.
                args = self.config.get("sdk_args")
                self._sdk = Robot(*args) if args else Robot()

            # Taking the robot out of Teach mode is a physical act -- someone
            # may be hand-guiding it at that moment -- so it is opt-in. The
            # agent otherwise observes whatever mode the operator chose and
            # refuses motion commands until they switch it themselves.
            if not self._sdk.is_robot_in_automatic_mode():
                if self.config.get("auto_switch_to_automatic", False):
                    log.info("switching %s to automatic mode", self.robot_id)
                    if not self._sdk.switch_to_automatic_mode():
                        raise RobotError(
                            ErrorCode.NOT_READY,
                            "robot would not switch to Automatic mode",
                        )
                else:
                    log.warning(
                        "%s is NOT in automatic mode -- connecting read-only. "
                        "Motion commands will be refused until it is switched "
                        "on the pendant (or set auto_switch_to_automatic)",
                        self.robot_id)

            # A configured ceiling applied once at connect, so a scheduler bug
            # cannot outrun it later. set_joint_speed takes 0-100.
            cap = self.config.get("joint_speed_percent")
            if cap is not None:
                self._sdk.set_joint_speed(float(cap))

            self._connected = True
            log.info("%s connected: %s, %s dof, mode=%s",
                     self.robot_id,
                     getattr(self._sdk, "robot_name", "?"),
                     getattr(self._sdk, "dof", "?"),
                     self._safe(self._sdk.motion_status, "?"))
        except RobotError:
            self._connected = False
            raise
        except Exception as exc:
            self._connected = False
            raise RobotError(
                ErrorCode.SDK_UNAVAILABLE,
                f"cannot reach MAiRA via NEURA SDK: {type(exc).__name__}: {exc}",
                {"robot_id": self.robot_id},
            ) from exc

    def disconnect(self) -> None:
        try:
            if self._sdk is not None and hasattr(self._sdk, "stop"):
                # Here stop() means what its docstring says: end the session.
                self._sdk.stop()
        except Exception:
            log.exception("error closing NEURA SDK (continuing)")
        finally:
            self._sdk = None
            self._skills.clear()
            self._connected = False

    # ------------------------------------------------------------ capability
    def capabilities(self) -> dict[str, Callable[..., Any]]:
        return {
            "MoveHome": self._move_home,
            "MoveToPoint": self._move_to_point,
            "PickObject": self._pick_object,
            "PlaceObject": self._place_object,
            "StartAssembly": self._start_assembly,
            "ResetError": self._reset_error,
        }

    # ------------------------------------------------------------- commands
    def _move_home(self, params: dict, cancel: CancelToken) -> dict:
        self._guard()
        target = self.config.get("home_point", "Home")
        self._move_joint(target, params)
        return {"pose": target, "joints": self._joints()}

    def _move_to_point(self, params: dict, cancel: CancelToken) -> dict:
        """Move to a point taught on the pendant, by name."""
        self._guard()
        name = params.get("point") or params.get("target")
        if not name:
            raise RobotError(ErrorCode.INVALID_PARAMS,
                             "MoveToPoint needs a 'point' name")
        self._move_joint(self._resolve(name), params)
        return {"pose": name, "joints": self._joints()}

    def _pick_object(self, params: dict, cancel: CancelToken) -> dict:
        self._guard()
        target = params.get("object") or params.get("target")
        if not target:
            raise RobotError(ErrorCode.INVALID_PARAMS,
                             "PickObject needs an 'object' parameter")

        skill = self._pick_skill()
        if skill is not None:
            cfg = self.config["skills"]["pick"]
            result = skill.execute(
                workspace_name=cfg["workspace_name"],
                gripper_name=cfg["gripper_name"],
                pick_type=cfg.get("pick_type", "top"),
                look_at_point=cfg["look_at_point"],
                object_names=[target],
                bin_name=cfg.get("bin_name"),
                use_path_planning=cfg.get("use_path_planning", True),
            )
            # Place needs Pick's output, so it is kept for the next command.
            self._skills["last_pick"] = result
            return {"picked": target, "holding": True, "vision": True}

        # Taught-point fallback: go to the point, close the gripper.
        self._move_joint(self._resolve(target), params)
        self._grip(True)
        return {"picked": target, "holding": True, "vision": False}

    def _place_object(self, params: dict, cancel: CancelToken) -> dict:
        self._guard()
        target = params.get("location") or params.get("target")
        if not target:
            raise RobotError(ErrorCode.INVALID_PARAMS,
                             "PlaceObject needs a 'location' parameter")

        skill = self._place_skill()
        if skill is not None:
            last = self._skills.get("last_pick")
            if last is None:
                raise RobotError(
                    ErrorCode.NOT_READY,
                    "the Place skill needs the output of a preceding "
                    "PickObject; none has run since this agent started",
                )
            skill.set_pick_params(last)
            skill.execute()
            self._skills.pop("last_pick", None)
            return {"placed_at": target, "holding": False, "vision": True}

        self._move_joint(self._resolve(target), params)
        self._grip(False)
        return {"placed_at": target, "holding": False, "vision": False}

    def _start_assembly(self, params: dict, cancel: CancelToken) -> dict:
        """A named multi-step routine, run as one fleet command.

        Cancellation is checked between steps rather than only at the start:
        an assembly is the longest thing this robot does and the most likely to
        need interrupting halfway.
        """
        self._guard()
        routine = params.get("routine", "default")

        program = self.config.get("programs", {}).get(routine)
        if program:
            # A routine taught on the pendant runs as one SDK call.
            self._call(self._sdk.execute_program, program)
            return {"routine": routine, "program": program}

        steps = self.config.get("assemblies", {}).get(routine)
        if steps is None:
            raise RobotError(
                ErrorCode.INVALID_PARAMS,
                f"no assembly routine named {routine!r}",
                {"known": sorted({**self.config.get("assemblies", {}),
                                  **self.config.get("programs", {})})},
            )
        done = []
        for step in steps:
            cancel.raise_if_cancelled()
            self.execute(step["command"], step.get("params", {}), cancel)
            done.append(step["command"])
        return {"routine": routine, "steps_completed": done}

    def _reset_error(self, params: dict, cancel: CancelToken) -> dict:
        """Clear a latched stop. Deliberately explicit -- never automatic."""
        errors_before = self._safe(self._sdk.get_errors, [])
        for method in ("reset_errors", "reset_warnings", "reset_collision"):
            fn = getattr(self._sdk, method, None)
            if fn is not None:
                self._safe(fn, None)
        # Motion was halted with pause(), so it has to be released explicitly.
        if self._safe(self._sdk.motion_status, "") == "PAUSED":
            self._safe(self._sdk.unpause, None)
        self._estopped = False
        return {"estop": False, "cleared": errors_before,
                "errors": self._safe(self._sdk.get_errors, [])}

    # ------------------------------------------------------------------ stop
    def stop(self) -> None:
        """Called from the transport thread while a motion may be running."""
        self._estopped = True
        try:
            if self._sdk is not None:
                # pause(), not stop() -- see the module docstring.
                self._sdk.pause()
        except Exception:
            # A stop that raises must not prevent the state machine latching.
            log.exception("SDK pause raised; robot state latched to ESTOP anyway")

    # -------------------------------------------------------------- internals
    def _move_joint(self, target, params: dict) -> None:
        """One joint-space move, with fleet error translation.

        `target` is either a list of joint angles in radians or the name of a
        point taught on the pendant; neurapy accepts both in the same argument.
        """
        speed = self._speed(params)
        ok = self._call(
            self._sdk.move_joint,
            target,
            speed=speed,
            acceleration=float(self.config.get("acceleration", speed)),
            enable_safety=bool(self.config.get("enable_safety", True)),
        )
        # move_joint reports failure by return value as well as by raising.
        if ok is False:
            raise RobotError(
                ErrorCode.MOTION_FAILED,
                f"move_joint to {target!r} returned False",
                {"errors": self._safe(self._sdk.get_errors, [])},
            )

    def _call(self, fn, *args, **kwargs):
        """Invoke an SDK method, mapping its exceptions into fleet errors."""
        try:
            return fn(*args, **kwargs)
        except Exception as exc:
            name = type(exc).__name__
            code, message = _ERROR_MAP.get(
                name, (ErrorCode.SDK_ERROR, f"{name}: {exc}"))
            raise RobotError(code, message, {"sdk_exception": name,
                                             "detail": str(exc)[:300]}) from exc

    def _grip(self, close: bool) -> None:
        """Close or open the tool, whichever call this cell's gripper uses."""
        name = "grasp" if close else "release"
        fn = getattr(self._sdk, name, None)
        if fn is None:
            raise RobotError(ErrorCode.UNSUPPORTED,
                             f"this SDK build has no {name}()")
        args = self.config.get("gripper", {}).get(name, {})
        self._call(fn, **args) if args else self._call(fn)

    def _pick_skill(self):
        return self._skill("pick", "Pick")

    def _place_skill(self):
        return self._skill("place", "Place")

    def _skill(self, key: str, cls_name: str):
        """Lazily build a neurapy_ai_skills object, if one is configured.

        Vision-driven picking is a much shorter path to a working PickObject
        than hand-taught poses, but it needs a workspace, a gripper name and a
        look-at point that only this cell knows -- so it is opt-in via config
        rather than assumed.
        """
        if self._simulate or key not in self.config.get("skills", {}):
            return None
        if key in self._skills:
            return self._skills[key]
        try:
            with _quiet_import():
                module = __import__(
                    f"neurapy_ai_skills.core.{key}.{key}", fromlist=[cls_name])
                skill = getattr(module, cls_name)(robot_handler=self._sdk)
        except Exception as exc:
            raise RobotError(
                ErrorCode.SDK_UNAVAILABLE,
                f"neurapy_ai_skills {cls_name} unavailable: "
                f"{type(exc).__name__}: {exc}",
            ) from exc
        self._skills[key] = skill
        return skill

    def _guard(self) -> None:
        if self._estopped:
            raise RobotError(
                ErrorCode.ESTOP,
                "robot is stopped; send ResetError before commanding motion",
            )

    def _speed(self, params: dict) -> float:
        """Clamp to the configured ceiling. A scheduler cannot exceed it.

        move_joint's own default is 0.25, so this stays on that scale rather
        than the 0-100 one set_joint_speed uses.
        """
        ceiling = float(self.config.get("max_speed", 0.25))
        requested = float(params.get("speed", ceiling))
        return max(0.01, min(requested, ceiling))

    def _resolve(self, name: str):
        """A configured pose, or the name itself for a point taught on the arm."""
        poses = self.config.get("poses", {})
        if name in poses:
            return poses[name]
        if self.config.get("allow_taught_points", True):
            return name           # neurapy looks it up in the point database
        raise RobotError(
            ErrorCode.INVALID_PARAMS,
            f"no pose named {name!r} configured for {self.robot_id}",
            {"known": sorted(poses)},
        )

    def _safe(self, fn, default):
        """Read from the SDK without letting a failure break the caller.

        The reason is kept rather than discarded. Silently returning None turns
        "the robot's CORBA service is unreachable" into a blank field on the
        dashboard, which is the single least useful thing this code could do
        with that information.
        """
        try:
            value = fn()
            self._read_errors.pop(getattr(fn, "__name__", str(fn)), None)
            return value
        except Exception as exc:
            name = getattr(fn, "__name__", str(fn))
            self._read_errors[name] = f"{type(exc).__name__}: {exc}"[:200]
            return default

    def _joints(self):
        return self._safe(self._sdk.get_current_joint_angles, None)

    def telemetry(self) -> dict:
        if self._sdk is None:
            return {"estopped": self._estopped}
        if self._simulate:
            return {"estopped": self._estopped, **self._sdk.telemetry()}
        self._read_errors.clear()
        telemetry = {
            "estopped": self._estopped,
            # The robot's own mode, distinct from the agent's task state. An
            # agent showing IDLE while someone hand-guides the arm is correct
            # but useless; this is the field that says what is really going on.
            "teach_mode": self._safe(self._sdk.is_robot_in_teach_mode, None),
            "automatic_mode": self._safe(
                self._sdk.is_robot_in_automatic_mode, None),
            "motion_status": self._safe(self._sdk.motion_status, None),
            "joints": self._joints(),
            # get_current_cartesian_pose is deprecated since 4.13.0 in favour
            # of get_tcp_pose_quaternion; prefer the new one where it exists.
            "tcp": self._safe(
                getattr(self._sdk, "get_tcp_pose_quaternion", None)
                or self._sdk.get_current_cartesian_pose, None),
            "errors": self._safe(self._sdk.get_errors, None),
            "warnings": self._safe(self._sdk.get_warnings, []),
            "in_collision": self._safe(self._sdk.is_robot_in_collision, None),
        }
        if self._ros is not None:
            ros = self._ros.snapshot()
            # ROS wins where it has an answer: these are live values, while the
            # SDK equivalents are currently None because CORBA is unreachable.
            telemetry.update({k: v for k, v in ros.items() if v is not None})
        if self._read_errors:
            # Surfaced, not logged and forgotten: a field reading None because
            # the robot is unreachable looks identical to one reading None
            # because nothing has moved yet.
            telemetry["read_errors"] = dict(self._read_errors)
        return telemetry
