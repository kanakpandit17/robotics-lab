"""DDS types this robot publishes that unitree_sdk2py 1.0.1 does not ship.

The installed Python SDK predates the R1's firmware. `dds_topics.py` shows the
robot publishing three types the SDK has never heard of, and DDS will not
deliver a sample across a type mismatch -- so those topics look *silent* rather
than undecodable, which is a considerably more misleading failure. That is
exactly how `rt/sportmodestate` was first written off as dead: it was probed
with the quadruped `unitree_go` type while the robot publishes the humanoid
`unitree_hg` one.

Defined here rather than by upgrading unitree_sdk2py, because the VR viewer,
arm gestures and locomotion code in this repository are all bound against 1.0.1
and an upgrade would put working things at risk to fix a diagnostic.

Field layout is transcribed from Unitree's own C++ IDL, not guessed:
    unitree_sdk2/include/unitree/idl/hg/SportModeState_.hpp

Note what it does NOT contain: any position. On the humanoid this topic is pure
FSM state. Odometry lives elsewhere -- and on this robot, nowhere we can read.

Do not add `from __future__ import annotations` here. It turns the field
annotations into strings, and cyclonedds resolves them by name against this
module's globals at class-creation time, failing with "Type types.uint32 ...
cannot be resolved". The SDK's own generated files omit it for this reason.
"""

from dataclasses import dataclass

import cyclonedds.idl as idl
import cyclonedds.idl.annotations as annotate
import cyclonedds.idl.types as types


@dataclass
@annotate.final
@annotate.autoid("sequential")
class SportModeState_(
        idl.IdlStruct, typename="unitree_hg.msg.dds_.SportModeState_"):
    """Locomotion FSM state -- `rt/sportmodestate` and `rt/lf/sportmodestate`.

    fsm_id / fsm_mode are the pair that decide whether an external SetVelocity
    is accepted or rejected with 127. Watch them while driving with the
    handheld controller to see which state the built-in controller occupies.
    """

    fsm_id: types.uint32
    fsm_mode: types.uint32
    task_id: types.uint32
    task_time: types.float32
