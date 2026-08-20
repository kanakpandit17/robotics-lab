"""Deterministic step sequence for the LARA -> R1 -> MAiRA -> R1 -> LARA battery-casing loop.

This is intentionally a plain linear state machine, not py_trees: ten steps that always run in
the same order is not a decision problem, so a behaviour tree would add machinery without adding
anything the state machine can't already do.

Today every step advances on internal sim state (arrived-at-target, timer elapsed, reach
complete) checked in r1_lara_maira_loop.py. When this is driven by MQTT instead, the STEP_LABEL
names below map directly onto fleet/v1/<robot_id>/event payloads (see fleet/fleetcore/topics.py) --
e.g. LARA_SIGNAL_READY becomes "wait for event battery_casing_ready on fleet/v1/lara-01/event"
instead of a local timer. Swapping that in touches r1_lara_maira_loop.py's per-step advance
condition; the step list, order, and logging here do not change.
"""
from __future__ import annotations

import enum


class Step(enum.Enum):
    LARA_SIGNAL_READY = 1
    NAV_HOME_TO_LARA = 2
    LARA_PICK = 3
    NAV_LARA_TO_MAIRA = 4
    MAIRA_PLACE = 5
    MAIRA_WORKING = 6
    NAV_BACK_TO_MAIRA = 7
    MAIRA_PICK = 8
    NAV_MAIRA_TO_LARA = 9
    LARA_DELIVER = 10


STEP_ORDER = list(Step)

STEP_LABEL = {
    Step.LARA_SIGNAL_READY: "LARA signals battery_casing_ready",
    Step.NAV_HOME_TO_LARA: "R1 navigating home -> LARA",
    Step.LARA_PICK: "R1 receiving casing from LARA",
    Step.NAV_LARA_TO_MAIRA: "R1 navigating LARA -> MAiRA",
    Step.MAIRA_PLACE: "R1 placing casing at MAiRA",
    Step.MAIRA_WORKING: "MAiRA performing assembly (R1 clears the cell)",
    Step.NAV_BACK_TO_MAIRA: "R1 navigating back to MAiRA",
    Step.MAIRA_PICK: "R1 receiving finished component from MAiRA",
    Step.NAV_MAIRA_TO_LARA: "R1 navigating MAiRA -> LARA",
    Step.LARA_DELIVER: "R1 delivering finished component to LARA",
}


class LoopFSM:
    """One step active at a time; advance() moves to the next, wrapping to a new cycle after 10."""

    def __init__(self, cycles: int = 1):
        self.cycles = cycles
        self.cycle_index = 0
        self.step_index = 0

    @property
    def step(self) -> Step:
        return STEP_ORDER[self.step_index]

    @property
    def step_number(self) -> int:
        return self.step.value

    @property
    def done(self) -> bool:
        return self.cycle_index >= self.cycles

    def label(self) -> str:
        return STEP_LABEL[self.step]

    def advance(self) -> None:
        self.step_index += 1
        if self.step_index >= len(STEP_ORDER):
            self.step_index = 0
            self.cycle_index += 1
