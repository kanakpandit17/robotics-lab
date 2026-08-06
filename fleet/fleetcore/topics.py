"""Topic namespace for the fleet bus.

Every topic is versioned and scoped to one robot, so adding a second robot is a
new namespace rather than a change to anything already running. The scheduler
subscribes with wildcards and discovers robots as they appear.

    fleet/v1/<robot_id>/cmd                  laptop -> agent
    fleet/v1/<robot_id>/state                agent  -> laptop   (retained)
    fleet/v1/<robot_id>/presence             agent  -> laptop   (retained, LWT)
    fleet/v1/<robot_id>/event                agent  -> laptop   (task progress)
    fleet/v1/_reply/<client_id>/<command_id> agent  -> one laptop

Why `state` and `presence` are separate
---------------------------------------
`state` is what the robot says about itself. `presence` is what the *broker*
says about the robot: it is published as a Last Will and Testament, so if the
workstation loses power mid-task the broker publishes `online: false` on its
behalf. A dead agent cannot tell you it is dead, which is exactly when you most
need to know.

Both are retained, so a scheduler that starts later immediately learns the last
known state of every robot without asking anyone.
"""

from __future__ import annotations

VERSION = "v1"
ROOT = "fleet"


def cmd(robot_id: str) -> str:
    return f"{ROOT}/{VERSION}/{robot_id}/cmd"


def state(robot_id: str) -> str:
    return f"{ROOT}/{VERSION}/{robot_id}/state"


def presence(robot_id: str) -> str:
    return f"{ROOT}/{VERSION}/{robot_id}/presence"


def event(robot_id: str) -> str:
    return f"{ROOT}/{VERSION}/{robot_id}/event"


def reply(client_id: str, command_id: str) -> str:
    return f"{ROOT}/{VERSION}/_reply/{client_id}/{command_id}"


def reply_wildcard(client_id: str) -> str:
    return f"{ROOT}/{VERSION}/_reply/{client_id}/+"


def all_states() -> str:
    return f"{ROOT}/{VERSION}/+/state"


def all_presence() -> str:
    return f"{ROOT}/{VERSION}/+/presence"


def all_events() -> str:
    return f"{ROOT}/{VERSION}/+/event"


def robot_from_topic(topic: str) -> str | None:
    """Pull the robot id out of any per-robot topic."""
    parts = topic.split("/")
    if len(parts) >= 4 and parts[0] == ROOT and parts[1] == VERSION:
        return parts[2] if parts[2] != "_reply" else None
    return None
