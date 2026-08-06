# Fleet — distributed control for heterogeneous robots

Your laptop sends `robot.execute("PickBattery")`. A Robot Edge Agent on the
machine with the cable in it translates that into the vendor SDK. The laptop
never imports a robot SDK, never opens a DDS socket, and never learns that ROS
exists.

```
LAPTOP                    │ NETWORK │            MAiRA WORKSTATION
                          │         │
  your scheduler          │         │   Edge Agent (agent/)
        ↓                 │         │        ↓
  Fleet / RobotClient ────┼─ MQTT ──┼──→ CommandHandler → Executor
        ↑                 │ broker  │        ↓
        └─ state, events ─┼─────────┼─── SDKWrapper (RobotAdapter)
                          │         │        ↓
                          │         │   NEURA SDK  ← ROS may live here
                          │         │        ↓        and nowhere else
                          │         │      MAiRA
```

Adding a robot adds an Edge Agent and an adapter. It changes nothing on the
laptop.

## Why MQTT

| | REST | WebSocket | gRPC | **MQTT** |
|---|---|---|---|---|
| Works behind NAT / venue wifi | needs inbound port | needs inbound port | needs inbound port | **both sides dial out** |
| Server → client push | polling only | yes | yes (streams) | **yes** |
| Knows a robot died | no | on TCP close | on TCP close | **broker publishes it (LWT)** |
| Late joiner gets current state | must poll | must ask | must ask | **retained messages** |
| Adding robot #2 | new address | new connection | new address | **new topic namespace** |
| Redelivery on flaky link | manual retry | manual | manual | **QoS 1** |
| Typed contract | no | no | **strong** | schema in `envelope.py` |

The deciding factor is **arena wifi**. Client isolation and NAT are normal
there, and every protocol except MQTT needs the workstation to be reachable
*inbound* from the laptop. With a broker, both sides make outbound connections
and never need to find each other.

Last Will and Testament is the second reason. A crashed workstation cannot
report its own crash — the broker does it. That single feature replaces a
heartbeat protocol you would otherwise have to write and debug.

gRPC would be the better answer on a wired network with joint-level streaming.
You have neither. Revisit it if you ever need 100 Hz trajectories.

## Layout

```
fleet/
├── fleetcore/           shared contract — the ONLY thing both sides agree on
│   ├── topics.py        topic namespace, one per robot
│   ├── envelope.py      Command / Reply / StateMessage / Presence
│   ├── errors.py        ErrorCode taxonomy, retryable flag
│   └── transport.py     Transport protocol; MQTT + in-process loopback
├── agent/               RUNS ON THE WORKSTATION ONLY
│   ├── main.py          entry point
│   ├── edge_agent.py    bus I/O, dedupe, presence, state publishing
│   ├── executor.py      single-flight execution, preemptible Stop, state machine
│   └── adapters/
│       ├── base.py      RobotAdapter ABC + CancelToken
│       ├── maira.py     NEURA SDK   ← the only file that imports NEURA
│       ├── unitree_r1.py Unitree DDS
│       └── simulator.py  fake robot for developing without hardware
├── scheduler/           RUNS ON THE LAPTOP ONLY
│   ├── fleet.py         discovery, capability routing, stop_all
│   ├── client.py        RobotClient — execute / execute_async / stop
│   └── cli.py           worked example
├── config/*.json
└── tests/test_stack.py  full stack over the loopback broker
```

`scheduler/` importing anything from `agent/` is the architectural violation to
watch for. There is no legitimate reason for it.

## Run it

**Today, with no robot and no broker** — the whole stack in one process:

```bash
python -m tests.test_stack
```

**With a broker**, on any machine both sides can reach:

```bash
docker run -d -p 1883:1883 eclipse-mosquitto
# workstation
python -m agent.main --config config/maira-agent.json --simulate
# laptop
python -m scheduler.cli --config config/laptop.json list
python -m scheduler.cli --config config/laptop.json demo maira-01
```

Drop `--simulate` once `SDK-BINDING` in `agent/adapters/maira.py` is filled in.

## Design decisions worth knowing

**Two-phase replies.** A command is answered twice: `accepted` immediately, then
`completed`/`failed` when the motion ends. A pick takes 30 seconds, and one
blocking reply cannot distinguish "still moving" from "the workstation died" —
opposite situations for a scheduler.

**Stop bypasses the queue.** It is handled on the transport thread and calls the
SDK directly while the worker is still blocked inside a motion. A Stop that
queued behind the motion it interrupts would be worthless. Measured at 1 ms
while BUSY.

**Commands are idempotent.** QoS 1 is at-least-once, so a flaky link *will*
redeliver. The agent remembers recent `command_id`s and replays the original
reply rather than picking the object up twice.

**ESTOP latches.** Stop leaves the robot refusing motion until an explicit
`ResetError`. Nothing clears it automatically.

**Losing the broker does not stop the robot.** Aborting a half-finished pick
because wifi blinked is worse than finishing it. New work is refused while
disconnected, and full state is republished on reconnect. Set
`stop_on_disconnect_s` if your cell needs the opposite — that is a safety
judgement about your setup, not a default anyone should pick for you.

**Capabilities are discovered, not configured.** Each agent advertises what it
can do, so `fleet.any_capable("Gesture")` routes to the R1 and
`any_capable("PickObject")` routes to MAiRA with no robot names in your code.
The R1 adapter deliberately does *not* advertise `PickObject` — it has no
gripper, and a capability it cannot honour would move the failure from planning
time to motion time.

## Adding a robot

1. Subclass `RobotAdapter`, implement `connect`, `capabilities`, `stop`.
2. Register it in `ADAPTERS` in `agent/main.py`.
3. Write a config, run an agent on the machine wired to that robot.

The scheduler is not touched. `tests/test_stack.py` case 10 does exactly this
with a gripper-less robot and asserts nothing upstream changed.

## Verified

`python -m tests.test_stack` — 32 checks, all passing:

discovery · capability advertisement · command completion · unsupported refused ·
bad params refused · BUSY on concurrent commands · **Stop preempting a running
motion in 1 ms** · ESTOP latch and reset · **duplicate delivery running the
adapter exactly once** · **unclean workstation death detected via the will,
in-flight commands failed fast** · heterogeneous fleet with capability routing ·
fleet-wide stop.

Not yet verified: the NEURA SDK binding, and MQTT against a live broker (the
transport constructs and registers its will; the protocol above it is what the
loopback broker exercises).
