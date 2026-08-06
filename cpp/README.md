# C++ arm example for the R1

Raise one joint 30°, hold, lower it back. Untested against hardware — it cannot
be tested until the two blockers below are cleared.

## C++ does not fix anything on its own

C++ and Python use the **same DDS transport**. Every failure in this project was
at the network layer, below the language. This program run from the Mac today
fails exactly as the Python did. Two things have to be true first.

## Blocker 1 — debug mode

Unitree's guide: *"Follow the operation process of 'Quick Start' to ensure that
the robot enters debugging mode."* Every SDK example is gated behind it. A robot
that is merely powered and standing is **not** in debug mode; that state stops
the official control services and hands control to the SDK. Without it,
`rt/lowcmd` is ignored.

## Blocker 2 — network

Measured on this robot: over its Wi-Fi AP only UDP **7400** is reachable. That
is the multicast discovery port, which is why topics are visible. Each
participant's *unicast* ports (`7410 + 2*index`), where data and RPC actually
flow, are bound to an interface not reachable from the AP — hence discovery
succeeds and every call returns 3102.

Unitree's supported path is a **network cable to the robot's switch**, putting
the PC on `192.168.123.x` — the same segment those unicast ports are bound to.
That is expected to bypass the problem entirely.

```bash
sudo ifconfig <iface> 192.168.123.222 netmask 255.255.255.0
ping 192.168.123.161
ifconfig                       # note the interface name on 192.168.123.x
```

## Build (Ubuntu 20.04)

macOS is not supported for the C++ SDK — Unitree state this explicitly. Use a
Linux machine or a VM. (The Python tooling in the parent directory *does* run on
macOS and is already working.)

```bash
git clone https://github.com/unitreerobotics/unitree_sdk2.git
cd unitree_sdk2 && mkdir build && cd build
cmake .. && sudo make install

cd /path/to/humanoid-try/cpp
mkdir build && cd build
cmake .. && make
```

## Run

```bash
./r1_arm_example eth0                  # READ ONLY: print the joint table
./r1_arm_example eth0 --joint 22       # dry run: print the trajectory
./r1_arm_example eth0 --joint 22 --go  # actually move
```

Replace `eth0` with your interface on `192.168.123.x`.

## Safety

**Hang or support the robot before `--go`.** Unitree say the same for the
equivalent H1 routine: *"Running this routine will cause multiple joints to
move. To protect the robot, hang it before running this routine."*

The R1 exposes no `rt/arm_sdk`, so the only path is `rt/lowcmd` — full
low-level control of every joint including the legs. This program snapshots
every joint and commands them all to hold their current angle, ramping only the
one you name; a frame with unset joints would drop a standing robot. Even so it
takes authority from the balance controller.

`--joint` is deliberately required with no default. Published index tables are
for the G1; the R1 has a different joint count and no public table. Run without
`--joint`, move the right arm by hand, re-run, and see which index changed.

Other guards: 60° hard cap, cosine-eased ramp (~24°/s peak), low gains so a
collision is not fought, and it always returns to the starting angle.

## CRC

Every `LowCmd_` carries a CRC32 that the robot validates. Get it wrong and the
command is **silently dropped** — no motion, no error. `Crc32Core()` in the
source handles it; keep the call if you modify the send path.
