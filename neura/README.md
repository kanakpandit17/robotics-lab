# neura — MAiRA 7M and LARA 8

Everything for the two Neura Robotics arms in the cell, gathered in one place.

| Path | What |
|------|------|
| `maira/neura_robot_description/` | MAiRA 7M (7 DOF) ROS description: URDF/xacro, visual + collision meshes |
| `maira/maira7M.usd`, `maira/configuration/` | MAiRA converted to USD for Isaac Sim (keep `configuration/` next to `maira7M.usd`; it is referenced relatively) |
| `maira/maira_sim_bridge.py`, `maira/maira_bridge.json` | MAiRA sim↔real bridge (Windows / Isaac Sim side) and its shared JSON |
| `lara/lara8/` | LARA 8 (6 DOF) URDF with visual (`.glb`) and collision (`.obj`) meshes |
| `lara/lara8.usd` | LARA converted to USD by `import_lara8.py` |
| `lara/import_lara8.py` | URDF → USD conversion |
| `lara/lara_sim_bridge.py`, `lara/lara_real_reader.py` | LARA sim↔real bridge: Isaac Sim side and WSL/neurapy side |
| `rg6/` | OnRobot RG6 meshes + linkage config (from `onrobot_rg6_reference_description`, MIT) used by the MuJoCo model |
| `mujoco_models.py` → `maira_rg6_spec()` | MuJoCo MAiRA 7M + RG6 (real parallelogram linkage, actuator `rg6`) + wrist RealSense camera `wrist_rgbd` (pose and intrinsics from the real robot's TF/camera_info) + table and Pringles can |
| `maira_rg6_sim.py` | Sim2real tools: `check` (sim vs real kinematics), `gripper` (opening → gap / tip depth), `render` (wrist RGB + depth + can estimate), `look` (camera IK + path check), `view` (viewer, run with mjpython) |
| `thesis/can_pose.py` | Read-only pose estimate of a lying can from the MAiRA wrist RealSense (table-plane RANSAC + cylinder fit); run in the `neura` container with the AI SDK env |

Still in the repo root, because they also drive the R1 and import root modules:
`r1_cell_maira.py`, `r1_maira_scene.py`, `r1_lara_maira_loop.py`, `arena_live_scene.py`, `loop_states.py`.
The fleet MAiRA adapter stays in `fleet/agent/adapters/maira.py` (part of the fleet package).

The scripts load assets from absolute workstation paths (`C:\Users\aimat\Desktop\humanoid\...`),
not from this folder; see the main README.

## Connecting this Mac to the robot

The robot is driven with **neurapy 5.1.85**, which on the client side is a small pure-Python
library: JSON over TCP to the control box at `192.168.2.13:65432`. Nothing else has to be
installed on the laptop (manual, ch. 4.1 "Quick Installation").

One-time setup (already done on this Mac):

```bash
neura/setup_mac.sh
```

- creates `neura/.venv` with the neurapy wheel from `thesis_Kanak/neura-sdk/`, plus
  `mac_compat/win32api.py` (neurapy imports the Windows-only `win32api` on every non-Linux
  OS; the stand-in maps its Ctrl-C → `robot.stop()` hook to POSIX signals);
- adds a **"Neura LAN"** network service on the *USB 10/100 LAN* adapter (en6) with
  `192.168.2.99/24` and no router, so Wi-Fi keeps the internet. The adapter's existing
  service (R1, `192.168.123.222`) is untouched. Using a different dongle:
  `neura/setup_mac.sh "USB 10/100/1000 LAN"`.

Each session: plug the cable into the RJ45 port at the lower left inside the control box and
into that adapter, then

```bash
neura/.venv/bin/python neura/check_connection.py   # read-only: ping, port, name, DOF, joints
```

To drive it (status / jog one joint / move / stop; asks before moving, 10 % speed by default):

```bash
neura/.venv/bin/python neura/maira_control.py status
neura/.venv/bin/python neura/maira_control.py jog --joint 7 --deg 5
# same from the Ubuntu 20.04 "neura" container (OrbStack):
docker exec -it -u neura neura ~/neura/api_ws/venv/bin/python /work/neura/maira_control.py status
```

```python
from neurapy.robot import Robot
r = Robot()            # or Robot(socket_address="192.168.2.13")
r.get_current_joint_angles()
```

### MAiRA <-> LARA over MQTT (mirror motion)

Both control boxes are `192.168.2.13`, so each robot needs its own computer: LARA on this laptop,
MAiRA on the ARENA workstation, broker (mosquitto) on the workstation `141.70.221.73:1883`
(laptop on the ARENA2036 Wi-Fi).

```bash
# workstation (MAiRA on its cable) -- the leader
python neura/neura_mqtt_publisher.py --broker 127.0.0.1                  # publishes maira-live
# laptop (LARA on its cable) -- the follower; dry run first, it never moves
neura/.venv/bin/python neura/neura_mqtt_follower.py --leader maira-live --dry-run
neura/.venv/bin/python neura/neura_mqtt_follower.py --leader maira-live
```

The follower copies the leader's TCP *offset* since start (not joints: 7 vs 6 DOF), clamped to
+-5 cm (2 cm down), orientation held, IK + joint limits checked, `move_linear` at 0.05 m/s. It
holds when the leader is stale or offline. Swap roles by running the publisher on the laptop and
the follower on the workstation (`--leader lara-live`).

### Not usable on this Mac

- **NeuraAISDK v3.3.12** (`thesis_Kanak/NeuraAISDK-v3.3.12/`): the vision / grasping / AI
  skills stack. It needs Ubuntu 20.04 on x86-64 with ROS Noetic (all `.deb`s are `amd64`)
  and talks to the AI PC's ROS master at `192.168.2.14`. It cannot be installed on
  macOS/Apple Silicon; it needs an Ubuntu 20.04 x86 machine (see `NEURA_SDK_INSTALL_NOTES.md`).
  Plain robot control through neurapy does not need it.
- **`thesis_Kanak/NEURA.zip`** (App Creator images) is an incomplete download: it has no
  zip directory, so it cannot be extracted. Download it again if you need it.

## LARA + Contactile GAL2: simulation, sim2real and Ons's package

- `mujoco_models.py` → `lara_gal2_spec()`, `add_lab_cell()`: LARA 8 + GAL2 + the crate cell,
  checked against the real robot (TCP within 1.3 mm; fingers close along TCP y).
- `lara_gal2_sim2real.py` (check / twin / grip / follow / teleop / taps / pick) and
  `contactile_gal2.py` (gripper over USB serial, works on macOS).
- `dc_sim2real/`: Ons's data-collection scripts (`ons/0.1_Data Collection/`), unchanged, in a
  real-time MuJoCo cell (`--sim`) or on the robot (`--real`); tap/IMU model fitted to a real trace.
- `make_ons_package.py` builds `../ons_lara_sim/` (+ `--zip`), Ons's simulation-only copy. Its
  README (`ons_package/README.md`) is the full handover: measurements, sim-vs-real results, limits.
