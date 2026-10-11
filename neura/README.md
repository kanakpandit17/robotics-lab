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

Still in the repo root, because they also drive the R1 and import root modules:
`r1_cell_maira.py`, `r1_maira_scene.py`, `r1_lara_maira_loop.py`, `arena_live_scene.py`, `loop_states.py`.
The fleet MAiRA adapter stays in `fleet/agent/adapters/maira.py` (part of the fleet package).

The scripts load assets from absolute workstation paths (`C:\Users\aimat\Desktop\humanoid\...`),
not from this folder; see the main README.
