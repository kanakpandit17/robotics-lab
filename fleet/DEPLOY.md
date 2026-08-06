# Deploying across the two machines

The MAiRA ethernet cable does not move. The workstation keeps its `192.168.x`
link to the robot and additionally talks to your laptop over the arena wifi —
separate interfaces, no conflict.

```
MacBook                    ARENA2036 wifi           Workstation
141.70.221.54                                       141.70.221.???
                                                          │
  fleetcore/                                        fleetcore/
  scheduler/          ──────── MQTT :1883 ────────▶ agent/
  config/laptop.json                                config/maira-agent.json
                                                    mosquitto  ← broker here
  pip: paho-mqtt                                    pip: paho-mqtt + NEURA SDK
                                                          │
                                                    ethernet (untouched)
                                                          ▼
                                                        MAiRA
```

**The broker belongs on the workstation.** It has a fixed address and stays
powered; your laptop sleeps and moves. That way the laptop is the only machine
that ever needs a config change.

**Never copy `agent/` to the laptop, or `scheduler/` to the workstation.** That
split is the entire guarantee that your laptop cannot acquire a dependency on
the NEURA SDK or on ROS.

---

## 1. Copy to the workstation

Everything except `scheduler/`:

```bash
# from the laptop, if the workstation runs Linux with ssh
rsync -av --exclude scheduler --exclude __pycache__ \
      ~/Desktop/humanoid-try/fleet/ user@WORKSTATION_IP:~/fleet/
```

On Windows, copy the folder over the network or with a USB stick. The files
needed are `fleetcore/`, `agent/`, `tools/`, `config/`, `requirements.txt`.

## 2. Install on the workstation — into the SDK's environment

This is the step that goes wrong most often. The agent **imports the NEURA SDK
directly**, so it has to run in the same interpreter that SDK lives in. A fresh
venv will import cleanly and then fail at `connect()`.

```bash
pip list | grep -i neura        # confirm you are in the right environment
pip install paho-mqtt
```

## 3. Broker on the workstation

```bash
# Linux
sudo apt install mosquitto
mosquitto -c config/mosquitto.conf -v

# Windows: install from https://mosquitto.org/download/ then
"C:\Program Files\mosquitto\mosquitto.exe" -c config\mosquitto.conf -v
```

Mosquitto 2.x binds to localhost only and refuses anonymous clients unless told
otherwise, which makes a default install look like a network fault — the
supplied `config/mosquitto.conf` sets `listener 1883 0.0.0.0` and
`allow_anonymous true`. Do not skip it.

## 4. Point the configs at it

```bash
# on the workstation
python tools/set_broker.py 127.0.0.1 --agent-only

# on the laptop, with the workstation's wifi IP
python tools/set_broker.py WORKSTATION_IP --laptop-only
```

## 5. Prove the path before involving the robot

```bash
# workstation
python -m agent.main --config config/maira-agent.json --simulate

# laptop
python -m scheduler.cli --config config/laptop.json list
python -m scheduler.cli --config config/laptop.json demo maira-01
```

`list` showing `maira-01  NEURA MAiRA  IDLE` means the whole chain works. Only
then fill in the SDK and drop `--simulate`.

## 6. Firewall

Port 1883 inbound on the workstation.

```powershell
# Windows, as administrator
New-NetFirewallRule -DisplayName "MQTT fleet" -Direction Inbound `
  -LocalPort 1883 -Protocol TCP -Action Allow
```

```bash
# Linux
sudo ufw allow 1883/tcp
```

## 7. Start on boot

**Linux** — `/etc/systemd/system/robot-agent.service`:

```ini
[Unit]
Description=MAiRA Robot Edge Agent
After=network-online.target
Wants=network-online.target

[Service]
Type=simple
User=YOUR_USER
WorkingDirectory=/home/YOUR_USER/fleet
ExecStart=/path/to/sdk/python -m agent.main --config config/maira-agent.json
Restart=always
RestartSec=5

[Install]
WantedBy=multi-user.target
```

```bash
sudo systemctl enable --now robot-agent
journalctl -u robot-agent -f
```

`Restart=always` is safe here: the agent stops the robot on shutdown and starts
in `STARTING` rather than resuming anything. A restart never continues a motion
it does not remember.

**Windows** — Task Scheduler, trigger *At startup*, action:

```
Program:   C:\path\to\sdk\python.exe
Arguments: -m agent.main --config config\maira-agent.json
Start in:  C:\path\to\fleet
```

Tick *Run whether user is logged on or not*, and on the Settings tab *If the
task fails, restart every 1 minute*.

---

## Checks when something is wrong

| Symptom | Look at |
|---|---|
| `list` shows nothing | broker reachable from the laptop? `nc -z WORKSTATION_IP 1883` |
| Agent logs "broker not reachable yet" | mosquitto running, and firewall open on 1883 |
| Robot shows `OFFLINE` but the agent is up | two agents with the same `robot_id`; one is fencing the other |
| `SDK_UNAVAILABLE` | agent is not running in the NEURA SDK's interpreter |
| Everything works, then stops after moving the laptop | new laptop IP is fine — only the broker's address matters |
