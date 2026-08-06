# r1vision — see what the Unitree R1 is looking at

A live viewer for the R1's head camera, over ethernet, on macOS.

## Confirmed on this unit (2026-07-27)

Everything below was measured against the actual robot, not inferred from
Unitree's Go2 documentation.

| | |
|---|---|
| Connection | **Wi-Fi only.** No ethernet port — a cable gets no carrier at all |
| Robot AP | `R1_03302_c33840b2`, password `88888888` |
| Robot IP | `192.168.12.1` (DHCP gives your Mac `192.168.12.x`) |
| Video | **`rtsp://192.168.12.1:8551/video`** — GStreamer RTSP server |
| Codec | H.264 High profile @ level 4.0, 1280×720 |
| Orientation | sensor mounted 90° out; needs `--rotate 270` |
| DDS | **absent.** No RTPS multicast, no topics, `videohub` returns 3102 |
| Other ports | `:53` dnsmasq, `:9991` Unitree app control channel |

So `--source dds` and the whole `unitree_sdk2py` path do *not* apply to this
robot — its video is plain RTSP on a non-standard port. Port 8551 is why a
normal scan finds nothing: almost every RTSP tool probes only 554 and 8554.

The camera is a single 1280×720 stream, not a side-by-side stereo pair,
despite the "binocular" spec.

## Motion control: blocked by the robot's DDS binding (2026-07-27)

**Root cause, measured:** on both robot addresses, of 147 SPDP ports scanned
(7400-7700, each candidate confirmed 3x), **only `:7400` is bound**.

`7400` is the multicast discovery port, bound on `0.0.0.0`. Each participant's
*unicast* ports (`7410 + 2*index`) — where RPC and sample data actually flow —
are bound to an interface not reachable from the Wi-Fi AP. Hence:

| | |
|---|---|
| Discovery | works — topics, participants and service names are all visible |
| Data flow | fails — every RPC returns 3102, `rt/lowstate` never delivers a sample |

This is why `arm_actions.py` sees the arm service exists yet cannot call it,
and why `rt/lowstate` is listed as published but yields nothing. It is a robot
network-binding property, not a bug in this code and not something a longer
timeout or wider peer sweep can fix. Both were tried:

- participant index sweeps 0-45, then 0-99 (202 unicast locators)
- explicit `host:7400` peers (this is what made discovery work at all)
- multicast and unicast, domains 0, 1, 2
- 65 s of retries with 15 s discovery settle

**What would actually unblock it**

1. **Run the code on the robot.** DDS works fine locally. Needs shell access —
   no SSH port is open on the Base; the EDU's Jetson would have one.
2. **Wired access to the internal segment**, where the unicast ports are bound.
   The R1 has no RJ45, so this needs the EDU expansion.
3. **Ask Unitree** how to expose SDK data flow over the AP, quoting the
   measurement above.
4. **Use the app** for arm gestures in the meantime — it drives the robot over
   its own 9991 channel and is unaffected by any of this.

## Earlier conclusion: not available on this unit (superseded)

Video works. Commanding the robot does not. This is a property of the robot's
current firmware/configuration, not of the code here — recorded so nobody
repeats the investigation.

The robot exposes **exactly two services**, on both its AP address
(`192.168.12.1`) and its internal address (`192.168.123.161` — same device,
two interfaces):

| Port | Service |
|------|---------|
| 8551 | GStreamer RTSP — video, one-way, carries no commands |
| 9991 | Unitree app control channel — proprietary, undocumented |

There is no SSH, no DDS, and no other open port anywhere in 1–65535.

What was tested for DDS, all negative:

- multicast SPDP listen on `239.255.0.1:7400`, 25 s — no RTPS traffic at all
- builtin topic enumeration (`DCPSPublication` / `DCPSSubscription`), domains 0, 1, 2
- unicast peer discovery to `192.168.12.1`
- unicast peer discovery to `192.168.123.161`, with a route added via the AP
- Unitree `VideoClient` / `videohub` RPC — returns error 3102

`192.168.123.161` **is** reachable, but only after adding a route, because the
AP's DHCP hands out no router option:

```bash
sudo route -n add -net 192.168.123.0/24 192.168.12.1   # gone on reconnect
```

Port 9991 accepts TCP connections but answers nothing to raw, newline, HTTP or
WebSocket probes — it is a binary protocol that expects the client to speak
first in a specific format.

**Two real fixes**, neither of which is a code change:

1. Confirm whether this is an R1 **Base** or **EDU**. The EDU adds a Jetson
   Orin and a normal Linux userland; SDK motion control may only exist there.
2. Ask Unitree support directly: *"how is unitree_sdk2 motion control exposed
   on the R1, given the robot presents only ports 8551 and 9991 and no DDS
   over its Wi-Fi AP?"* Their [R1 SDK guide](https://support.unitree.com/home/en/R1_developer)
   exists but does not cover the network path.

`arm_wave.py` is written and ready for the moment a channel appears; it is
untested against hardware and cannot be tested until then.

---

## Quick start

The venv is already built (Python 3.12 — OpenCV has no arm64 wheel for the
3.14 that `python3` points at on your machine).

1. Power on the R1 and join its Wi-Fi: `R1_03302_c33840b2` / `88888888`
2. `./watch.sh`

That is it. `watch.sh` checks you are on the robot's AP and opens the live view
already rotated the right way up. Any `r1vision.view` flag passes through:

```bash
./watch.sh              # live view
./watch.sh --grid       # with the thirds grid
./watch.sh --scale 1.5  # bigger window
```

Equivalent long form:

```bash
./.venv/bin/python -m r1vision.view \
    --source rtsp://192.168.12.1:8551/video --rotate 270
```

If the robot's address changes: `R1_HOST=192.168.12.5 ./watch.sh`.

To skip the `./.venv/bin/` prefix: `source .venv/bin/activate`.

---

## Step 1 — network

**The R1 connects over Wi-Fi 6, not ethernet.** Its published spec lists Wi-Fi 6
and Bluetooth 5.2 and no RJ45, on both Base and EDU. This differs from the
Go2/G1/H1, which do have gigabit ethernet — most Unitree networking guides you
will find online are written for those and do not apply here. If you plug an
ethernet cable into the R1 you will get no carrier at all
(`media: autoselect (none)`, `status: inactive`), because there is nothing on
the other end to drive the line.

Two ways to get on the same network as the robot:

**A. Join the robot's own access point (recommended).** The R1 broadcasts an
SSID at power-on, typically named after the model and serial. Join it from
macOS Wi-Fi settings; the robot runs DHCP on its own subnet, so you get an
address automatically and `discover` will sweep it.

**B. Put the robot on your LAN with the Unitree app** (pairs over Bluetooth,
then hands the robot your Wi-Fi credentials). Both devices end up on the same
network and `discover` finds it there.

Option B will *not* work on eduroam or most campus/corporate Wi-Fi: those
enable client isolation, which blocks the peer-to-peer traffic video streaming
needs. Use option A, or a personal router/hotspot.

`discover` sweeps `192.168.123.0/24` plus any /24 your Mac is directly on. It
deliberately refuses to sweep wider subnets — on eduroam your Mac holds a /19,
and scanning that would touch thousands of unrelated machines. Override with
`--subnet 192.168.12` if you know the robot's range.

### If your unit does turn out to have ethernet

`./setup_net.sh` sets a wired adapter to a static `192.168.123.222/24`, chosen
to avoid the addresses Unitree uses (`.10` motion control, `.15` switch,
`.161`/`.164` onboard computer, `.220` their default for your PC), since their
onboard segment runs no DHCP server.

```bash
./setup_net.sh              # auto-detect the adapter
./setup_net.sh en6          # or name it
./setup_net.sh en6 --undo   # give it back to DHCP when you're done
```

## Step 2 — discovery

```bash
python -m r1vision.discover
```

It runs four passes:

1. **Interfaces** — lists wired adapters, flags no-link and wrong-subnet, and
   prints the copy-pasteable fix.
2. **Sweep** — pings all of `192.168.123.0/24`, merges in the ARP table to
   catch devices that ignore ICMP, and labels known Unitree addresses.
3. **Probe** — port-scans each host, then actively tests every candidate video
   endpoint: RTSP `DESCRIBE` on `:554`/`:8554` across nine common paths, and
   HTTP on the usual ports across twelve paths, sniffing for
   `multipart/x-mixed-replace` or raw JPEG magic bytes.
4. **DDS** — tries the Unitree SDK `videohub` RPC if the SDK is installed.

Anything that returns a real frame gets printed as a ready-to-run command.

Useful flags: `--host 192.168.123.161` (skip the sweep), `--skip-dds`,
`--subnet 192.168.1` (if your R1 is on a different range).

## Step 3 — view

```bash
python -m r1vision.view --source <what discover printed>
```

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| `q` / `Esc` | quit | `s` | save PNG snapshot |
| `space` | pause | `r` | start/stop mp4 recording |
| `g` | thirds grid | `h` | toggle HUD |
| `[` `]` | shrink / grow | | |

Snapshots and recordings land in `captures/`.

The HUD shows source, resolution, live FPS, frame and drop counts, elapsed
time and wall clock. The green reticle marks the optical axis — the centre of
the crosshair is where the head is pointed.

Other flags: `--mirror`, `--rotate 90|180|270`, `--scale 1.5`, `--width 1280`,
`--grid`, `--no-reticle`, `--size 1280x720` (required for bare `udp://`/`rtp://`
if ffprobe can't infer it).

Sources accepted by `--source`:

| Value | Backend | Notes |
|-------|---------|-------|
| `webcam`, `webcam:1` | OpenCV | your Mac's camera, for testing the UI |
| `rtsp://…`, `rtmp://…` | OpenCV + FFmpeg | forced over TCP; UDP smears frames |
| `http://…/stream` | hand-rolled MJPEG | lower latency than OpenCV's parser |
| `udp://@:5000`, `rtp://…`, `*.sdp` | ffmpeg subprocess | pip's OpenCV has no GStreamer |
| `dds` | `unitree_sdk2py` | needs `--interface en5` |

The viewer reconnects on its own if the stream drops.

---

## If discovery finds nothing

That means the R1 isn't running a network video server by default — likely, and
not a fault. Go in and start one:

```bash
ssh unitree@192.168.123.161      # password is usually 123 or Unitree0408
```

Find the camera:

```bash
ls /dev/video*
v4l2-ctl --list-devices
```

Then push a stream back to your Mac (`192.168.123.222`):

```bash
# on the robot
ffmpeg -f v4l2 -i /dev/video0 -vcodec libx264 -preset ultrafast \
       -tune zerolatency -f mpegts udp://192.168.123.222:5000
```

```bash
# on your Mac
python -m r1vision.view --source udp://@:5000 --size 1280x720
```

If the head camera is a RealSense (as on some EDU configs), `rs-server` or a
small `pyrealsense2` + MJPEG script on the robot is the tidier route — then
point `--source` at `http://192.168.123.161:8080/stream`.

The R1's head camera is **binocular**. If a stream comes back
double-width (e.g. 2560×720), it's both eyes side by side; that's the raw
stereo pair, not a bug.

## DDS backend

Only needed for `--source dds`, and only worth setting up if discovery reports
`videohub` answering. Unitree pins `cyclonedds==0.10.2`, which has no macOS
arm64 wheel and needs a source build:

```bash
git clone https://github.com/eclipse-cyclonedds/cyclonedds -b releases/0.10.x
cd cyclonedds && mkdir build install && cd build
cmake .. -DCMAKE_INSTALL_PREFIX=../install && cmake --build . --target install
export CYCLONEDDS_HOME=$PWD/../install
pip install cyclonedds==0.10.2
pip install git+https://github.com/unitreerobotics/unitree_sdk2_python.git
```

Note that `videohub` is confirmed on the Go2 and is *not* confirmed on the R1 —
try the network transports first.

## Troubleshooting

| Symptom | Cause |
|---------|-------|
| discovery finds no interfaces | adapter unplugged; `networksetup -listallhardwareports` |
| link `inactive` | cable, or robot still booting (~60s after power-on) |
| sweep finds nothing | Mac not on `192.168.123.x` — run `./setup_net.sh` |
| RTSP opens then stalls | already forced to TCP; try `--source udp://…` push instead |
| DDS error 3104 | known intermittent SDK fault; the source rides out 30 before dropping |
| black window, no error | stream is up but the camera is disabled robot-side |

## Layout

```
r1vision/
  net.py        interface detection, static-IP command, TCP probe
  discover.py   sweep + port scan + transport probing
  sources.py    the five frame backends behind one interface
  viewer.py     window, HUD, snapshot/record, reconnect
setup_net.sh    macOS static IP helper
selftest.py     end-to-end proof with no robot attached
```
