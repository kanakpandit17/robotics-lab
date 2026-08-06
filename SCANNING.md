# Scanning the cell with SLAM3R

[SLAM3R](https://github.com/PKU-VCL-3DV/SLAM3R) (arXiv 2412.09401) reconstructs
a dense 3D point cloud from **monocular RGB video only** — no depth, stereo,
LiDAR or IMU. That matches the R1's single camera exactly, which is why this
paper is a good fit for your robot.

## The one hard constraint: it cannot run on your Mac

SLAM3R requires **CUDA**. Its install pins PyTorch built against CUDA 11.8,
and it compiles custom CUDA kernels (`curope`) plus optionally xformers. Your
MacBook is an **M3 Pro** — Apple Silicon, no NVIDIA GPU, no CUDA. There is no
MPS or CPU path in the project, and the custom kernels would not build.

So the work splits in two:

| Where | What |
|-------|------|
| **Mac** | capture frames (`scan_capture.py`), fix rotation (`fix_rotation.py`), undistort (`undistort.py`) |
| **CUDA machine** | run SLAM3R on those frames (`slam3r_colab.ipynb`) |

The paper reports 20+ FPS on an RTX 4090D (24 GB). A smaller card works for
offline reconstruction; you simply wait longer — a free Colab T4 is enough,
because the models run at 224×224 regardless of what you feed them.

## Step 1 — capture (on the Mac)

Ethernet to the robot, then:

```bash
python scan_capture.py --out scans/cell01
```

Move the robot around the cell while this runs. Press `q` to stop.

The tool is not just a recorder. It:

- **paces frames** at a steady 6 fps (`--fps`), rather than the stream's jitter
- **rejects motion-blurred frames**, which wreck feature matching. The
  threshold self-calibrates from the scene: your camera's sharp frames score
  ~310 and a blurred one ~2, but that spread shifts with lighting, so a fixed
  number would be wrong somewhere. It relaxes automatically if it starts
  rejecting everything.
- **shows live overlap** between consecutive frames. Below 0.5 it prints
  `SLOW DOWN` — that is the single most useful signal during a scan, because
  frames that do not overlap cannot be registered to each other.

Output:

```
scans/cell01/
    000000.jpg 000001.jpg …    what SLAM3R consumes
    manifest.json               per-frame sharpness and overlap
    preview.mp4                 quick check before transferring
```

### How to move for a good scan

- **Slow and steady.** Walking pace at most. Watch the overlap readout.
- **Loop back.** Returning to somewhere you have already been lets the system
  close the loop and corrects accumulated drift.
- **Circle objects** you care about rather than only looking at them head-on.
  Reconstruction needs parallax; a static viewpoint gives none.
- **Avoid pure rotation on the spot.** Spinning without translating produces no
  parallax and is the classic way to get a broken reconstruction.
- **200–800 frames** is a reasonable target for one cell.

## Step 2 — undistort (on the Mac)

The R1's camera is a **160° fisheye**. SLAM3R's own `docs/recon_tips.md` names
wide-angle lenses and unfamiliar distortion as known failure modes — it inherits
DUSt3R's roughly-pinhole assumption, and was trained on ScanNet++, Aria and
CO3D-v2, none of which look like this. Feeding raw frames is the most likely way
to get a warped reconstruction.

```bash
python undistort.py scans/cell01 --auto
```

`--auto` finds the lens's field of view without a calibration target, by
maximising the total length of straight line segments the detector finds in the
scene. A built environment is full of genuinely straight edges — extrusion, door
frames, ceiling beams — and they only come out straight at the right setting.
On `cell01` this peaks sharply at **160°**, which is now the default. Use
`--tune` for a slider and a grid overlay if you would rather judge by eye.

Output goes to `scans/cell01_undist/`.

### Why the output is square

The I2P and L2W models run at **224×224**, and DUSt3R's loader resizes the short
side to 224 then centre-crops a square. A 1280×720 frame therefore becomes
398×224 and gets cropped to 224×224 — **44% of your horizontal field of view is
discarded before the network sees anything**. Undistorting straight to a square
makes that crop a no-op, and the pixels it would have thrown away are the
worst-distorted edges anyway.

## Step 3 — reconstruct (on a CUDA machine)

Open **`slam3r_colab.ipynb`** in Google Colab (Runtime → T4 GPU), drag
`scans/cell01_undist.zip` into the file pane, and run the cells. It installs
SLAM3R, pulls the checkpoints from HuggingFace, runs `recon.py` with parameters
matched to how this scan was captured, previews the cloud as three orthogonal
projections, and downloads the `.ply`.

Two install details the notebook handles that trip people up on a hand-rolled
setup: torch is pinned to **2.5.0**, because 2.6 flipped `torch.load` to
`weights_only=True` by default and that breaks the SLAM3R checkpoints; and the
build is **cu121**, not the repo's cu118, to match Colab's driver.

Output is a dense point cloud in a single global coordinate frame — no metric
scale, since monocular reconstruction cannot recover it.

To run it somewhere other than Colab, the equivalent by hand:

```bash
git clone https://github.com/PKU-VCL-3DV/SLAM3R.git && cd SLAM3R
conda create -n slam3r python=3.11 cmake=3.14.0 -y && conda activate slam3r
pip install torch==2.5.0 torchvision==0.20.0 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
cd slam3r/pos_embed/curope/ && python setup.py build_ext --inplace && cd ../../..

python recon.py --img_dir /path/to/cell01_undist --test_name cell01 \
    --keyframe_stride 3 --win_r 5 --buffer_size 150 --gpu_id 0
```

## Field of view, after correction

The R1's sensor is mounted sideways, so `scan_capture.py` rotates frames to
1280×720 landscape. `undistort.py` then produces a ~720×600 pinhole image
covering 100°. That is a normal-looking camera, but it is one camera: you still
need more overlapping viewpoints than a stereo or RGB-D rig would ask for.
Plan on covering the cell more thoroughly than feels necessary.

## Moving the robot

`scan_capture.py` does not drive the robot; it only records. To move it:

- the **Unitree app** (simplest, works today)
- carry or push it
- the `sport`/`loco` DDS API — available on the robot, not yet wired up here.
  Ask if you want that; a humanoid walking under program control needs a
  conversation about clear space and an e-stop first.
