"""The immersive WebXR page served to the headset.

Everything is adjustable from inside the headset -- rotation, flat vs curved,
size, distance -- because guessing the right orientation from outside and
restarting the server each time is miserable. Change it while wearing it.

Hand-rolled WebGL2, no Three.js or A-Frame, so the page is one self-contained
file with nothing to fetch.

WebXR supplies the projection and view matrices per eye, so there is no camera
maths here; we build the mesh and hand those matrices to the shader.

Driving
-------
Thumbsticks steer, and letting go of them stops the robot -- the same feel as
Unitree's own controller. The stick itself is the deadman: it is spring
centred, so releasing it produces a zero, and zero velocity means the robot
stops walking while continuing to stand and balance. It does not go limp.
(Damp, which does release the joints, is a separate deliberate command.)

An optional `--deadman` mode additionally requires a grip button to be held.
That guards against brushing a stick, at the cost of not behaving like the
controller everyone already knows.

Raw stick values go to the server unshaped. Deadzone and response curve are
applied there, in one tested place, and the server answers with the velocity it
actually committed to -- so the number on the HUD is what the robot was told,
not what the stick was asked for.
"""

from __future__ import annotations


# The gestures worth reaching for first, in the order they are most used. Every
# action the robot reports still gets a button; this only decides what comes
# first and what the four controller buttons map to.
FAVOURITES = (
    "wave_above_head", "right_hand_up", "shake_hand", "release_arm",
    "both_hands_up", "high_five", "hug", "emphasize",
)


def _label(name: str) -> str:
    """'wave_above_head' -> 'Wave above head'."""
    return name.replace("_", " ").capitalize()


def action_buttons(actions: dict[str, int] | None = None) -> str:
    """Buttons for every gesture the robot reports, favourites first.

    Generated rather than hand-written: the R1 lists 22 actions and hardcoding
    four of them meant the other eighteen were unreachable without editing the
    server. `arm_actions.ACTIONS` is the same table `--list` prints.
    """
    if actions is None:
        from arm_actions import ACTIONS as actions

    ordered = [n for n in FAVOURITES if n in actions]
    ordered += sorted(n for n in actions if n not in FAVOURITES)

    rows = []
    for name in ordered:
        # release_arm is the one to reach for when a gesture leaves an arm
        # somewhere awkward, so it is styled apart from the rest.
        cls = ' class="primary"' if name == "release_arm" else ""
        rows.append(f'  <button{cls} onclick="act(\'{name}\')">'
                    f'{_label(name)}</button>')
    return "\n" + "\n".join(rows) + "\n"


def render_page(walk_enabled: bool = False, deadman: bool = False,
                teleop_enabled: bool = False, record_enabled: bool = False,
                teleop_hands: tuple[str, ...] = (),
                actions: dict[str, int] | None = None) -> str:
    """The page, with driving/teleop/recording each compiled in or out.

    Absent unless the server was started with the matching flag, rather than
    present and refused. A control that does nothing teaches you to press it
    harder.
    """
    hands_js = "[" + ",".join(f'"{h}"' for h in teleop_hands) + "]"
    return (_PAGE
            .replace("__WALK_ENABLED__", "true" if walk_enabled else "false")
            .replace("__DEADMAN__", "true" if deadman else "false")
            .replace("__TELEOP_ENABLED__", "true" if teleop_enabled else "false")
            .replace("__RECORD_ENABLED__", "true" if record_enabled else "false")
            .replace("__TELEOP_HANDS__", hands_js)
            .replace("__ACTION_BUTTONS__", action_buttons(actions)))


_PAGE = """<!doctype html>
<html><head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1,user-scalable=no">
<title>R1 Vision</title>
<style>
  :root { color-scheme: dark; }
  * { margin:0; padding:0; box-sizing:border-box; }
  html,body { height:100%; background:#000; overflow:hidden; color:#eee;
              font-family:system-ui,-apple-system,sans-serif; }
  canvas { display:block; width:100%; height:100%; touch-action:none; }
  /* One scrolling column pinned to the bottom, rather than three bars stacked
     at hardcoded offsets: the action list is as long as the robot says it is,
     so nothing above it can assume a fixed height. */
  #panel { position:fixed; left:0; right:0; bottom:0; max-height:72vh;
           overflow-y:auto; display:flex; flex-direction:column; }
  #ui { padding:.7rem; display:flex; gap:.5rem; flex-wrap:wrap;
        align-items:center; background:rgba(0,0,0,.75);
        backdrop-filter:blur(8px); }
  button { flex:1 1 auto; min-width:7.5rem; padding:.9rem; font-size:1rem;
           border:1px solid #444; border-radius:.5rem; background:#1a1a1a;
           color:#eee; cursor:pointer; }
  button:active { background:#333; }
  button.primary { background:#1d4ed8; border-color:#3b82f6; }
  #msg { position:fixed; top:.6rem; left:.8rem; font-size:.85rem; color:#7f7;
         text-shadow:0 0 4px #000; }
  label { display:flex; align-items:center; gap:.4rem; font-size:.85rem;
          flex:1 1 auto; min-width:10rem; }
  input[type=range] { flex:1; }
  #ui2 { padding:.7rem; display:flex; gap:.4rem; flex-wrap:wrap;
         background:rgba(0,0,0,.6); backdrop-filter:blur(8px); }
  /* Gestures are many and small; the settings row below stays full size. */
  #ui2 button { min-width:6.5rem; padding:.6rem .5rem; font-size:.85rem; }
  #act { position:fixed; top:2.2rem; left:.8rem; font-size:.85rem; color:#fd7;
         text-shadow:0 0 4px #000; }
  #vel { position:fixed; top:3.8rem; left:.8rem; font-size:.85rem; color:#7cf;
         text-shadow:0 0 4px #000; font-variant-numeric:tabular-nums; }
  #teleop { position:fixed; top:5.4rem; left:.8rem; font-size:.85rem; color:#c9f;
            text-shadow:0 0 4px #000; font-variant-numeric:tabular-nums; }
  #ui3 { padding:.7rem; display:flex; gap:.5rem; flex-wrap:wrap;
         background:rgba(40,0,0,.6); backdrop-filter:blur(8px); }
  button.danger { background:#7f1d1d; border-color:#dc2626; }
  #ui3 .note { flex:1 1 100%; font-size:.78rem; color:#fca5a5; }
</style>
</head><body>
<canvas id="c"></canvas>
<div id="msg">starting…</div>
<div id="act"></div>
<div id="vel"></div>
<div id="teleop"></div>
<div id="panel">
  <div id="ui3" style="display:none">
    <button onclick="loco('start')">Stand &amp; balance</button>
    <button onclick="loco('stop')">Stop</button>
    <button class="danger" onclick="damp()">Damp (will sag)</button>
    <span class="note">Left stick moves, right stick turns. Let go and it stops
      walking — it keeps standing and balancing. Damp is the only thing that
      releases the joints.</span>
  </div>
  <div id="ui2">__ACTION_BUTTONS__</div>
  <div id="ui">
    <button class="primary" id="vr">Enter VR</button>
    <button id="rot">Rotate: 0&deg;</button>
    <button id="shape">Screen: Flat</button>
    <label>Size <input type="range" id="size" min="20" max="220" value="90"></label>
    <label>Distance <input type="range" id="dist" min="2" max="25" value="8"></label>
    <button onclick="document.getElementById('panel').style.display='none'">Hide</button>
  </div>
</div>
<img id="src" src="/stream.mjpg" style="display:none" alt="">

<script>
const canvas = document.getElementById('c');
const gl = canvas.getContext('webgl2', {xrCompatible:true, antialias:true});
const img = document.getElementById('src');
const msg = document.getElementById('msg');
if (!gl) msg.textContent = 'WebGL2 unavailable';

const VS = `#version 300 es
in vec3 aPos; in vec2 aUV;
uniform mat4 uProj, uView;
out vec2 vUV;
void main(){ vUV = aUV; gl_Position = uProj * uView * vec4(aPos, 1.0); }`;

const FS = `#version 300 es
precision mediump float;
in vec2 vUV; uniform sampler2D uTex; out vec4 o;
void main(){ o = texture(uTex, vUV); }`;

function shader(t, src){
  const s = gl.createShader(t);
  gl.shaderSource(s, src); gl.compileShader(s);
  if(!gl.getShaderParameter(s, gl.COMPILE_STATUS)) throw gl.getShaderInfoLog(s);
  return s;
}
const prog = gl.createProgram();
gl.attachShader(prog, shader(gl.VERTEX_SHADER, VS));
gl.attachShader(prog, shader(gl.FRAGMENT_SHADER, FS));
gl.linkProgram(prog); gl.useProgram(prog);
const uProj = gl.getUniformLocation(prog, 'uProj');
const uView = gl.getUniformLocation(prog, 'uView');

// ---- state ---------------------------------------------------------------
let imgAspect = 16/9;   // learned from the stream
let rot = 0;            // 0 / 90 / 180 / 270, applied to texture coords
let flat = true;        // flat panel by default; curved is the alternative
let sizeDeg = 90;       // angular width of the screen
let dist = 8;           // metres in front of the viewer
let vao, indexCount;

// Rotating texture coordinates rather than the mesh keeps the screen itself
// square-on to the viewer at every rotation.
function rotUV(u, v, r){
  if (r === 90)  return [v, 1-u];
  if (r === 180) return [1-u, 1-v];
  if (r === 270) return [1-v, u];
  return [u, v];
}

function buildMesh(){
  // A 90/270 turn swaps which image axis is horizontal, so the shape of the
  // screen has to swap with it or the picture comes out stretched.
  const eff = (rot === 90 || rot === 270) ? 1/imgAspect : imgAspect;
  const hFov = sizeDeg * Math.PI/180;
  const pos = [], uv = [], idx = [];

  if (flat) {
    const halfW = dist * Math.tan(hFov/2);
    const halfH = halfW / eff;
    const N = 2, M = 2;
    for (let j=0;j<=M;j++) for (let i=0;i<=N;i++){
      const u=i/N, v=j/M;
      pos.push(-halfW + 2*halfW*u, halfH - 2*halfH*v, -dist);
      const [tu,tv] = rotUV(u, v, rot); uv.push(tu, tv);
    }
    for (let j=0;j<M;j++) for (let i=0;i<N;i++){
      const a=j*(N+1)+i, b=a+1, c=a+N+1, d=c+1;
      idx.push(a,c,b, b,c,d);
    }
  } else {
    const vFov = 2*Math.atan(Math.tan(hFov/2)/eff);
    const N = 64, M = 40;
    for (let j=0;j<=M;j++) for (let i=0;i<=N;i++){
      const u=i/N, v=j/M;
      const yaw=(u-0.5)*hFov, pitch=(0.5-v)*vFov;
      pos.push(dist*Math.sin(yaw)*Math.cos(pitch),
               dist*Math.sin(pitch),
               -dist*Math.cos(yaw)*Math.cos(pitch));
      const [tu,tv] = rotUV(u, v, rot); uv.push(tu, tv);
    }
    for (let j=0;j<M;j++) for (let i=0;i<N;i++){
      const a=j*(N+1)+i, b=a+1, c=a+N+1, d=c+1;
      idx.push(a,c,b, b,c,d);
    }
  }

  if (vao) gl.deleteVertexArray(vao);
  vao = gl.createVertexArray(); gl.bindVertexArray(vao);
  const pb=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, pb);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(pos), gl.STATIC_DRAW);
  const aPos=gl.getAttribLocation(prog,'aPos');
  gl.enableVertexAttribArray(aPos); gl.vertexAttribPointer(aPos,3,gl.FLOAT,false,0,0);
  const ub=gl.createBuffer(); gl.bindBuffer(gl.ARRAY_BUFFER, ub);
  gl.bufferData(gl.ARRAY_BUFFER, new Float32Array(uv), gl.STATIC_DRAW);
  const aUV=gl.getAttribLocation(prog,'aUV');
  gl.enableVertexAttribArray(aUV); gl.vertexAttribPointer(aUV,2,gl.FLOAT,false,0,0);
  const ib=gl.createBuffer(); gl.bindBuffer(gl.ELEMENT_ARRAY_BUFFER, ib);
  gl.bufferData(gl.ELEMENT_ARRAY_BUFFER, new Uint16Array(idx), gl.STATIC_DRAW);
  indexCount = idx.length;
  gl.bindVertexArray(null);
}
buildMesh();

const rotBtn = document.getElementById('rot');
rotBtn.onclick = () => {
  rot = (rot + 90) % 360;
  rotBtn.textContent = 'Rotate: ' + rot + '\\u00b0';
  buildMesh();
};
const shapeBtn = document.getElementById('shape');
shapeBtn.onclick = () => {
  flat = !flat;
  shapeBtn.textContent = 'Screen: ' + (flat ? 'Flat' : 'Curved');
  buildMesh();
};
document.getElementById('size').oninput = e => { sizeDeg = +e.target.value; buildMesh(); };
document.getElementById('dist').oninput = e => { dist = +e.target.value; buildMesh(); };

// ---- video texture -------------------------------------------------------
const tex = gl.createTexture();
gl.bindTexture(gl.TEXTURE_2D, tex);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_MIN_FILTER, gl.LINEAR);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_S, gl.CLAMP_TO_EDGE);
gl.texParameteri(gl.TEXTURE_2D, gl.TEXTURE_WRAP_T, gl.CLAMP_TO_EDGE);
gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,1,1,0,gl.RGB,gl.UNSIGNED_BYTE,
              new Uint8Array([18,18,18]));
let frames = 0, t0 = performance.now();
function upload(){
  if (img.complete && img.naturalWidth) {
    gl.bindTexture(gl.TEXTURE_2D, tex);
    // An <img> fed by MJPEG keeps updating in place, so re-uploading every
    // frame is what makes this live rather than a still.
    gl.texImage2D(gl.TEXTURE_2D,0,gl.RGB,gl.RGB,gl.UNSIGNED_BYTE,img);
    const a = img.naturalWidth / img.naturalHeight;
    if (Math.abs(a - imgAspect) > 0.01) { imgAspect = a; buildMesh(); }
    frames++;
  }
}

function draw(proj, view){
  gl.useProgram(prog); gl.bindVertexArray(vao);
  gl.uniformMatrix4fv(uProj,false,proj);
  gl.uniformMatrix4fv(uView,false,view);
  gl.bindTexture(gl.TEXTURE_2D, tex);
  gl.drawElements(gl.TRIANGLES, indexCount, gl.UNSIGNED_SHORT, 0);
}

// ---- flat preview before entering VR -------------------------------------
let yaw=0, pitch=0, dragging=false, lx=0, ly=0;
canvas.addEventListener('pointerdown', e => { dragging=true; lx=e.clientX; ly=e.clientY; });
addEventListener('pointerup', () => dragging=false);
addEventListener('pointermove', e => {
  if(!dragging) return;
  yaw -= (e.clientX-lx)*0.004; pitch -= (e.clientY-ly)*0.004;
  pitch = Math.max(-1.4, Math.min(1.4, pitch));
  lx=e.clientX; ly=e.clientY;
});
function persp(fovy, aspect, near, far){
  const f=1/Math.tan(fovy/2), nf=1/(near-far);
  return new Float32Array([f/aspect,0,0,0, 0,f,0,0, 0,0,(far+near)*nf,-1, 0,0,2*far*near*nf,0]);
}
function viewMat(y,p){
  const cy=Math.cos(y),sy=Math.sin(y),cp=Math.cos(p),sp=Math.sin(p);
  return new Float32Array([cy,sy*sp,-sy*cp,0, 0,cp,sp,0, sy,-cy*sp,cy*cp,0, 0,0,0,1]);
}
function preview(){
  if (xrSession) return;
  const w=canvas.clientWidth, h=canvas.clientHeight;
  if (canvas.width!==w || canvas.height!==h){ canvas.width=w; canvas.height=h; }
  upload();
  gl.bindFramebuffer(gl.FRAMEBUFFER, null);
  gl.viewport(0,0,canvas.width,canvas.height);
  gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  draw(persp(1.2, canvas.width/canvas.height, 0.1, 100), viewMat(yaw,pitch));
  const now=performance.now();
  if (now-t0 > 1000){
    msg.textContent = `${(frames*1000/(now-t0)).toFixed(0)} fps · `
      + `${img.naturalWidth}x${img.naturalHeight}`
      + (navigator.xr ? '' : ' · WebXR needs https');
    frames=0; t0=now;
  }
  requestAnimationFrame(preview);
}
requestAnimationFrame(preview);

// ---- robot control -------------------------------------------------------
const actEl = document.getElementById('act');
let lastAct = 0;
function act(name){
  // The server also rate-limits, but stopping repeats here avoids firing a
  // burst every frame while a trigger is held down.
  const now = performance.now();
  if (now - lastAct < 1500) return;
  lastAct = now;
  actEl.textContent = name + ' …';
  fetch('/action?name=' + encodeURIComponent(name))
    .then(r => r.json())
    .then(j => { actEl.textContent = name + ': ' + j.detail;
                 setTimeout(() => actEl.textContent = '', 2500); })
    .catch(e => { actEl.textContent = name + ': ' + e; });
}

// ---- locomotion ----------------------------------------------------------
const WALK = __WALK_ENABLED__;
const velEl = document.getElementById('vel');
if (WALK) document.getElementById('ui3').style.display = 'flex';

function loco(cmd){
  return fetch('/loco?cmd=' + encodeURIComponent(cmd))
    .then(r => r.json())
    .then(j => { actEl.textContent = cmd + ': ' + j.detail;
                 setTimeout(() => actEl.textContent = '', 2500); });
}

// Damp releases the joints. A humanoid that is standing will sag and probably
// go over, so this asks first and lives nowhere near a controller button.
function damp(){
  if (confirm('Damp releases the joints. A standing robot will sag and may fall.\\n\\nIs someone supporting it?'))
    loco('damp');
}

// Grip is button 1 in the xr-standard mapping, thumbstick is axes 2 and 3.
const GRIP = 1;
const REQUIRE_GRIP = __DEADMAN__;
// Matches the server's deadzone. Only used to decide whether the stick is being
// touched at all; the authoritative shaping happens in locomotion.py.
const DEADZONE = 0.15;
let driveAt = 0, driving = false, driveInFlight = false;

function pollDrive(session){
  if (!WALK) return;
  let vx = 0, vy = 0, w = 0, grip = false;

  for (const src of session.inputSources){
    const gp = src.gamepad;
    if (!gp) continue;
    if (gp.buttons[GRIP] && gp.buttons[GRIP].pressed) grip = true;
    const ax = gp.axes;
    if (ax.length < 4) continue;
    // Sticks report forward and left as negative, so every axis is inverted
    // to reach the robot's convention: +x ahead, +y left, +yaw counterclockwise.
    if (src.handedness === 'left'){ vx = -ax[3]; vy = -ax[2]; }
    else if (src.handedness === 'right'){ w = -ax[2]; }
  }

  // The stick is spring centred, so "let go" and "commanded zero" are the same
  // event. That is the deadman, and it is why this feels like the Unitree
  // controller: push to walk, release to stop, and stopping is a stop -- the
  // robot holds its stance rather than going limp.
  let active = Math.abs(vx) > DEADZONE || Math.abs(vy) > DEADZONE ||
               Math.abs(w) > DEADZONE;
  if (REQUIRE_GRIP && !grip) active = false;
  if (!active){ vx = 0; vy = 0; w = 0; }

  const now = performance.now();
  if (now - driveAt < 100) return;          // 10 Hz is plenty for walking
  driveAt = now;

  // Once stopped, stop talking. The release still sends one explicit zero;
  // after that silence is itself a stop signal to the server's watchdog.
  if (!active && !driving) return;
  driving = active;

  // Same backpressure fix as pollTeleop: skip rather than pile up if the
  // previous /drive hasn't resolved yet.
  if (driveInFlight) return;
  driveInFlight = true;

  fetch(`/drive?vx=${vx.toFixed(3)}&vy=${vy.toFixed(3)}&w=${w.toFixed(3)}`)
    .then(r => r.json())
    .then(j => {
      velEl.textContent = j.ok
        ? `${j.vx>=0?'+':''}${j.vx.toFixed(2)} m/s  ${j.vy>=0?'+':''}${j.vy.toFixed(2)} side  ${j.w>=0?'+':''}${j.w.toFixed(2)} rad/s`
        : 'drive: ' + j.detail;
      if (!active) setTimeout(() => velEl.textContent = 'standing', 400);
    })
    .finally(() => { driveInFlight = false; })
    .catch(e => { velEl.textContent = 'drive: ' + e; });
}

// Anything that takes the operator's attention away is a stop. The server would
// time out on its own, but saying so explicitly stops the robot sooner.
function releaseDrive(){
  if (!WALK || !driving) return;
  driving = false;
  fetch('/drive?vx=0&vy=0&w=0').catch(() => {});
}
addEventListener('visibilitychange', () => { if (document.hidden) { releaseDrive(); releaseTeleop(); } });
addEventListener('pagehide', () => { releaseDrive(); releaseTeleop(); });
addEventListener('blur', () => { releaseDrive(); releaseTeleop(); });

// Controller buttons, because in immersive VR the HTML overlay is not drawn --
// only WebGL is. Physical buttons are the only in-headset control surface.
// Grip is absent from this map: while walking is enabled it is the deadman,
// while teleop is enabled it is the per-hand clutch (read continuously in
// pollTeleop, not as a discrete press here), and an arm gesture firing every
// time you take hold of the robot would be its own hazard either way.
const TELEOP = __TELEOP_ENABLED__;
const RECORD = __RECORD_ENABLED__;
const TELEOP_HANDS = __TELEOP_HANDS__;
// When both hands are teleop-driven, grip has nothing free to host STOP on
// (both are claimed by their own clutch) -- it falls back to the A/X button
// instead, on either hand. Still reachable with the thumb without letting
// go of grip, same "no aim" quality, just a different button once grip
// itself is fully spoken for.
const BOTH_HANDS_DRIVEN = TELEOP_HANDS.includes('left') && TELEOP_HANDS.includes('right');
const BUTTON_MAP = {
  0: 'wave_above_head',    // trigger (unless TELEOP, which repurposes it below)
  4: 'right_hand_up',      // A / X
  5: 'shake_hand',         // B / Y
};
const pressed = {};
function pollControllers(session){
  for (const src of session.inputSources){
    const gp = src.gamepad;
    if (!gp) continue;
    gp.buttons.forEach((b, i) => {
      const key = src.handedness + i;
      if (i === GRIP){
        // release_arm-on-grip is a convenience for the no-teleop, no-walk
        // case only; with teleop on, grip is spoken for by the clutch --
        // EXCEPT on a hand TELEOP_HANDS doesn't actually drive, which is
        // free for the STOP button instead of sitting idle.
        if (TELEOP && !TELEOP_HANDS.includes(src.handedness)) {
          if (b.pressed && !pressed[key]) estop(gp);
          pressed[key] = b.pressed;
          return;
        }
        if (!TELEOP && !WALK && b.pressed && !pressed[key]) act('release_arm');
        pressed[key] = b.pressed;
        return;
      }
      if (i === 3){
        // Thumbstick click: nothing unless recording is enabled, since it is
        // easy to hit by accident while steering the stick.
        if (RECORD && b.pressed && !pressed[key]) toggleRecord();
        pressed[key] = b.pressed;
        return;
      }
      if (i === 4 && TELEOP && BOTH_HANDS_DRIVEN){
        if (b.pressed && !pressed[key]) estop(gp);
        pressed[key] = b.pressed;
        return;
      }
      // No dexterous hand exists yet -- the trigger fires the existing
      // 'clamp' canned gesture as a placeholder "close hand" cue while
      // teleop is on, not a real grasp.
      const action = (i === 0 && TELEOP) ? 'clamp' : BUTTON_MAP[i];
      if (b.pressed && !pressed[key] && action) act(action);
      pressed[key] = b.pressed;
    });
  }
}

// ---- arm teleop ------------------------------------------------------------
// One HTTP call per hand per poll, same trade quest_stream.py's other
// controls already make: on a local network the latency is small enough that
// plain fetch beats the complexity of a second transport, and the server
// answers with what it actually did with the pose -- engaged or frozen,
// current angles -- so the HUD reflects the robot, not the request.
let teleopAt = 0;
let teleopStatus = null;
let recording = false;
const teleopEl = document.getElementById('teleop');

// The #teleop text HUD below is not a reliable feedback channel: like the
// button labels, ordinary HTML is not composited into the headset once an
// immersive session starts, only the WebGL canvas is. Controller vibration
// is the one channel that reliably reaches the operator in VR, so grip
// state is confirmed by feel, not by text nobody can see -- a strong pulse
// the instant grip is physically pressed (confirms the right button was
// found, before any server round trip), a lighter one on release.
function haptic(gp, intensity, ms){
  try {
    const act = gp && gp.hapticActuators && gp.hapticActuators[0];
    if (act && act.pulse) act.pulse(intensity, ms);
  } catch (e) { /* not supported on this controller -- silently skip */ }
}
const lastLocalGrip = { left: false, right: false };

// Taking the headset off does not necessarily end the WebXR session, and it
// fires none of the page-level events (visibilitychange/blur/pagehide are
// for the 2D browser tab, not the XR session) -- so none of those stop
// anything here. session.visibilityState is the signal that actually
// reflects the headset, and releasing on it is what stops a stale grip
// (e.g. a controller left resting against something, still tracked, still
// reporting pressed) from continuing to drive the arm once nobody is
// looking. Confirmed on hardware: without this, the arm kept moving for
// close to a minute after VR was exited, and had to be stopped by
// unplugging the robot's ethernet -- there was no software path that could
// reach it. This is that path, and it is deliberately not the only one:
// see also the xrSession 'visibilitychange' listener at session start, and
// MAX_CONTINUOUS_ENGAGE_S in arm_teleop.py, which bounds it from the robot
// side regardless of what the browser does or fails to do.
function releaseTeleop(){
  if (!TELEOP) return;
  for (const hand of ['left', 'right']) {
    fetch(`/teleop?hand=${hand}&px=0&py=0&pz=0&qw=1&qx=0&qy=0&qz=0&grip=0`)
      .catch(() => {});
  }
}

// The reachable-without-thinking stop: no aim, no menu, one press. Lives on
// left grip while left is not itself a driven hand (built for right-hand-
// only teleop, where left grip would otherwise sit idle); once both hands
// are driven and grip has nothing free to claim, it falls back to the A/X
// button (4) instead -- see BOTH_HANDS_DRIVEN above. TELEOP_HANDS is what
// keeps grip from silently doing the wrong thing (colliding with a real
// clutch) instead of just not firing. /estop does more than releasing
// weight would on its own: it hands the joints back to the robot's own
// 'ai' controller, which matters on a robot where nothing else may be
// holding them.
function estop(gp){
  haptic(gp, 1.0, 200);
  fetch('/estop').then(r => r.json()).then(j => {
    msg.textContent = 'STOP: ' + j.detail;
  }).catch(() => {});
  if (xrSession) xrSession.end();
}

// Per hand, not global: true while that hand's most recent /teleop fetch is
// still in flight. Without this, the 33ms timer above fires a new fetch on
// schedule no matter how many earlier ones are still unresolved -- fine on
// a fast link, but over a slow or congested one (this laptop's wifi radio
// doing double duty as both a client and a hosted hotspot, for one) each
// request can take far longer than 33ms, and nothing was stopping them from
// piling up without bound. Measured on hardware: a 30-40 SECOND lag between
// moving a hand and the arm responding -- far too large to be the control
// loop or IK solve (milliseconds), and exactly what an ever-growing queue of
// stale, already-outdated requests would produce. This also explains grip
// readings that looked stuck at false: a real grip=true press could be
// sitting behind dozens of older queued requests for tens of seconds before
// ever reaching the server. Skipping a poll when the previous one for that
// hand has not resolved yet means a slow link sends fewer updates, but the
// next one sent always carries the CURRENT gamepad state, not a queued one.
const teleopInFlight = { left: false, right: false };

function pollTeleop(session, frame){
  if (!TELEOP) return;
  if (session.visibilityState !== 'visible') { releaseTeleop(); return; }
  const now = performance.now();
  if (now - teleopAt < 33) return;   // ~30 Hz
  teleopAt = now;

  for (const src of session.inputSources){
    if (src.handedness !== 'left' && src.handedness !== 'right') continue;
    if (teleopInFlight[src.handedness]) continue;
    const space = src.gripSpace || src.targetRaySpace;
    if (!space) continue;
    const pose = frame.getPose(space, xrRefSpace);
    if (!pose) continue;
    const p = pose.transform.position, o = pose.transform.orientation;
    const gp = src.gamepad;
    const grip = !!(gp && gp.buttons[GRIP] && gp.buttons[GRIP].pressed);

    if (grip !== lastLocalGrip[src.handedness]) {
      lastLocalGrip[src.handedness] = grip;
      haptic(gp, grip ? 0.8 : 0.3, grip ? 70 : 35);
    }

    teleopInFlight[src.handedness] = true;
    fetch(`/teleop?hand=${src.handedness}`
         + `&px=${p.x.toFixed(4)}&py=${p.y.toFixed(4)}&pz=${p.z.toFixed(4)}`
         + `&qw=${o.w.toFixed(4)}&qx=${o.x.toFixed(4)}&qy=${o.y.toFixed(4)}&qz=${o.z.toFixed(4)}`
         + `&grip=${grip ? 1 : 0}`)
      .then(r => r.json())
      .then(j => { if (j.ok) { teleopStatus = j; updateTeleopHud(); } })
      .catch(() => {})
      .finally(() => { teleopInFlight[src.handedness] = false; });
  }
}

function updateTeleopHud(){
  const eng = (teleopStatus && teleopStatus.engaged) || {};
  const parts = Object.keys(eng).map(h => `${h}:${eng[h] ? 'ENGAGED' : 'free'}`);
  const w = teleopStatus ? Number(teleopStatus.weight || 0).toFixed(2) : '0.00';
  const rec = recording ? '  ● REC' : '';
  teleopEl.textContent = (TELEOP ? parts.join('  ') + `  w=${w}` : '') + rec;
}

function toggleRecord(){
  const cmd = recording ? 'stop' : 'start';
  fetch('/record?cmd=' + cmd)
    .then(r => r.json())
    .then(j => {
      if (j.ok) recording = (cmd === 'start');
      actEl.textContent = `record ${cmd}: ${j.detail}`;
      setTimeout(() => actEl.textContent = '', 2500);
      updateTeleopHud();
    })
    .catch(e => { actEl.textContent = 'record: ' + e; });
}

// ---- WebXR ---------------------------------------------------------------
let xrSession=null, xrRefSpace=null;
const vrBtn=document.getElementById('vr');
if (!navigator.xr){ vrBtn.textContent='VR needs https'; vrBtn.disabled=true; }
else navigator.xr.isSessionSupported('immersive-vr').then(ok=>{
  if(!ok){ vrBtn.textContent='VR not supported'; vrBtn.disabled=true; }
});
vrBtn.onclick = async () => {
  if (xrSession){ xrSession.end(); return; }
  try {
    xrSession = await navigator.xr.requestSession('immersive-vr',
                  {optionalFeatures:['local-floor']});
  } catch(e){ msg.textContent='VR failed: '+e; return; }
  vrBtn.textContent='Exit VR';
  await gl.makeXRCompatible();
  xrSession.updateRenderState({ baseLayer: new XRWebGLLayer(xrSession, gl) });
  xrRefSpace = await xrSession.requestReferenceSpace('local');
  xrSession.addEventListener('end', () => {
    releaseDrive();
    releaseTeleop();
    xrSession=null; vrBtn.textContent='Enter VR'; requestAnimationFrame(preview);
  });
  // Event-driven, not just the per-frame check in pollTeleop: fires the
  // instant the headset comes off, rather than waiting for the next poll.
  xrSession.addEventListener('visibilitychange', () => {
    if (xrSession.visibilityState !== 'visible') releaseTeleop();
  });
  xrSession.requestAnimationFrame(onXRFrame);
};
function onXRFrame(t, frame){
  const session=frame.session;
  session.requestAnimationFrame(onXRFrame);
  pollControllers(session);
  pollDrive(session);
  pollTeleop(session, frame);
  const pose=frame.getViewerPose(xrRefSpace);
  if(!pose) return;
  upload();
  const layer=session.renderState.baseLayer;
  gl.bindFramebuffer(gl.FRAMEBUFFER, layer.framebuffer);
  gl.clearColor(0,0,0,1); gl.clear(gl.COLOR_BUFFER_BIT|gl.DEPTH_BUFFER_BIT);
  for (const view of pose.views){
    const vp=layer.getViewport(view);
    gl.viewport(vp.x, vp.y, vp.width, vp.height);
    draw(view.projectionMatrix, view.transform.inverse.matrix);
  }
}
</script>
</body></html>
"""
