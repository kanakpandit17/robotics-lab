"""Live fleet dashboard. Runs on the laptop, next to the scheduler.

    python -m scheduler.dashboard --config config/laptop.json
    open http://127.0.0.1:8420

Why a local server rather than a static page
--------------------------------------------
The data is on your MQTT broker, and a browser cannot open a raw MQTT socket.
This process holds the one broker connection, keeps a rolling history in memory,
and pushes it to the page over Server-Sent Events -- one long-lived HTTP
response, no websocket handshake and no client library to ship.

It subscribes to the same wildcard topics the scheduler uses, so it sees every
robot on the bus without being told they exist, and it shows commands issued by
*other* schedulers too. That is the point of a bus: the dashboard is one more
observer, not a privileged component.

Read-only by default. `--allow-control` adds Stop, MoveHome and ResetError
buttons; without it the page cannot command anything, which is what you want if
it is open on a second screen next to a live robot.
"""

from __future__ import annotations

import argparse
import json
import logging
import queue
import sys
import threading
import time
from collections import deque
from enum import Enum
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fleetcore import topics  # noqa: E402
from fleetcore.transport import build_transport  # noqa: E402
from scheduler.fleet import Fleet  # noqa: E402

log = logging.getLogger("dashboard")

HISTORY = 400          # events kept in memory; enough to scroll a shift's work
SUBSCRIBERS: set[queue.Queue] = set()
SUB_LOCK = threading.Lock()
EVENTS: deque = deque(maxlen=HISTORY)
STATES: dict[str, dict] = {}
STARTED = time.time()


def publish(kind: str, robot_id: str, text: str, level: str = "info",
            data: dict | None = None) -> None:
    """Record one line and fan it out to every open browser."""
    entry = {
        "t": time.time(),
        "kind": kind,
        "robot": robot_id,
        "text": text,
        "level": level,
        "data": data or {},
    }
    EVENTS.append(entry)
    message = dumps({"type": "event", "event": entry})
    _fanout(message)


def push_state(robot_id: str, state: dict) -> None:
    STATES[robot_id] = state
    _fanout(dumps({"type": "state", "robot": robot_id, "state": state}))


def dumps(obj) -> str:
    """JSON that cannot fail on a value the envelope happens to carry.

    Telemetry is whatever the adapter chose to put there, and the envelope
    types mix dataclasses, str Enums and occasionally an enum class. One
    unserializable value should degrade to its repr, not take down the stream
    that is meant to be showing you what went wrong.
    """
    return json.dumps(obj, default=str)


def as_dict(msg) -> dict:
    """A StateMessage as plain data, whatever shape it arrives in."""
    import dataclasses

    if dataclasses.is_dataclass(msg) and not isinstance(msg, type):
        raw = dataclasses.asdict(msg)
    elif isinstance(msg, dict):
        raw = dict(msg)
    else:
        # vars() on anything else -- an Enum member, say -- returns its private
        # internals, which then render as a robot card full of _value_ and
        # __objclass__. Better to show nothing than to show that.
        return {}
    # .value, not str(): on Python 3.11+ str(RobotState.IDLE) is
    # "RobotState.IDLE", which no consumer of this feed expects.
    return {k: (v.value if isinstance(v, Enum) else v) for k, v in raw.items()}


def _fanout(message: str) -> None:
    with SUB_LOCK:
        dead = []
        for q in SUBSCRIBERS:
            try:
                q.put_nowait(message)
            except queue.Full:
                # A browser that stopped reading must not back-pressure the
                # MQTT thread; drop it and let it reconnect.
                dead.append(q)
        for q in dead:
            SUBSCRIBERS.discard(q)


# --------------------------------------------------------------------- bus
class BusWatcher:
    """Turns bus traffic into dashboard events.

    Subscribes to states, presence and the agents' own event topic. Commands
    and replies are picked up from the event stream the agents publish, so the
    dashboard sees work issued by any scheduler, not only its own.
    """

    def __init__(self, transport):
        self.transport = transport
        self.fleet = Fleet(transport, client_id=None)

    def start(self) -> None:
        self.fleet.start()
        self.transport.subscribe(topics.all_events(), self._on_event)
        for client in self.fleet.robots:
            self._bind(client)
        threading.Thread(target=self._poll_new, daemon=True).start()

    def _poll_new(self) -> None:
        """Attach a state listener to robots as they appear."""
        known: set[str] = set()
        while True:
            for client in self.fleet.robots:
                if client.robot_id not in known:
                    known.add(client.robot_id)
                    self._bind(client)
                    publish("presence", client.robot_id,
                            f"discovered {client.robot_id}", "good")
            time.sleep(1.0)

    def _bind(self, client) -> None:
        def on_state(msg) -> None:
            state = as_dict(msg)
            push_state(client.robot_id, state)
            publish("state", client.robot_id,
                    f"state {state.get('state')}"
                    + (f" · {state['current_task']}" if state.get("current_task")
                       else ""),
                    _level_for(state.get("state")), state)

        client.on_state_change(on_state)
        # No seeding from client.state -- that attribute is the RobotState enum,
        # not the StateMessage. The agent publishes state retained, so
        # subscribing delivers the current one immediately anyway.
        last = getattr(client, "last_state", None)
        if last is not None:
            push_state(client.robot_id, as_dict(last))

    def _on_event(self, topic: str, payload: bytes) -> None:
        robot = topics.robot_from_topic(topic) or "?"
        try:
            body = json.loads(payload.decode())
        except Exception:
            publish("event", robot, payload[:200].decode("utf-8", "replace"))
            return
        kind = body.get("event", "event")
        text = body.get("message") or kind
        level = "bad" if body.get("error") else "info"
        publish(kind, robot, text, level, body)


def _level_for(state: str | None) -> str:
    return {
        "IDLE": "good", "BUSY": "busy", "STARTING": "info",
        "ERROR": "bad", "ESTOP": "bad", "OFFLINE": "bad",
    }.get(state or "", "info")


# -------------------------------------------------------------------- page
PAGE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1">
<title>MAiRA — fleet dashboard</title>
<style>
  :root{
    --bg:#f4f6f8; --panel:#fff; --panel2:#eef1f5; --ink:#111821; --soft:#4a5769;
    --mut:#6b7889; --rule:#d7dee6; --accent:#0b6b78;
    --good:#2c6e49; --goodbg:#e4f0e8; --busy:#8a5a06; --busybg:#f8eeda;
    --bad:#9c3228; --badbg:#f8e6e3; --mono:ui-monospace,"SF Mono",Menlo,monospace;
  }
  @media (prefers-color-scheme:dark){
    :root{--bg:#0c1116;--panel:#141b23;--panel2:#1b242e;--ink:#e7ecf2;
    --soft:#aebbc9;--mut:#7d8b9c;--rule:#26313d;--accent:#4bc4d1;
    --good:#6fca94;--goodbg:#162b21;--busy:#e0ab5c;--busybg:#2d2415;
    --bad:#ef8377;--badbg:#331a19;}
  }
  *{box-sizing:border-box}
  body{margin:0;background:var(--bg);color:var(--ink);
       font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",system-ui,sans-serif}
  header{display:flex;align-items:baseline;gap:1rem;flex-wrap:wrap;
         padding:1rem 1.25rem;border-bottom:1px solid var(--rule);background:var(--panel)}
  h1{margin:0;font-size:1.05rem;letter-spacing:-.01em}
  .sub{color:var(--mut);font-family:var(--mono);font-size:.75rem}
  #conn{margin-left:auto;font-family:var(--mono);font-size:.72rem;
        padding:.15rem .5rem;border-radius:.25rem}
  .wrap{display:grid;grid-template-columns:minmax(19rem,26rem) 1fr;gap:1rem;
        padding:1rem 1.25rem;align-items:start}
  @media (max-width:60rem){.wrap{grid-template-columns:1fr}}
  .card{background:var(--panel);border:1px solid var(--rule);border-radius:.4rem;
        padding:.9rem 1rem;margin-bottom:.75rem;border-left:3px solid var(--rule)}
  .card.good{border-left-color:var(--good)} .card.busy{border-left-color:var(--busy)}
  .card.bad{border-left-color:var(--bad)}
  .rid{font-weight:640;font-size:1rem}
  .pill{font-family:var(--mono);font-size:.68rem;text-transform:uppercase;
        letter-spacing:.07em;padding:.14rem .45rem;border-radius:.25rem}
  .pill.good{color:var(--good);background:var(--goodbg)}
  .pill.busy{color:var(--busy);background:var(--busybg)}
  .pill.bad{color:var(--bad);background:var(--badbg)}
  .pill.info{color:var(--mut);background:var(--panel2)}
  .row{display:flex;align-items:center;gap:.5rem;flex-wrap:wrap}
  dl{display:grid;grid-template-columns:auto 1fr;gap:.15rem .7rem;margin:.6rem 0 0;
     font-family:var(--mono);font-size:.76rem}
  dt{color:var(--mut)} dd{margin:0;overflow-wrap:anywhere}
  .joints{display:flex;flex-direction:column;gap:.2rem;margin-top:.5rem}
  .j{display:grid;grid-template-columns:1.6rem 1fr 3.6rem;gap:.4rem;align-items:center;
     font-family:var(--mono);font-size:.7rem}
  .bar{height:.5rem;background:var(--panel2);border-radius:.25rem;position:relative}
  .bar i{position:absolute;top:0;bottom:0;background:var(--accent);border-radius:.25rem}
  .bar u{position:absolute;top:-2px;bottom:-2px;width:1px;background:var(--rule);left:50%}
  #log{background:var(--panel);border:1px solid var(--rule);border-radius:.4rem;
       height:calc(100vh - 8rem);overflow-y:auto;font-family:var(--mono);font-size:.75rem}
  .le{display:grid;grid-template-columns:5.2rem 6rem 1fr;gap:.6rem;padding:.28rem .8rem;
      border-bottom:1px solid var(--rule)}
  .le:hover{background:var(--panel2)}
  .le .ts{color:var(--mut)} .le .rb{color:var(--accent)}
  .le.good .tx{color:var(--good)} .le.busy .tx{color:var(--busy)}
  .le.bad .tx{color:var(--bad)}
  .bartop{display:flex;gap:.4rem;padding:.5rem .8rem;border-bottom:1px solid var(--rule);
          background:var(--panel2);align-items:center}
  button{font:inherit;font-size:.78rem;padding:.3rem .6rem;border:1px solid var(--rule);
         border-radius:.3rem;background:var(--panel);color:var(--ink);cursor:pointer}
  button:hover{background:var(--panel2)}
  button.danger{color:var(--bad);border-color:var(--bad)}
  input[type=search]{font:inherit;font-size:.78rem;padding:.3rem .5rem;flex:1;
        border:1px solid var(--rule);border-radius:.3rem;background:var(--panel);color:var(--ink)}
  .empty{padding:1.5rem;color:var(--mut);text-align:center}
</style></head><body>

<header>
  <h1>MAiRA — fleet dashboard</h1>
  <span class="sub" id="up"></span>
  <span id="conn" class="pill bad">connecting</span>
</header>

<div class="wrap">
  <section id="robots"><div class="empty">waiting for a robot to announce itself…</div></section>
  <section>
    <div class="bartop">
      <input type="search" id="filter" placeholder="filter log…">
      <button id="pause">Pause</button>
      <button id="clear">Clear</button>
      <span class="sub" id="count"></span>
    </div>
    <div id="log"></div>
  </section>
</div>

<script>
const CONTROL = __CONTROL__;
const robotsEl = document.getElementById('robots');
const logEl = document.getElementById('log');
const connEl = document.getElementById('conn');
const filterEl = document.getElementById('filter');
const countEl = document.getElementById('count');
let paused = false, entries = [], states = {};

const fmtTime = t => new Date(t*1000).toTimeString().slice(0,8);
const cls = s => ({IDLE:'good',BUSY:'busy',STARTING:'info',
                   ERROR:'bad',ESTOP:'bad',OFFLINE:'bad'})[s] || 'info';

function num(v, d=3){ return (typeof v === 'number') ? v.toFixed(d) : v; }

function renderRobots(){
  const ids = Object.keys(states).sort();
  if(!ids.length) return;
  robotsEl.innerHTML = ids.map(id => {
    const s = states[id] || {}, t = s.telemetry || {};
    const k = cls(s.state);
    // Joint bars are drawn against +/-pi, which is the range neurapy reports
    // in; the tick marks the zero position so a drifted joint is obvious.
    const joints = Array.isArray(t.joints) ? t.joints.map((v,i) => {
      const pct = Math.max(0, Math.min(100, (v/Math.PI + 1) * 50));
      const lo = Math.min(50, pct), w = Math.abs(pct - 50);
      const nm = (t.joint_names&&t.joint_names[i]) ? t.joint_names[i].replace('joint','J') : 'J'+(i+1);
      return `<div class="j"><span>${nm}</span>
        <div class="bar"><u></u><i style="left:${lo}%;width:${w}%"></i></div>
        <span>${num(v)}</span></div>`;
    }).join('') : '';
    const errs = (t.errors||[]).length, warns = (t.warnings||[]).length;
    return `<div class="card ${k}">
      <div class="row">
        <span class="rid">${id}</span>
        <span class="pill ${t.moving ? 'busy' : (t.teach_mode ? 'info' : 'good')}"
              style="font-size:.8rem">${
          t.moving ? '\u25cf MOVING' : (t.teach_mode ? 'TEACH \u00b7 still' : 'STILL')
        }</span>
        <span class="pill ${k}" title="agent task state">task: ${s.state||'?'}</span>
        ${s.current_task ? `<span class="pill busy">${s.current_task}</span>` : ''}
        ${errs ? `<span class="pill bad">${errs} error${errs>1?'s':''}</span>` : ''}
        ${warns ? `<span class="pill busy">${warns} warn</span>` : ''}
        ${t.estopped ? '<span class="pill bad">ESTOP</span>' : ''}
        ${t.teach_mode ? '<span class="pill busy">TEACH MODE</span>' : ''}
        ${t.moving ? '<span class="pill busy">MOVING</span>' : ''}
        ${t.ros_connected===false ? '<span class="pill bad">ROS DOWN</span>' : ''}
      </div>
      <div style="font-family:var(--mono);font-size:1.35rem;margin:.4rem 0;
                  color:${t.moving?'var(--busy)':'var(--mut)'}">
        ${t.peak_joint_speed!=null ? t.peak_joint_speed.toFixed(4)+' rad/s' : '—'}
      </div>
      <dl>
        <dt>model</dt><dd>${s.model||'—'}</dd>
        <dt>sdk</dt><dd>${s.sdk_connected ? 'connected' : 'not connected'}</dd>
        <dt>motion</dt><dd>${t.moving ? 'MOVING' : 'still'}${t.peak_joint_speed!=null ? '  ('+t.peak_joint_speed.toFixed(4)+' rad/s)' : ''}</dd>
        <dt>source</dt><dd>${t.ros_connected ? 'rosbridge live' : (t.motion_status||'—')}</dd>
        <dt>mode</dt><dd>${t.teach_mode ? 'TEACH' : (t.automatic_mode ? 'AUTOMATIC' : '—')}</dd>
        <dt>tcp</dt><dd>${Array.isArray(t.tcp) ? t.tcp.map(v=>num(v,3)).join('  ') : '—'}</dd>
        ${t.in_collision ? '<dt>collision</dt><dd>YES</dd>' : ''}
      </dl>
      ${joints ? `<div class="joints">${joints}</div>` : ''}
      ${t.read_errors ? `<dl><dt>read fail</dt><dd>${
          Object.entries(t.read_errors).map(([k,v])=>k+': '+v).join('<br>')}</dd></dl>` : ''}
      ${(t.errors||[]).length ? `<dl><dt>errors</dt><dd>${
          t.errors.map(e=>String(e)).join('<br>')}</dd></dl>` : ''}
      ${CONTROL ? `<div class="row" style="margin-top:.6rem">
        <button onclick="cmd('${id}','Stop')" class="danger">Stop</button>
        <button onclick="cmd('${id}','MoveHome')">MoveHome</button>
        <button onclick="cmd('${id}','ResetError')">ResetError</button>
      </div>` : ''}
    </div>`;
  }).join('');
}

function cmd(robot, command){
  if(!confirm(`Send ${command} to ${robot}?`)) return;
  fetch(`/command?robot=${encodeURIComponent(robot)}&cmd=${encodeURIComponent(command)}`,
        {method:'POST'}).then(r=>r.json()).then(j=>{
    if(!j.ok) alert(`${command} failed: ${j.detail}`);
  });
}

function renderLog(){
  const f = filterEl.value.toLowerCase();
  const shown = f ? entries.filter(e =>
      (e.robot+e.kind+e.text).toLowerCase().includes(f)) : entries;
  countEl.textContent = `${shown.length}/${entries.length}`;
  logEl.innerHTML = shown.slice(-300).map(e =>
    `<div class="le ${e.level}"><span class="ts">${fmtTime(e.t)}</span>`
    + `<span class="rb">${e.robot}</span><span class="tx">${
        e.text.replace(/[<>&]/g, c => ({'<':'&lt;','>':'&gt;','&':'&amp;'}[c]))
      }</span></div>`).join('');
  if(!paused) logEl.scrollTop = logEl.scrollHeight;
}

document.getElementById('pause').onclick = e => {
  paused = !paused; e.target.textContent = paused ? 'Resume' : 'Pause';
};
document.getElementById('clear').onclick = () => { entries = []; renderLog(); };
filterEl.oninput = renderLog;

function connect(){
  const es = new EventSource('/stream');
  es.onopen = () => { connEl.textContent = 'live'; connEl.className = 'pill good'; };
  es.onerror = () => {
    connEl.textContent = 'reconnecting'; connEl.className = 'pill bad';
    // EventSource retries on its own; this only reflects it in the header.
  };
  es.onmessage = ev => {
    const m = JSON.parse(ev.data);
    if(m.type === 'event'){ entries.push(m.event); if(!paused) renderLog(); }
    else if(m.type === 'state'){ states[m.robot] = m.state; renderRobots(); }
    else if(m.type === 'snapshot'){
      states = m.states; entries = m.events; renderRobots(); renderLog();
    }
  };
}
connect();
setInterval(() => {
  const s = Math.floor(performance.now()/1000);
  document.getElementById('up').textContent =
    `${Object.keys(states).length} robot(s) · dashboard up ${Math.floor(s/60)}m${s%60}s`;
}, 1000);
</script></body></html>"""


# ------------------------------------------------------------------ server
class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    watcher: BusWatcher = None       # type: ignore[assignment]
    allow_control = False

    def log_message(self, *a):
        pass

    def _send(self, body: bytes, ctype: str, code: int = 200) -> None:
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path in ("/", "/index.html"):
            page = PAGE.replace("__CONTROL__",
                                "true" if self.allow_control else "false")
            self._send(page.encode(), "text/html; charset=utf-8")
            return

        if self.path == "/stream":
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-cache")
            self.send_header("Connection", "keep-alive")
            self.end_headers()
            q: queue.Queue = queue.Queue(maxsize=500)
            with SUB_LOCK:
                SUBSCRIBERS.add(q)
            try:
                # Everything already known, so a browser opened at any moment
                # starts with the full picture rather than only what happens next.
                snapshot = dumps({"type": "snapshot",
                                       "states": STATES,
                                       "events": list(EVENTS)})
                self.wfile.write(f"data: {snapshot}\n\n".encode())
                self.wfile.flush()
                while True:
                    try:
                        msg = q.get(timeout=15)
                        self.wfile.write(f"data: {msg}\n\n".encode())
                    except queue.Empty:
                        self.wfile.write(b": keepalive\n\n")
                    self.wfile.flush()
            except (BrokenPipeError, ConnectionResetError):
                pass
            finally:
                with SUB_LOCK:
                    SUBSCRIBERS.discard(q)
            return

        if self.path == "/api/state":
            self._send(dumps(STATES).encode(), "application/json")
            return

        self.send_error(404)

    def do_POST(self):
        if not self.path.startswith("/command"):
            self.send_error(404)
            return
        if not self.allow_control:
            self._send(json.dumps({"ok": False,
                                   "detail": "dashboard is read-only; "
                                             "restart it with --allow-control"}
                                  ).encode(), "application/json", 403)
            return
        from urllib.parse import parse_qs, urlparse
        q = parse_qs(urlparse(self.path).query)
        robot = q.get("robot", [""])[0]
        command = q.get("cmd", [""])[0]
        try:
            client = self.watcher.fleet[robot]
            reply = (client.stop() if command == "Stop"
                     else client.execute(command, timeout_s=120.0))
            ok = getattr(reply, "ok", True)
            detail = getattr(reply, "detail", "") or str(getattr(reply, "result", ""))
            publish("command", robot, f"{command} -> {detail or 'ok'}",
                    "good" if ok else "bad")
        except Exception as exc:
            publish("command", robot, f"{command} failed: {exc}", "bad")
            self._send(json.dumps({"ok": False, "detail": str(exc)}).encode(),
                       "application/json")
            return
        self._send(json.dumps({"ok": bool(ok), "detail": detail}).encode(),
                   "application/json")


class Server(ThreadingHTTPServer):
    daemon_threads = True

    def handle_error(self, request, client_address):
        exc = sys.exc_info()[1]
        # A browser closing an SSE stream is normal, not an error worth a stack.
        if isinstance(exc, (BrokenPipeError, ConnectionResetError, TimeoutError)):
            return
        super().handle_error(request, client_address)


def main() -> int:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--port", type=int, default=8420)
    ap.add_argument("--bind", default="127.0.0.1",
                    help="0.0.0.0 to view it from another machine")
    ap.add_argument("--allow-control",
                    action="store_true",
                    help="add Stop / MoveHome / ResetError buttons. Off by "
                         "default so a dashboard left open cannot move a robot")
    ap.add_argument("-v", "--verbose", action="store_true")
    args = ap.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(name)-22s %(message)s",
        datefmt="%H:%M:%S")

    cfg = json.loads(Path(args.config).read_text())
    transport = build_transport(cfg.get("broker", {}), client_id="dashboard")
    transport.connect()

    watcher = BusWatcher(transport)
    watcher.start()
    Handler.watcher = watcher
    Handler.allow_control = args.allow_control

    publish("dashboard", "-", "dashboard started", "good")
    server = Server((args.bind, args.port), Handler)
    print(f"\n  dashboard  http://{args.bind}:{args.port}")
    print(f"  control    {'ENABLED' if args.allow_control else 'read-only'}")
    print("  Ctrl-C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopping")
    finally:
        transport.disconnect()
    return 0


if __name__ == "__main__":
    sys.exit(main())
