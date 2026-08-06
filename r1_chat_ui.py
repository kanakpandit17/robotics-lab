"""Gradio chat UI to command R1 in Isaac Sim. Type natural language; a rule-based parser turns
it into {vx, turn, arm} and writes r1_cmd.json, which the running r1_chat_sim.py reads each frame.
"""
import json
import os

import gradio as gr

CMD_FILE = r"C:\Users\aimat\Desktop\humanoid\r1_cmd.json"

state = {"seq": 0, "vx": 0.0, "turn": 0.0, "arm": "home", "cam_dist": 6.0, "cam_height": 3.5, "cam_yaw": 0.7, "stand": 0, "nav": 0}


def write_cmd():
    state["seq"] += 1
    with open(CMD_FILE, "w") as f:
        json.dump(state, f)


def parse(text):
    """Update the persistent command state from a natural-language line."""
    t = " " + text.lower() + " "

    # ---- camera (checked first so "look left" != "turn left") ----
    is_cam = any(w in t for w in ["look", "zoom", "camera", "view", "closer", "further", "overview", "wide"])
    if "zoom in" in t or "closer" in t:
        state["cam_dist"] = max(2.5, state["cam_dist"] - 2.0)
    if "zoom out" in t or "further" in t or "back up" in t:
        state["cam_dist"] = min(25.0, state["cam_dist"] + 2.0)
    if "overview" in t or "wide" in t or "top view" in t:
        state["cam_dist"], state["cam_height"] = 16.0, 12.0
    if is_cam and "left" in t:
        state["cam_yaw"] += 0.5
    if is_cam and "right" in t:
        state["cam_yaw"] -= 0.5
    if is_cam and ("up" in t or "higher" in t or "above" in t):
        state["cam_height"] = min(20.0, state["cam_height"] + 1.5)
    if is_cam and ("down" in t or "lower" in t):
        state["cam_height"] = max(0.5, state["cam_height"] - 1.5)

    # ---- stand up (reset R1 upright) ----
    if any(w in t for w in ["stand", "get up", "upright", "reset"]):
        state["stand"] += 1
        state["vx"] = 0.0
        state["turn"] = 0.0
        state["arm"] = "home"

    # ---- autonomous navigation to a (new random) goal ----
    is_nav = any(w in t for w in ["goal", "navigate", "auto", "explore"])
    if is_nav:
        state["nav"] += 1
        state["vx"] = 0.0
        state["turn"] = 0.0

    # ---- base motion (skip left/right if it was a camera 'look') ----
    if any(w in t for w in ["stop", "halt", "freeze", "stay"]):
        state["vx"] = 0.0
        state["turn"] = 0.0
    if not is_nav and any(w in t for w in ["forward", "ahead", "straight", "drive", "go ", "move"]):
        state["vx"] = 1.0
    if any(w in t for w in ["back", "reverse"]):
        state["vx"] = -1.0
    if not is_cam and "left" in t:
        state["turn"] = 1.0
    if not is_cam and "right" in t:
        state["turn"] = -1.0
    if "slow" in t:
        state["vx"] *= 0.5
    if any(w in t for w in ["fast", "faster"]):
        state["vx"] = 1.0

    # arms
    if "wave" in t:
        state["arm"] = "wave"
    elif any(w in t for w in ["raise", "hands up", "arms up", "lift"]):
        state["arm"] = "raise"
    elif "reach" in t:
        state["arm"] = "reach"
    elif any(w in t for w in ["home", "rest", "lower", "arms down", "relax", "down"]):
        state["arm"] = "home"
    if any(w in t for w in ["stop", "halt"]):
        state["arm"] = "home"

    return state["vx"], state["turn"], state["arm"]


def send(text, history):
    if not text.strip():
        return history, ""
    vx, turn, arm = parse(text)
    write_cmd()
    move = "stopped" if vx == 0 and turn == 0 else f"drive vx={vx:+.1f}, turn={turn:+.1f}"
    reply = f"🤖 R1: {move}, arms='{arm}'  ·  📷 dist={state['cam_dist']:.0f}"
    history = history + [
        {"role": "user", "content": text},
        {"role": "assistant", "content": reply},
    ]
    return history, ""


with gr.Blocks(title="R1 Chat Control") as demo:
    gr.Markdown(
        "# 🤖 R1 Chat Control\n"
        "The camera **auto-follows R1** — no viewport wrestling needed. Control robot + camera by typing.\n\n"
        "**Move:** `drive forward` · `turn left` · `wave` · `reach` · `stop`  \n"
        "**Camera:** `zoom in` · `zoom out` · `look left` · `look right` · `overview`"
    )
    chat = gr.Chatbot(height=360)
    txt = gr.Textbox(placeholder="e.g. drive forward and wave  /  zoom in", show_label=False, autofocus=True)
    with gr.Row():
        for label in ["go to goal", "stand up", "drive forward", "turn left", "turn right", "wave", "stop"]:
            gr.Button(label, size="sm").click(lambda h, l=label: send(l, h), [chat], [chat, txt])
    with gr.Row():
        for label in ["zoom in", "zoom out", "look left", "look right", "overview"]:
            gr.Button(label, size="sm").click(lambda h, l=label: send(l, h), [chat], [chat, txt])
    txt.submit(send, [txt, chat], [chat, txt])

if __name__ == "__main__":
    # ensure a home command exists at startup
    if not os.path.isfile(CMD_FILE):
        json.dump(state, open(CMD_FILE, "w"))
    demo.launch(server_name="127.0.0.1", server_port=7860, inbrowser=False)
