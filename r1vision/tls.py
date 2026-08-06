"""Self-signed TLS so WebXR will run.

`navigator.xr` only exists in a secure context. Plain http:// over a LAN IP is
not one, so the Quest browser reports "WebXR not available" no matter what the
headset supports. HTTPS with a self-signed certificate is enough -- the browser
warns once, you accept, and the API appears.

The certificate lists every local IP as a SAN so the same file works whichever
network the headset reaches this Mac on.
"""

from __future__ import annotations

import shutil
import ssl
import subprocess
from pathlib import Path

CERT_DIR = Path("certs")
CERT = CERT_DIR / "r1vision.crt"
KEY = CERT_DIR / "r1vision.key"


def ensure_cert(ips: list[str], days: int = 3650) -> tuple[Path, Path]:
    """Create the cert if missing, or if this machine has new addresses."""
    CERT_DIR.mkdir(exist_ok=True)

    if CERT.exists() and KEY.exists():
        try:
            text = subprocess.run(
                ["openssl", "x509", "-in", str(CERT), "-noout", "-text"],
                capture_output=True, text=True, timeout=10,
            ).stdout
            if all(ip in text for ip in ips):
                return CERT, KEY
        except (subprocess.SubprocessError, OSError):
            pass

    if not shutil.which("openssl"):
        raise RuntimeError("openssl not found; re-run with --http (no VR)")

    san = ",".join([f"IP:{ip}" for ip in ips] + ["IP:127.0.0.1", "DNS:localhost"])
    subprocess.run(
        [
            "openssl", "req", "-x509", "-newkey", "rsa:2048", "-nodes",
            "-keyout", str(KEY), "-out", str(CERT),
            "-days", str(days), "-subj", "/CN=r1vision",
            "-addext", f"subjectAltName={san}",
        ],
        check=True, capture_output=True, timeout=60,
    )
    return CERT, KEY


def wrap(sock, ips: list[str]):
    """Wrap a listening socket in TLS."""
    cert, key = ensure_cert(ips)
    ctx = ssl.SSLContext(ssl.PROTOCOL_TLS_SERVER)
    ctx.load_cert_chain(certfile=str(cert), keyfile=str(key))
    return ctx.wrap_socket(sock, server_side=True)
