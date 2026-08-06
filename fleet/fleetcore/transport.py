"""The bus, behind an interface thin enough to swap.

`Transport` is four methods. `MqttTransport` is the real one; `LoopbackTransport`
runs the identical protocol in-process so the whole system -- scheduler, agent,
simulated robot -- can be exercised on one machine with no broker installed.
Nothing above this module knows which is in use.

Why MQTT and not a socket
-------------------------
Neither side has to be reachable. Both dial *out* to the broker, which matters
on shared venue wifi where inbound connections and client isolation are facts of
life. It also means adding a robot is a new topic namespace rather than another
address the scheduler must be told about.

Three MQTT features are load-bearing here and are the reason this is not a
hand-rolled WebSocket:

* **Last Will and Testament** -- the broker publishes `offline` for an agent that
  stopped answering. A crashed workstation cannot report its own crash.
* **Retained messages** -- a scheduler that starts an hour later immediately
  knows every robot's last state without polling anything.
* **QoS 1** -- redelivery over a link that drops. Which is why commands carry an
  id and the agent de-duplicates: at-least-once means duplicates.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Callable, Protocol

log = logging.getLogger(__name__)

MessageHandler = Callable[[str, bytes], None]


class Transport(Protocol):
    def connect(self) -> None: ...
    def disconnect(self) -> None: ...
    def publish(self, topic: str, payload: bytes, qos: int = 1,
                retain: bool = False) -> None: ...
    def subscribe(self, topic_filter: str, handler: MessageHandler,
                  qos: int = 1) -> None: ...


def _matches(topic_filter: str, topic: str) -> bool:
    """MQTT wildcard matching: `+` is one level, `#` is the rest."""
    f, t = topic_filter.split("/"), topic.split("/")
    for i, part in enumerate(f):
        if part == "#":
            return True
        if i >= len(t):
            return False
        if part != "+" and part != t[i]:
            return False
    return len(f) == len(t)


class MqttTransport:
    """paho-mqtt with reconnect, offline queueing and a will.

    Reconnection is delegated to paho's own loop rather than reimplemented:
    `reconnect_delay_set` gives exponential backoff, and subscriptions are
    replayed in `on_connect` because a broker forgets them across a session.
    """

    def __init__(self, client_id: str, host: str, port: int = 1883,
                 username: str | None = None, password: str | None = None,
                 keepalive: int = 20, will: tuple[str, bytes] | None = None,
                 tls: bool = False):
        self.client_id = client_id
        self.host, self.port = host, port
        self.keepalive = keepalive
        self._subs: list[tuple[str, MessageHandler, int]] = []
        self._connected = threading.Event()
        self._lock = threading.Lock()

        try:
            import paho.mqtt.client as mqtt
        except ImportError as exc:  # pragma: no cover - environment problem
            raise RuntimeError(
                "paho-mqtt is required for MqttTransport: pip install paho-mqtt"
            ) from exc

        # CallbackAPIVersion landed in paho 2.x; 1.x has no such argument.
        try:
            self._c = mqtt.Client(mqtt.CallbackAPIVersion.VERSION1,
                                  client_id=client_id, clean_session=True)
        except AttributeError:
            self._c = mqtt.Client(client_id=client_id, clean_session=True)

        if username:
            self._c.username_pw_set(username, password)
        if tls:
            self._c.tls_set()
        if will:
            # Retained, so the "offline" marker outlives the disconnection.
            self._c.will_set(will[0], will[1], qos=1, retain=True)

        # Keep publishing while the link is down; paho flushes on reconnect.
        self._c.max_queued_messages_set(1000)
        self._c.reconnect_delay_set(min_delay=1, max_delay=30)
        self._c.on_connect = self._on_connect
        self._c.on_disconnect = self._on_disconnect
        self._c.on_message = self._on_message

    # ------------------------------------------------------------- lifecycle
    def connect(self, timeout: float = 10.0) -> None:
        log.info("connecting to mqtt://%s:%s as %s",
                 self.host, self.port, self.client_id)
        self._c.connect_async(self.host, self.port, keepalive=self.keepalive)
        self._c.loop_start()
        if not self._connected.wait(timeout):
            # Not fatal: paho keeps retrying in the background, and an agent
            # that starts before the broker should not simply exit.
            log.warning("broker not reachable yet; retrying in background")

    def disconnect(self) -> None:
        try:
            self._c.disconnect()
        finally:
            self._c.loop_stop()

    @property
    def connected(self) -> bool:
        return self._connected.is_set()

    def wait_connected(self, timeout: float) -> bool:
        return self._connected.wait(timeout)

    # ---------------------------------------------------------------- pub/sub
    def publish(self, topic: str, payload: bytes, qos: int = 1,
                retain: bool = False) -> None:
        self._c.publish(topic, payload, qos=qos, retain=retain)

    def subscribe(self, topic_filter: str, handler: MessageHandler,
                  qos: int = 1) -> None:
        with self._lock:
            self._subs.append((topic_filter, handler, qos))
        if self._connected.is_set():
            self._c.subscribe(topic_filter, qos=qos)

    # -------------------------------------------------------------- callbacks
    def _on_connect(self, client, _userdata, _flags, rc, *_a):
        if rc != 0:
            log.error("mqtt connect refused, rc=%s", rc)
            return
        self._connected.set()
        log.info("connected to broker")
        with self._lock:
            subs = list(self._subs)
        for topic_filter, _handler, qos in subs:
            client.subscribe(topic_filter, qos=qos)

    def _on_disconnect(self, _client, _userdata, rc, *_a):
        self._connected.clear()
        if rc != 0:
            log.warning("unexpected disconnect (rc=%s); paho will retry", rc)

    def _on_message(self, _client, _userdata, msg):
        with self._lock:
            subs = list(self._subs)
        for topic_filter, handler, _qos in subs:
            if _matches(topic_filter, msg.topic):
                try:
                    handler(msg.topic, msg.payload)
                except Exception:
                    # One bad handler must not take down the network loop.
                    log.exception("handler failed for %s", msg.topic)


class LoopbackTransport:
    """An in-process broker implementing the same contract.

    Exists so the full stack can be run and tested without installing anything:
    `LoopbackBroker` is shared by a scheduler and one or more agents in the same
    interpreter. It implements retained messages and wills, because those are
    the semantics the code above depends on -- a fake that dropped them would
    pass tests the real system fails.
    """

    def __init__(self, broker: "LoopbackBroker", client_id: str,
                 will: tuple[str, bytes] | None = None):
        self.broker, self.client_id, self._will = broker, client_id, will
        self._subs: list[tuple[str, MessageHandler]] = []
        self.connected = False

    def connect(self, timeout: float = 0.0) -> None:
        self.broker.register(self)
        self.connected = True

    def disconnect(self) -> None:
        self.connected = False
        self.broker.unregister(self)

    def wait_connected(self, timeout: float) -> bool:
        return self.connected

    def publish(self, topic: str, payload: bytes, qos: int = 1,
                retain: bool = False) -> None:
        self.broker.publish(topic, payload, retain=retain)

    def subscribe(self, topic_filter: str, handler: MessageHandler,
                  qos: int = 1) -> None:
        self._subs.append((topic_filter, handler))
        self.broker.deliver_retained(topic_filter, handler)

    def _deliver(self, topic: str, payload: bytes) -> None:
        for topic_filter, handler in list(self._subs):
            if _matches(topic_filter, topic):
                try:
                    handler(topic, payload)
                except Exception:
                    log.exception("handler failed for %s", topic)

    def kill(self) -> None:
        """Drop the client without a clean disconnect, firing its will.

        The point of the loopback broker: this makes "the workstation lost
        power" a testable event rather than a thing you hope you handled.
        """
        self.connected = False
        self.broker.unregister(self, fire_will=True)


class LoopbackBroker:
    def __init__(self):
        self._clients: list[LoopbackTransport] = []
        self._retained: dict[str, bytes] = {}
        self._lock = threading.RLock()

    def register(self, client: LoopbackTransport) -> None:
        with self._lock:
            self._clients.append(client)

    def unregister(self, client: LoopbackTransport, fire_will: bool = False) -> None:
        with self._lock:
            if client in self._clients:
                self._clients.remove(client)
            will = client._will if fire_will else None
        if will:
            self.publish(will[0], will[1], retain=True)

    def publish(self, topic: str, payload: bytes, retain: bool = False) -> None:
        with self._lock:
            if retain:
                self._retained[topic] = payload
            targets = list(self._clients)
        for c in targets:
            c._deliver(topic, payload)

    def deliver_retained(self, topic_filter: str, handler: MessageHandler) -> None:
        with self._lock:
            items = list(self._retained.items())
        for topic, payload in items:
            if _matches(topic_filter, topic):
                try:
                    handler(topic, payload)
                except Exception:
                    log.exception("retained delivery failed for %s", topic)


def build_transport(cfg: dict, client_id: str,
                    will: tuple[str, bytes] | None = None) -> Transport:
    """One factory, so switching bus type is a config edit."""
    kind = cfg.get("transport", "mqtt")
    if kind == "loopback":
        broker = cfg.get("_broker")
        if broker is None:
            raise ValueError("loopback transport needs a shared '_broker'")
        return LoopbackTransport(broker, client_id, will=will)
    return MqttTransport(
        client_id=client_id,
        host=cfg.get("host", "127.0.0.1"),
        port=int(cfg.get("port", 1883)),
        username=cfg.get("username"),
        password=cfg.get("password"),
        keepalive=int(cfg.get("keepalive", 20)),
        tls=bool(cfg.get("tls", False)),
        will=will,
    )
