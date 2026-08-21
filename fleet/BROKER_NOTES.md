# Getting MAiRA's telemetry onto this laptop — two things that were wrong

Written while wiring MAiRA into `arena_live_scene.py`. Both problems below are on the
*infrastructure* side, not in the fleet code, which is fine as written.

## 1. The laptop's IP changed, so the MAiRA agent is dialling nobody

`config/maira-agent.json` on the workstation says:

```json
"broker": { "host": "141.70.221.54", "port": 1883 }
```

but this laptop's ARENA2036 wifi address is now **141.70.221.199**, and `.54` answers
nothing at all:

```
> ping 141.70.221.54
Reply from 141.70.221.199: Destination host unreachable.
```

So the agent's outbound connection never lands, and `fleet/v1/maira-01/state` is never
published. Nothing on the laptop can fix this — the change has to happen on the
workstation.

**Fix (on the workstation):** set `broker.host` to `141.70.221.199` and restart the agent.

**Better fix:** this will break again on the next DHCP lease. Either reserve the laptop's
address on the ARENA2036 DHCP server, or run the broker on the workstation instead and
point *this* machine outward — the broker only has to be somewhere both machines can
reach, and the workstation is the one with a stable role.

Verify from the laptop once the agent restarts:

```
"C:\Program Files\mosquitto\mosquitto_sub.exe" -h 141.70.221.199 -t 'fleet/v1/#' -v
```

`maira-01/presence` and `maira-01/state` should appear immediately — both are retained,
so they arrive on subscribe without waiting for the next publish.

## 2. Two brokers are running on port 1883, and they are not the same bus

Installing mosquitto with winget also registers an **auto-starting Windows service**, which
binds loopback only:

```
TCP    0.0.0.0:1883      LISTENING   36268   <- fleet broker (config/mosquitto.conf)
TCP    127.0.0.1:1883    LISTENING   11588   <- winget's service, stock config
TCP    [::1]:1883        LISTENING   11588
```

Because `127.0.0.1` is the more specific bind, it wins for loopback connections. Confirmed
by publishing a retained message to one address and subscribing on the other — it does not
cross over. So:

- anything connecting to `127.0.0.1:1883` (including `config/laptop.json`, which is what
  the **dashboard** uses) reaches the stock service: an empty bus with no robots on it;
- anything connecting from the LAN reaches the real fleet broker.

That is why the dashboard can look completely dead while the bus is fine.

**Fix — stop and disable the service.** Needs an *Administrator* PowerShell; a normal shell
gets `Cannot open mosquitto service on computer '.'`:

```powershell
Stop-Service mosquitto
Set-Service mosquitto -StartupType Disabled
```

Then start the real one from the repo, which binds `0.0.0.0` and allows anonymous:

```
"C:\Program Files\mosquitto\mosquitto.exe" -c config\mosquitto.conf -v
```

Until the service is disabled, pass the LAN address explicitly rather than `127.0.0.1`:

```
python arena_live_scene.py --mqtt-host 141.70.221.199
```

## Already checked, not a problem

- **Firewall** — winget added inbound Allow rules for `mosquitto.exe` (TCP + UDP, any port)
  on the Public profile, and both Wi-Fi and Ethernet are Public. The workstation can reach
  port 1883 once it targets the right address.
- **Anonymous access** — `config/mosquitto.conf` sets `allow_anonymous true`, and an
  anonymous publish to the LAN broker succeeds.
