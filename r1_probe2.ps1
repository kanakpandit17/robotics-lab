# Async R1 probe #2: identify 192.168.12.26:8888, deep-scan robot hosts, and test DDS from WSL.
# Switches to the robot AP, gathers everything, switches back to ARENA2036.
#   powershell -ExecutionPolicy Bypass -File C:\Users\aimat\Desktop\humanoid\r1_probe2.ps1
# Then wait ~1.5 min; when back on ARENA2036, tell Claude "done2".

$log = "C:\Users\aimat\Desktop\humanoid\r1_probe2_log.txt"
"=== R1 PROBE #2 $(Get-Date) ===" | Out-File $log -Encoding utf8
function L($m){ $m | Tee-Object -FilePath $log -Append | Out-Null; Write-Host $m }

L "Switching WiFi to robot AP ..."
netsh wlan disconnect | Out-Null; Start-Sleep -Seconds 2
$ip = $null
for ($i=1; $i -le 8; $i++) {
  netsh wlan connect name="R1_03302_c33840b2" ssid="R1_03302_c33840b2" | Out-Null
  for ($j=0; $j -lt 3; $j++) { Start-Sleep -Seconds 2; $ip=(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias WLAN -ErrorAction SilentlyContinue).IPAddress; if ($ip -like '192.168.12.*'){break} }
  L ("  connect attempt {0} -> IP={1}" -f $i,$ip); if ($ip -like '192.168.12.*'){break}
}

if ($ip -notlike '192.168.12.*') {
  L "FAILED to connect robot AP."
} else {
  L "CONNECTED PC IP=$ip"
  # live hosts
  $tasks = 1..254 | ForEach-Object { (New-Object System.Net.NetworkInformation.Ping).SendPingAsync("192.168.12.$_",600) }
  [System.Threading.Tasks.Task]::WaitAll($tasks) | Out-Null
  $live=@(); for ($i=0;$i -lt 254;$i++){ if($tasks[$i].Result.Status -eq 'Success'){ $live += "192.168.12.$($i+1)" } }
  L ("live hosts: " + ($live -join ', '))

  # deep port scan of robot hosts (skip our own .22)
  $targets = @('192.168.12.1','192.168.12.26') + ($live | Where-Object { $_ -ne '192.168.12.22' })
  $targets = $targets | Select-Object -Unique
  $ports = 21,22,23,80,443,554,1883,1935,3000,5000,5555,5901,6080,7000,7070,8000,8001,8080,8081,8443,8554,8888,8889,9000,9090,9999,11311,18080,37000,50051
  foreach ($h in $targets) {
    L "--- scan $h ---"
    foreach ($p in $ports) {
      $c=New-Object System.Net.Sockets.TcpClient
      try { $iar=$c.BeginConnect($h,$p,$null,$null); if($iar.AsyncWaitHandle.WaitOne(450) -and $c.Connected){ L ("  {0}:{1} OPEN" -f $h,$p) } } catch {}
      $c.Close()
    }
  }

  # identify HTTP services (try 8888 on .26 and .1, plus any http-ish ports)
  foreach ($u in @("http://192.168.12.26:8888/","http://192.168.12.1:8888/","http://192.168.12.26:8080/","http://192.168.12.1:80/")) {
    try {
      $r = Invoke-WebRequest -Uri $u -TimeoutSec 5 -UseBasicParsing -ErrorAction Stop
      L ("HTTP $u -> $($r.StatusCode) ct=$($r.Headers['Content-Type']) server=$($r.Headers['Server'])")
      $body = ($r.Content | Out-String); if ($body.Length -gt 300){ $body = $body.Substring(0,300) }
      L ("   body: " + ($body -replace '\s+',' '))
    } catch { L ("HTTP $u -> $($_.Exception.Message)") }
  }

  # DDS test from WSL (does mirrored WSL see the robot subnet? does R1 publish LowState?)
  L "--- WSL networking + DDS test ---"
  $wsl = wsl.exe -d Ubuntu -- bash -lc "echo IFACES:; ip -4 addr show | grep 'inet 192.168.12' || echo '  (WSL does NOT see 192.168.12.x)'; echo PING:; ping -c2 -W2 192.168.12.1 2>&1 | tail -2; echo DDS:; ~/r1bridge/.venv/bin/python /mnt/c/Users/aimat/Desktop/humanoid/r1_read_state.py --secs 8 2>&1 | tail -8"
  $wsl | ForEach-Object { L ("  " + $_) }
}

L "Restoring ARENA2036 ..."
for ($i=0;$i -lt 6;$i++){ netsh wlan connect name="ARENA2036" ssid="ARENA2036" | Out-Null; Start-Sleep -Seconds 4; $w=(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias WLAN -ErrorAction SilentlyContinue).IPAddress; if($w -like '141.70.*'){ L "  back on ARENA2036 IP=$w"; break } }
L "=== DONE. Tell Claude 'done2'. ==="
