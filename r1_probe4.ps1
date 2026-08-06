# Async R1 probe #4: test the DDS (SDK) channel over WiFi, and check the classic Unitree
# onboard subnet 192.168.123.x. Switches to robot AP, tests, switches back to ARENA2036.
#   powershell -ExecutionPolicy Bypass -File C:\Users\aimat\Desktop\humanoid\r1_probe4.ps1
# Wait ~1.5 min; back on ARENA2036, tell Claude "done4".

$log = "C:\Users\aimat\Desktop\humanoid\r1_probe4_log.txt"
"=== R1 PROBE #4 $(Get-Date) ===" | Out-File $log -Encoding utf8
function L($m){ $m | Tee-Object -FilePath $log -Append | Out-Null; Write-Host $m }

L "Switching WiFi to robot AP ..."
netsh wlan disconnect | Out-Null; Start-Sleep -Seconds 2
$ip = $null
for ($i=1; $i -le 8; $i++) {
  netsh wlan connect name="R1_03302_c33840b2" ssid="R1_03302_c33840b2" | Out-Null
  for ($j=0;$j -lt 3;$j++){ Start-Sleep -Seconds 2; $ip=(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias WLAN -ErrorAction SilentlyContinue).IPAddress; if($ip -like '192.168.12.*'){break} }
  L ("  connect attempt {0} -> IP={1}" -f $i,$ip); if($ip -like '192.168.12.*'){break}
}

try {
  if ($ip -notlike '192.168.12.*') { L "FAILED to connect robot AP." }
  else {
    L "CONNECTED PC IP=$ip"
    L "== check classic Unitree onboard subnet 192.168.123.x (via gateway) =="
    foreach ($h in '192.168.123.161','192.168.123.1','192.168.123.15','192.168.123.18','192.168.123.10') {
      $st = 'Unreachable'
      try { $st = (New-Object System.Net.NetworkInformation.Ping).Send($h, 500).Status } catch { $st = 'NoRoute' }
      if ($st -eq 'Success') {
        L "  $h ALIVE"
        foreach ($p in 22,80,554,8080,8888) { $c=New-Object System.Net.Sockets.TcpClient; try{ $iar=$c.BeginConnect($h,$p,$null,$null); if($iar.AsyncWaitHandle.WaitOne(400) -and $c.Connected){ L "    ${h}:${p} OPEN" } }catch{}; $c.Close() }
      } else { L "  $h unreachable ($st)" }
    }
    L "== DDS test from WSL =="
    $out = wsl.exe -d Ubuntu -- bash /mnt/c/Users/aimat/Desktop/humanoid/r1_dds_test.sh 2>&1
    $out | ForEach-Object { L ("  " + $_) }
  }
} finally {
  L "Restoring ARENA2036 ..."
  for ($i=0;$i -lt 6;$i++){ netsh wlan connect name="ARENA2036" ssid="ARENA2036" | Out-Null; Start-Sleep -Seconds 4; $w=(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias WLAN -ErrorAction SilentlyContinue).IPAddress; if($w -like '141.70.*'){ L "  back on ARENA2036 IP=$w"; break } }
  L "=== DONE. Tell Claude 'done4'. ==="
}
