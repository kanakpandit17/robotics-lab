# Async R1 probe #5 (decisive): is the robot speaking DDS on WiFi at all?
#   - sniff RTPS discovery multicast (any DDS traffic on the interface)
#   - also try the 'go' message set on rt/lowstate
#   powershell -ExecutionPolicy Bypass -File C:\Users\aimat\Desktop\humanoid\r1_probe5.ps1
# Wait ~40s; back on ARENA2036, tell Claude "done5".

$log = "C:\Users\aimat\Desktop\humanoid\r1_probe5_log.txt"
"=== R1 PROBE #5 $(Get-Date) ===" | Out-File $log -Encoding utf8
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
    L "== DDS presence sniff (10s) + 'go' message try (8s) from WSL =="
    $out = wsl.exe -d Ubuntu -- bash -lc "echo '--- sniff ---'; ~/r1bridge/.venv/bin/python /mnt/c/Users/aimat/Desktop/humanoid/dds_sniff.py 192.168.12.22 10; echo '--- msgset go ---'; ~/r1bridge/.venv/bin/python /mnt/c/Users/aimat/Desktop/humanoid/r1_read_state.py --msgset go --secs 8 2>&1 | tail -6" 2>&1
    $out | ForEach-Object { L ("  " + $_) }
  }
} finally {
  L "Restoring ARENA2036 ..."
  for ($i=0;$i -lt 6;$i++){ netsh wlan connect name="ARENA2036" ssid="ARENA2036" | Out-Null; Start-Sleep -Seconds 4; $w=(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias WLAN -ErrorAction SilentlyContinue).IPAddress; if($w -like '141.70.*'){ L "  back on ARENA2036 IP=$w"; break } }
  L "=== DONE. Tell Claude 'done5'. ==="
}
