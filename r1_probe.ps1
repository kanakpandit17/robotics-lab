# Self-contained R1 robot probe for the async workflow.
# It: (1) switches WiFi to the robot AP, (2) discovers hosts + open ports (SSH/video),
#     (3) switches WiFi BACK to ARENA2036 so the internet (and Claude) return.
# Run in a normal PowerShell window:
#     powershell -ExecutionPolicy Bypass -File C:\Users\aimat\Desktop\humanoid\r1_probe.ps1
# Then wait ~1 min. When you're back on ARENA2036, tell Claude "done".

$log = "C:\Users\aimat\Desktop\humanoid\r1_probe_log.txt"
"=== R1 PROBE $(Get-Date) ===" | Out-File $log -Encoding utf8
function L($m){ $m | Tee-Object -FilePath $log -Append | Out-Null; Write-Host $m }

L "Switching WiFi to the robot AP R1_03302_c33840b2 ..."
netsh wlan disconnect | Out-Null
Start-Sleep -Seconds 2
$ip = $null
for ($i=1; $i -le 8; $i++) {
  netsh wlan connect name="R1_03302_c33840b2" ssid="R1_03302_c33840b2" | Out-Null
  for ($j=0; $j -lt 3; $j++) {
    Start-Sleep -Seconds 2
    $ip = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias WLAN -ErrorAction SilentlyContinue).IPAddress
    if ($ip -like '192.168.12.*') { break }
  }
  L ("  connect attempt {0} -> IP={1}" -f $i, $ip)
  if ($ip -like '192.168.12.*') { break }
}

if ($ip -notlike '192.168.12.*') {
  L "FAILED to connect/hold the robot AP. Visible networks:"
  (netsh wlan show networks | Select-String "^SSID ") | ForEach-Object { L ("  " + $_.ToString().Trim()) }
} else {
  L ("CONNECTED to robot AP. PC IP=$ip")
  L ("ping 192.168.12.1: " + (New-Object System.Net.NetworkInformation.Ping).Send("192.168.12.1",800).Status)

  L "Discovering live hosts on 192.168.12.0/24 (parallel) ..."
  $tasks = 1..254 | ForEach-Object { (New-Object System.Net.NetworkInformation.Ping).SendPingAsync("192.168.12.$_", 600) }
  [System.Threading.Tasks.Task]::WaitAll($tasks) | Out-Null
  $live = @()
  for ($i=0; $i -lt 254; $i++) { if ($tasks[$i].Result.Status -eq 'Success') { $live += "192.168.12.$($i+1)" } }
  L ("live hosts: " + ($live -join ', '))

  $ports = 22,2222,80,443,554,3389,5000,8000,8080,8443,8554,8888,1935,9090,9000,37000,50051
  foreach ($h in $live) {
    L "--- scanning $h ---"
    foreach ($p in $ports) {
      $c = New-Object System.Net.Sockets.TcpClient
      try {
        $iar = $c.BeginConnect($h, $p, $null, $null)
        if ($iar.AsyncWaitHandle.WaitOne(450) -and $c.Connected) {
          L ("  {0}:{1} OPEN" -f $h, $p)
          if ($p -eq 22 -or $p -eq 2222) {
            try { $s=$c.GetStream(); $s.ReadTimeout=1500; $b=New-Object byte[] 160; $n=$s.Read($b,0,160); L ("     SSH banner: " + [Text.Encoding]::ASCII.GetString($b,0,$n).Trim()) } catch {}
          }
        }
      } catch {}
      $c.Close()
    }
  }
}

L "Restoring ARENA2036 (internet) ..."
for ($i=0; $i -lt 6; $i++) {
  netsh wlan connect name="ARENA2036" ssid="ARENA2036" | Out-Null
  Start-Sleep -Seconds 4
  $w = (Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias WLAN -ErrorAction SilentlyContinue).IPAddress
  if ($w -like '141.70.*') { L "  back on ARENA2036, IP=$w"; break }
}
L "=== DONE. You can tell Claude 'done'. If still offline, reconnect ARENA2036 from the WiFi menu. ==="
