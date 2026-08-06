# Async R1 probe #3: safely identify the video service at 192.168.12.26:8888
# without hanging (reads only the first ~1KB of each response, then closes).
#   powershell -ExecutionPolicy Bypass -File C:\Users\aimat\Desktop\humanoid\r1_probe3.ps1
# Wait ~1 min; back on ARENA2036, tell Claude "done3".

$log = "C:\Users\aimat\Desktop\humanoid\r1_probe3_log.txt"
"=== R1 PROBE #3 $(Get-Date) ===" | Out-File $log -Encoding utf8
function L($m){ $m | Tee-Object -FilePath $log -Append | Out-Null; Write-Host $m }

function SafeGet($h,$port,$path){
  $c = New-Object System.Net.Sockets.TcpClient
  try {
    $iar = $c.BeginConnect($h,$port,$null,$null)
    if (-not ($iar.AsyncWaitHandle.WaitOne(2500) -and $c.Connected)) { return "  GET $path -> NO CONNECT" }
    $c.EndConnect($iar)
    $s = $c.GetStream(); $s.ReadTimeout = 2500
    $req = "GET $path HTTP/1.1`r`nHost: ${h}:${port}`r`nUser-Agent: probe`r`nAccept: */*`r`nConnection: close`r`n`r`n"
    $rb = [Text.Encoding]::ASCII.GetBytes($req); $s.Write($rb,0,$rb.Length); $s.Flush()
    $buf = New-Object byte[] 1024; $sb = New-Object System.Text.StringBuilder; $total = 0
    try { while ($total -lt 900) { $n = $s.Read($buf,0,$buf.Length); if ($n -le 0){break}; [void]$sb.Append([Text.Encoding]::ASCII.GetString($buf,0,$n)); $total += $n } } catch {}
    $head = (($sb.ToString()) -split "`r`n`r`n")[0]
    return "  GET $path -> " + (($head -replace "`r","") -replace "`n"," | ")
  } catch { return "  GET $path -> ERR $($_.Exception.Message)" }
  finally { $c.Close() }
}

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
    L "CONNECTED PC IP=$ip. Identifying 192.168.12.26:8888 ..."
    foreach ($path in @('/','/?action=snapshot','/?action=stream','/video_feed','/stream','/snapshot','/video','/mjpg/video.mjpg','/cam.mjpg','/image','/index.html','/api')) {
      L (SafeGet '192.168.12.26' 8888 $path)
    }
    L "Also checking 192.168.12.26 for SSH/other ports ..."
    foreach ($p in 22,80,443,554,1935,8080,8554,9000) {
      $c=New-Object System.Net.Sockets.TcpClient
      try{ $iar=$c.BeginConnect('192.168.12.26',$p,$null,$null); if($iar.AsyncWaitHandle.WaitOne(450) -and $c.Connected){ L "  192.168.12.26:$p OPEN" } }catch{}
      $c.Close()
    }
  }
} finally {
  L "Restoring ARENA2036 ..."
  for ($i=0;$i -lt 6;$i++){ netsh wlan connect name="ARENA2036" ssid="ARENA2036" | Out-Null; Start-Sleep -Seconds 4; $w=(Get-NetIPAddress -AddressFamily IPv4 -InterfaceAlias WLAN -ErrorAction SilentlyContinue).IPAddress; if($w -like '141.70.*'){ L "  back on ARENA2036 IP=$w"; break } }
  L "=== DONE. Tell Claude 'done3'. ==="
}
