# Overnight chain: wait for the in-flight nav_v3 run to finish training, then resume from its
# newest checkpoint and keep training all night. Detached — survives the Claude session ending.
# NOTE: Isaac Sim frequently HANGS after printing its done-marker, so we watch the LOG for the
# marker (not just process exit) and force-kill a hung process before resuming.
$ErrorActionPreference = "SilentlyContinue"

$targetPid   = 7720
$runLog      = "C:\Users\aimat\Desktop\humanoid\r1_nav_v3_train.log"
$overnightLog= "C:\Users\aimat\Desktop\humanoid\r1_nav_overnight.log"
$ckDir       = "C:\Users\aimat\Desktop\humanoid\IsaacLab\logs\rsl_rl\r1_nav_v3"
$py          = "C:\Users\aimat\miniconda3\envs\env_isaaclab\python.exe"

"CHAIN: armed at $(Get-Date). Waiting for PID $targetPid to finish 800 iters..." | Out-File $overnightLog

# 1) wait for the done-marker in the log OR the process to exit
while ($true) {
    $alive = Get-Process -Id $targetPid -ErrorAction SilentlyContinue
    $done  = Select-String -Path $runLog -Pattern "R1_NAV_TRAIN_DONE" -Quiet -ErrorAction SilentlyContinue
    if ($done -or (-not $alive)) { break }
    Start-Sleep -Seconds 20
}
"CHAIN: first run finished (marker or exit) at $(Get-Date)" | Out-File $overnightLog -Append

# 2) force-kill if Isaac hung on shutdown, so the GPU is free
Stop-Process -Id $targetPid -Force -ErrorAction SilentlyContinue
Start-Sleep -Seconds 15

# 3) newest checkpoint
$ck = Get-ChildItem "$ckDir\model_*.pt" -ErrorAction SilentlyContinue |
      Sort-Object { [int]($_.BaseName -replace 'model_','') } | Select-Object -Last 1
if ($null -eq $ck) { "CHAIN: ERROR no checkpoint in $ckDir" | Out-File $overnightLog -Append; exit 1 }
"CHAIN: resuming from $($ck.FullName) at $(Get-Date)" | Out-File $overnightLog -Append

# 4) long overnight continuation
$env:OMNI_KIT_ACCEPT_EULA = "YES"
Set-Location "C:\Users\aimat\Desktop\humanoid"
& $py "r1_leg_nav_train.py" --resume --checkpoint $ck.FullName --num_envs 2048 --max_iterations 3000 *>> $overnightLog
"CHAIN: overnight run ended at $(Get-Date)" | Out-File $overnightLog -Append
