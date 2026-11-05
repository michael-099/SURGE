# SURGE safety monitor. Checks the machine every 60 s and writes SAFETY_STOP.txt in the repo root when a limit is hit.
# SURGE jobs poll that file and exit cleanly (saving last.pt). This script never kills a process.
# Limits: GPU >= 90 C, hardware slowdown active, any new nvlddmkm / WHEA System-log event, disk < 10 GB free on C: or D:,
# free RAM < 4 GB. Log: runs/safety.log (one line per check that fails, a summary every 15 min).
param([int]$Interval = 60)
$ErrorActionPreference = 'Continue'
$Root = Split-Path -Parent $PSScriptRoot
$Stop = Join-Path $Root 'SAFETY_STOP.txt'
$LogDir = Join-Path $Root 'runs'
New-Item -ItemType Directory -Force $LogDir | Out-Null
$Log = Join-Path $LogDir 'safety.log'
$Start = Get-Date
$maxT = 0
function Write-Log($msg) {
    # Retry: another process (e.g. a `tail -F` reader on Windows) can hold the file open without write sharing.
    $line = "$(Get-Date -Format s) $msg"
    for ($i = 0; $i -lt 10; $i++) {
        try { Add-Content -Path $Log -Value $line -Encoding utf8 -ErrorAction Stop; return } catch { Start-Sleep -Milliseconds 500 }
    }
    try { Add-Content -Path "$Log.fallback" -Value $line -Encoding utf8 -ErrorAction Stop } catch {}
}
function Trip($why) {
    $txt = "SAFETY STOP $(Get-Date -Format s) (PID $PID): $why`r`nSURGE jobs exit cleanly when they see this file. Remove it only after checking the cause."
    Set-Content -Path $Stop -Value $txt -Encoding utf8
    Write-Log "TRIP: $why"
}
Write-Log "monitor started (PID $PID), interval $Interval s"
$lastSummary = Get-Date
while ($true) {
    $problems = @()
    try {
        $q = & nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks_event_reasons.hw_slowdown --format=csv,noheader,nounits 2>$null
        if (-not $q) { $q = & nvidia-smi --query-gpu=temperature.gpu,power.draw,utilization.gpu,clocks_throttle_reasons.hw_slowdown --format=csv,noheader,nounits 2>$null }
        $f = ($q -split ',') | ForEach-Object { $_.Trim() }
        $temp = [int]$f[0]; $pow = $f[1]; $util = $f[2]; $slow = $f[3]
        if ($temp -gt $maxT) { $maxT = $temp }
        if ($temp -ge 90) { $problems += "GPU temperature $temp C" }
        if ($slow -eq 'Active') { $problems += 'GPU hardware slowdown active' }
    } catch { $problems += "nvidia-smi failed: $_" }
    try {
        $ev = Get-WinEvent -FilterHashtable @{LogName = 'System'; ProviderName = 'nvlddmkm', 'Microsoft-Windows-WHEA-Logger'; StartTime = $Start } -ErrorAction SilentlyContinue
        if ($ev) { $problems += "System log: $($ev.Count) new nvlddmkm/WHEA event(s), first ID $($ev[-1].Id) at $($ev[-1].TimeCreated)" }
    } catch {}
    foreach ($d in 'C', 'D') {
        $free = (Get-PSDrive $d).Free / 1GB
        if ($free -lt 10) { $problems += ("disk {0}: {1:N1} GB free" -f $d, $free) }
    }
    $ram = (Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB
    if ($ram -lt 4) { $problems += ("free RAM {0:N1} GB" -f $ram) }
    if ($problems.Count -gt 0) {
        if (-not (Test-Path $Stop)) { Trip ($problems -join '; ') } else { Write-Log ("still: " + ($problems -join '; ')) }
    }
    if (((Get-Date) - $lastSummary).TotalMinutes -ge 15) {
        Write-Log ("check OK: GPU $temp C (max $maxT C since last summary), $pow W, util $util %, RAM free {0:N0} GB" -f $ram)
        $maxT = 0; $lastSummary = Get-Date
    }
    Start-Sleep -Seconds $Interval
}
