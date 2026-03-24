$p = Get-CimInstance Win32_Process -Filter "ProcessId=7428"
if ($p) {
    Write-Host "Process:" $p.ProcessName
    Write-Host "Command:" $p.CommandLine
} else {
    Write-Host "Process 7428 not found"
}
