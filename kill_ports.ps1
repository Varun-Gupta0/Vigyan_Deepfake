$ErrorActionPreference = 'SilentlyContinue'
Get-NetTCPConnection -LocalPort 8000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
Get-NetTCPConnection -LocalPort 3000 | ForEach-Object { Stop-Process -Id $_.OwningProcess -Force }
Write-Host "Killed processes on ports 8000 and 3000"
