$ErrorActionPreference = 'SilentlyContinue'
Write-Host "Stopping uvicorn/python processes..."
Get-Process python -ErrorAction SilentlyContinue | Where-Object { $_.CommandLine -like "*uvicorn*" -or $_.CommandLine -like "*main:app*" } | Stop-Process -Force -ErrorAction SilentlyContinue
Write-Host "Checking port 8000..."
Get-NetTCPConnection -LocalPort 8000 -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Found process on 8000, PID:" $_.OwningProcess
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
}
Get-NetTCPConnection -LocalPort 3000 -ErrorAction SilentlyContinue | ForEach-Object {
    Write-Host "Found process on 3000, PID:" $_.OwningProcess
    Stop-Process -Id $_.OwningProcess -Force -ErrorAction SilentlyContinue
}
Write-Host "Done."
