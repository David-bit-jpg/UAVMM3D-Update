param([string]$Pat)
# 数命令行里含 $Pat 的 bash.exe 进程（只数 bash，不会把查询自身算进去）
@(Get-CimInstance Win32_Process -Filter "Name='bash.exe'" | Where-Object { $_.CommandLine -match [regex]::Escape($Pat) }).Count
