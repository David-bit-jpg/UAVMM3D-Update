# 只停本会话的对比实验进程（命令行含 cmpfull 的 train/predict 及其数据加载子进程），按 PID 停
$all = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
$main = $all | Where-Object { $_.CommandLine -match 'cmpfull' -and $_.CommandLine -match 'train|predict' }
$ids = @($main | ForEach-Object { $_.ProcessId })
$kids = $all | Where-Object { $_.CommandLine -match 'spawn_main\(parent_pid=(\d+)' -and $ids -contains [int]$Matches[1] }
foreach ($p in @($main) + @($kids)) { if ($p) { Stop-Process -Id $p.ProcessId -Force -ErrorAction SilentlyContinue; "stopped $($p.ProcessId)" } }
