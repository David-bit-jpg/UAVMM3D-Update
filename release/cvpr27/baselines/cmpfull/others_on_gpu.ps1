# 输出显卡上"别人的" Python 进程数（不含本会话 cmpfull 的进程及其子进程）
$gp = @(nvidia-smi --query-compute-apps=pid --format=csv,noheader | ForEach-Object { [int]$_.Trim() })
$all = Get-CimInstance Win32_Process -Filter "Name='python.exe'"
$ours = @($all | Where-Object { $_.CommandLine -match 'cmpfull' } | ForEach-Object { $_.ProcessId })
$n = 0
foreach ($p in $all) { if ($gp -contains $p.ProcessId) {
  $par = if ($p.CommandLine -match 'spawn_main\(parent_pid=(\d+)') { [int]$Matches[1] } else { -1 }
  if (-not ($ours -contains $p.ProcessId) -and -not ($ours -contains $par)) { $n++ } } }
$n
