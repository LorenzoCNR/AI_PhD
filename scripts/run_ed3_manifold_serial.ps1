$ErrorActionPreference = 'Stop'

$workspace = 'C:\Users\loren\Desktop\USB_Content\New project\AI_PhD'
$python = 'C:\Users\loren\miniconda3\envs\rat_manifold_v1\python.exe'
$runner = Join-Path $workspace 'cebra_replica\rat_paper_ed3_manifold_trace.py'
$root = Join-Path $workspace 'results\rat_paper_replication_2023_v1\ED3\manifold'
$logRoot = Join-Path $root 'run_logs'
$numbaCache = Join-Path $env:TEMP 'AI_PhD_rat_ed3_numba_cache'
$tag = Get-Date -Format 'yyyyMMdd_HHmmss'
$statePath = Join-Path $root 'SERIAL_QUEUE_STATE.json'
$queueLogPath = Join-Path $logRoot "serial_queue_$tag.log"
$jobs = @(
    @{rat='achilles'; method='umap'}, @{rat='achilles'; method='tsne'},
    @{rat='buddy'; method='umap'},    @{rat='buddy'; method='tsne'},
    @{rat='cicero'; method='umap'},   @{rat='cicero'; method='tsne'},
    @{rat='gatsby'; method='umap'},   @{rat='gatsby'; method='tsne'}
)

function Write-QueueState([string]$status, [string]$detail, [int]$completed) {
    $state = [ordered]@{
        campaign_id = 'rat_ed3_manifold_seed42_window10_flat_serial_v1'
        status = $status
        completed_jobs = $completed
        planned_jobs = $jobs.Count
        detail = $detail
        controller_pid = $PID
        updated_local = (Get-Date -Format o)
    }
    [System.IO.File]::WriteAllText($statePath, ($state | ConvertTo-Json -Depth 5), [System.Text.UTF8Encoding]::new($false))
    Add-Content -LiteralPath $queueLogPath -Value "$(Get-Date -Format o) [$status] completed=$completed/$($jobs.Count) $detail"
}

function Repair-StaleRunningMarker([string]$destination) {
    $markerPath = Join-Path $destination 'RUNNING.json'
    if (-not (Test-Path -LiteralPath $markerPath)) { return }
    $marker = Get-Content -LiteralPath $markerPath -Raw | ConvertFrom-Json
    if ($marker.status -ne 'running') { return }
    $pidValue = 0
    try { $pidValue = [int]$marker.pid } catch { return }
    if ($pidValue -le 0) { return }
    $process = Get-Process -Id $pidValue -ErrorAction SilentlyContinue
    $isSameFitProcess = $false
    if ($process -and $process.Path -ieq $python -and $marker.started_utc) {
        try {
            $markerStart = [DateTimeOffset]::Parse([string]$marker.started_utc).LocalDateTime
            $ageDifference = [math]::Abs(($process.StartTime - $markerStart).TotalSeconds)
            $isSameFitProcess = $ageDifference -lt 30
        } catch { $isSameFitProcess = $false }
    } elseif ($process -and $process.Path -ieq $python -and -not $marker.started_utc) {
        $isSameFitProcess = $true
    }
    if ($isSameFitProcess) {
        throw "Refusing to resume $destination; its recorded Python process PID $pidValue is still alive."
    }
    $marker.status = 'stale_interrupted'
    $marker.pid = -1
    [System.IO.File]::WriteAllText($markerPath, ($marker | ConvertTo-Json -Depth 10), [System.Text.UTF8Encoding]::new($false))
    Add-Content -LiteralPath $queueLogPath -Value "$(Get-Date -Format o) [recovery] marked stale RUNNING PID $pidValue in $destination; no final embedding/manifest was available."
}

New-Item -ItemType Directory -Path $logRoot -Force | Out-Null
New-Item -ItemType Directory -Path $numbaCache -Force | Out-Null
$env:NUMBA_CACHE_DIR = $numbaCache
$env:NUMBA_NUM_THREADS = '1'
$env:OMP_NUM_THREADS = '1'
$env:MKL_NUM_THREADS = '1'
$env:OPENBLAS_NUM_THREADS = '1'
if (Test-Path -LiteralPath $queueLogPath) { throw "Refusing to overwrite $queueLogPath" }
Add-Content -LiteralPath $queueLogPath -Value "$(Get-Date -Format o) Started serial ED3 manifold queue."

try {
    $done = 0
    foreach ($job in $jobs) {
        $dest = Join-Path $root "$($job.method)\$($job.rat)\window10_flat\seed_42"
        Repair-StaleRunningMarker $dest
        $stdout = Join-Path $logRoot "serial_${tag}_$($job.rat)_$($job.method).stdout.log"
        $stderr = Join-Path $logRoot "serial_${tag}_$($job.rat)_$($job.method).stderr.log"
        if ((Test-Path -LiteralPath $stdout) -or (Test-Path -LiteralPath $stderr)) {
            throw "Refusing to overwrite logs for $($job.rat)/$($job.method)"
        }
        Write-QueueState 'running' "$($job.rat)/$($job.method)" $done
        $arguments = @('-B', "`"$runner`"", '--rat', $job.rat, '--method', $job.method,
                       '--seed', '42', '--input-representation', 'window10_flat')
        $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $workspace `
            -WindowStyle Hidden -RedirectStandardOutput $stdout -RedirectStandardError $stderr -PassThru -Wait
        if ($process.ExitCode -ne 0) {
            throw "$($job.rat)/$($job.method) exited with code $($process.ExitCode); inspect $stderr"
        }
        $manifest = Join-Path $dest 'manifest.json'
        $embedding = Join-Path $dest 'embedding.npz'
        if (-not (Test-Path -LiteralPath $manifest) -or -not (Test-Path -LiteralPath $embedding)) {
            throw "$($job.rat)/$($job.method) exited zero without a final manifest and embedding."
        }
        $record = Get-Content -LiteralPath $manifest -Raw | ConvertFrom-Json
        if ($record.status -ne 'complete') {
            throw "$($job.rat)/$($job.method) manifest is not complete."
        }
        $done++
        Write-QueueState 'job_complete' "$($job.rat)/$($job.method) verified from manifest." $done
    }
    Write-QueueState 'complete' 'All eight rat/method embeddings have verified manifests.' $done
}
catch {
    $done = if (Test-Path -LiteralPath $statePath) {
        [int](Get-Content -LiteralPath $statePath -Raw | ConvertFrom-Json).completed_jobs
    } else { 0 }
    Write-QueueState 'stopped_on_error' $_.Exception.Message $done
    exit 1
}
