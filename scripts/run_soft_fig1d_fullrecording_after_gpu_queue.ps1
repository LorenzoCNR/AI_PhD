$ErrorActionPreference = 'Stop'

$workspace = 'C:\Users\loren\Desktop\USB_Content\New project\AI_PhD'
$campaign = Join-Path $workspace 'results\rat_paper_fig1e_ed4_6seed_v1'
$output = Join-Path $workspace 'results\user_soft_rat_fullrecording_t1_3conditions_seed42_v1'
$python = 'C:\Users\loren\miniconda3\envs\cebra\python.exe'
$queueStatePath = Join-Path $campaign 'GPU_QUEUE_STATE.json'
$pivaeLockPath = Join-Path $campaign 'pivae\RUNNING.lock'
$statePath = Join-Path $output 'SOFT_QUEUE_STATE.json'
$powershellPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'
$mutex = New-Object System.Threading.Mutex($false, 'Local\AI_PhD_RatPaper6SeedGpuQueue')
$ownsMutex = $false
$watcherTag = Get-Date -Format 'yyyyMMdd_HHmmss_fff'

function Write-SoftQueueState([string]$status, [string]$detail,
                              [string]$queueStatus = '', [int]$queueControllerPid = 0,
                              [int]$queueWorkerPid = 0, [string]$lastProgressLocal = '') {
    $state = [ordered]@{
        campaign_id = 'user_soft_rat_fullrecording_t1_3conditions_seed42_v1'
        status = $status
        detail = $detail
        watcher_pid = $PID
        queue_status = $queueStatus
        queue_controller_pid = $queueControllerPid
        queue_worker_pid = $queueWorkerPid
        last_progress_local = $lastProgressLocal
        updated_local = (Get-Date -Format o)
    }
    $json = $state | ConvertTo-Json -Depth 5
    $parent = Split-Path -Parent $statePath
    New-Item -ItemType Directory -Path $parent -Force | Out-Null
    $stateTemporaryPath = "$statePath.$PID.tmp"
    [System.IO.File]::WriteAllText($stateTemporaryPath, $json, [System.Text.UTF8Encoding]::new($false))
    for ($stateWriteAttempt = 0; $stateWriteAttempt -lt 40; $stateWriteAttempt++) {
        try {
            if (Test-Path -LiteralPath $statePath) {
                [System.IO.File]::Replace($stateTemporaryPath, $statePath, [NullString]::Value)
            } else {
                [System.IO.File]::Move($stateTemporaryPath, $statePath)
            }
            break
        } catch {
            $stateWriteError = $_.Exception.GetBaseException()
            if ($stateWriteAttempt -ge 39 -or
                ($stateWriteError -isnot [System.IO.IOException] -and
                 $stateWriteError -isnot [System.UnauthorizedAccessException])) { throw }
            Start-Sleep -Milliseconds 250
        }
    }
}

function Get-VerifiedProcess([int]$processId, [string]$expectedPath, [string]$expectedStartedLocal = '') {
    if ($processId -le 0) { return $null }
    $process = Get-Process -Id $processId -ErrorAction SilentlyContinue
    if (-not $process) { return $null }
    if ($expectedPath -and $process.Path -ine $expectedPath) { return $null }
    if ($expectedStartedLocal) {
        $expectedStart = [datetime]::Parse($expectedStartedLocal).ToLocalTime()
        if ([math]::Abs(($process.StartTime - $expectedStart).TotalSeconds) -gt 120) { return $null }
    }
    return $process
}

function Get-GpuWorkFingerprint([string]$stage) {
    $root = switch -Wildcard ($stage) {
        'conv_pivae*' { Join-Path $campaign 'pivae'; break }
        'cebra*' { Join-Path $campaign 'cebra'; break }
        'fig2bc*' { Join-Path $workspace 'results\rat_paper_fig2bc_32d_v1'; break }
        'soft*' { Join-Path $campaign 'soft'; break }
        default { $null }
    }
    if (-not $root -or -not (Test-Path -LiteralPath $root)) { return '' }
    $latest = Get-ChildItem -LiteralPath $root -Recurse -File -ErrorAction SilentlyContinue |
        Where-Object { $_.Name -match '^(training_history\.csv|loss_history\.csv|campaign\.log|CAMPAIGN_STATE\.json|run_state\.json)$' } |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1
    if (-not $latest) { return '' }
    return "$($latest.FullName)|$($latest.Length)|$($latest.LastWriteTimeUtc.Ticks)"
}

try {
    $watchStarted = Get-Date
    $lastProgressAt = $watchStarted
    $lastFingerprint = ''
    $nextHeartbeat = $watchStarted
    $nextInactivityDiagnostic = $watchStarted.AddMinutes(20)
    Write-SoftQueueState 'waiting_for_existing_gpu_campaign' 'Waiting for the GPU queue to report successful completion and release its live controller/worker processes.'
    while ($true) {
        $queueState = if (Test-Path -LiteralPath $queueStatePath) {
            Get-Content -LiteralPath $queueStatePath -Raw | ConvertFrom-Json
        } else { $null }
        $stage = if ($queueState) { [string]$queueState.stage } else { '' }
        $queueStatus = if ($queueState) { [string]$queueState.status } else { 'missing' }
        $fingerprint = "$stage|$(Get-GpuWorkFingerprint $stage)"
        if ($fingerprint -ne $lastFingerprint) {
            $lastFingerprint = $fingerprint
            $lastProgressAt = Get-Date
            $nextInactivityDiagnostic = $lastProgressAt.AddMinutes(20)
        }

        $controllerStarted = if ($queueState -and $queueState.controller_started_local) {
            [string]$queueState.controller_started_local
        } else { '' }
        $controller = if ($queueState -and $queueState.controller_pid) {
            Get-VerifiedProcess ([int]$queueState.controller_pid) $powershellPath $controllerStarted
        } else { $null }
        if ($controller -and -not $controllerStarted -and $queueState.updated_local) {
            $stateUpdated = [datetime]::Parse([string]$queueState.updated_local).ToLocalTime()
            if ($controller.StartTime -gt $stateUpdated.AddSeconds(5)) { $controller = $null }
        }

        $worker = if ($queueState -and [int]$queueState.worker_pid -gt 0) {
            Get-VerifiedProcess ([int]$queueState.worker_pid) ([string]$queueState.worker_path) ([string]$queueState.worker_started_local)
        } else { $null }

        $pivae = $null
        $pivaeLockUnverifiable = $false
        if (Test-Path -LiteralPath $pivaeLockPath) {
            try {
                $pivaeLock = Get-Content -LiteralPath $pivaeLockPath -Raw | ConvertFrom-Json
                if ([int]$pivaeLock.pid -le 0 -or -not $pivaeLock.started_utc) {
                    throw 'PiVAE lock has no verifiable PID/start time.'
                }
                $pivaeStart = if ($pivaeLock.started_utc) { ([datetime]::Parse($pivaeLock.started_utc).ToLocalTime()).ToString('o') } else { '' }
                $pivae = Get-VerifiedProcess ([int]$pivaeLock.pid) $python.Replace('envs\cebra', 'envs\pivae_tf') $pivaeStart
            } catch {
                $pivae = $null
                $pivaeLockUnverifiable = $true
            }
        }
        $untrackedPivaeAlive = @(Get-Process -Name python -ErrorAction SilentlyContinue |
            Where-Object { $_.Path -ieq $python.Replace('envs\cebra', 'envs\pivae_tf') }).Count -gt 0

        $queueSucceeded = $queueStatus -eq 'complete'
        if ($queueSucceeded -and -not $controller -and -not $worker -and -not $pivae -and -not $pivaeLockUnverifiable -and -not $untrackedPivaeAlive) {
            try {
                $ownsMutex = $mutex.WaitOne(0)
            } catch [System.Threading.AbandonedMutexException] {
                $ownsMutex = $true
            }
            if ($ownsMutex) {
                $confirmedState = if (Test-Path -LiteralPath $queueStatePath) {
                    Get-Content -LiteralPath $queueStatePath -Raw | ConvertFrom-Json
                } else { $null }
                if ($confirmedState -and $confirmedState.status -eq 'complete') { break }
                $mutex.ReleaseMutex()
                $ownsMutex = $false
            }
        }

        $inactiveMinutes = [math]::Round(((Get-Date) - $lastProgressAt).TotalMinutes, 1)
        $queueControllerPid = if ($queueState) { [int]$queueState.controller_pid } else { 0 }
        $queueWorkerPid = if ($queueState) { [int]$queueState.worker_pid } else { 0 }
        $upstreamFailed = $queueStatus -match '(?i)failed|error|stopped|blocked'
        if ($inactiveMinutes -ge 20 -and (Get-Date) -ge $nextInactivityDiagnostic) {
            $message = if ($pivaeLockUnverifiable) {
                'Cannot verify the piVAE lock owner; keeping Soft queued until the lock can be inspected.'
            } elseif ($upstreamFailed) {
                "GPU queue reports '$queueStatus' at stage '$stage'; waiting for an explicit successful recovery. No fit will be started on a failed/incomplete queue. No monitored progress for $inactiveMinutes minutes."
            } else {
                "Still waiting for successful queue completion; stage='$stage', queue_status='$queueStatus', controller_pid=$queueControllerPid, worker_pid=$queueWorkerPid, live_controller=$([bool]$controller), live_worker=$([bool]$worker), live_pivae=$([bool]$pivae). No monitored progress for $inactiveMinutes minutes. No process will be stopped."
            }
            Write-SoftQueueState 'inactivity_diagnostic' $message $queueStatus $queueControllerPid $queueWorkerPid $lastProgressAt.ToString('o')
            $nextInactivityDiagnostic = (Get-Date).AddMinutes(5)
            $nextHeartbeat = (Get-Date).AddMinutes(5)
        }
        elseif ((Get-Date) -ge $nextHeartbeat) {
            $waitingStatus = if ($pivaeLockUnverifiable) { 'waiting_for_lock_verification' } elseif ($upstreamFailed) { 'waiting_for_queue_recovery' } else { 'waiting_for_existing_gpu_campaign' }
            $message = "Queue stage='$stage', status='$queueStatus'; live_controller=$([bool]$controller), live_worker=$([bool]$worker), live_pivae=$([bool]$pivae), unverifiable_pivae_lock=$pivaeLockUnverifiable; last monitored progress $inactiveMinutes minutes ago."
            Write-SoftQueueState $waitingStatus $message $queueStatus $queueControllerPid $queueWorkerPid $lastProgressAt.ToString('o')
            $nextHeartbeat = (Get-Date).AddMinutes(5)
        }
        Start-Sleep -Seconds 60
    }

    $stdoutPath = Join-Path $output "soft_fullrecording_$watcherTag.stdout.log"
    $stderrPath = Join-Path $output "soft_fullrecording_$watcherTag.stderr.log"
    if ((Test-Path -LiteralPath $stdoutPath) -or (Test-Path -LiteralPath $stderrPath)) {
        throw "Refusing to overwrite existing Soft execution logs under $output"
    }
    Write-SoftQueueState 'starting' 'GPU queue is released; launching 24 full-recording Stable Soft fits (4 rats x 3 conditions x CNN/Transformer), seed 42, native 3D, batch 512, 10000 updates, embedding T=1, metadata T=0.5.'
    $arguments = @(
        '-B', '-m', 'user_soft.full_recording_runner',
        '--execute',
        '--output', 'results/user_soft_rat_fullrecording_t1_3conditions_seed42_v1',
        '--conditions', 'time_direction_position', 'time_only', 'direction_position',
        '--embedding-temperature', '1.0',
        '--max-new-fits', '24'
    )
    $process = Start-Process -FilePath $python -ArgumentList $arguments -WorkingDirectory $workspace `
        -WindowStyle Hidden -RedirectStandardOutput $stdoutPath -RedirectStandardError $stderrPath -PassThru
    [void]$process.Handle
    $lastProgressAt = Get-Date
    $nextHeartbeat = $lastProgressAt.AddMinutes(5)
    $nextInactivityDiagnostic = $lastProgressAt.AddMinutes(20)
    $lastFingerprint = ''
    Write-SoftQueueState 'running' "Full-recording Soft worker PID $($process.Id) launched; logs: $stdoutPath ; $stderrPath."
    while (-not $process.HasExited) {
        $latest = Get-ChildItem -LiteralPath $output -Recurse -File -ErrorAction SilentlyContinue |
            Where-Object { $_.Name -match '^(training_history\.csv|loss_history\.csv|CAMPAIGN_STATE\.json|run_state\.json)$' } |
            Sort-Object LastWriteTime -Descending | Select-Object -First 1
        $fingerprint = if ($latest) { "$($latest.FullName)|$($latest.Length)|$($latest.LastWriteTimeUtc.Ticks)" } else { '' }
        foreach ($logPath in @($stdoutPath, $stderrPath)) {
            if (Test-Path -LiteralPath $logPath) {
                $logFile = Get-Item -LiteralPath $logPath
                $fingerprint += "|$($logFile.Length)|$($logFile.LastWriteTimeUtc.Ticks)"
            }
        }
        if ($fingerprint -ne $lastFingerprint) {
            $lastFingerprint = $fingerprint
            $lastProgressAt = Get-Date
            $nextInactivityDiagnostic = $lastProgressAt.AddMinutes(20)
        }
        $inactiveMinutes = [math]::Round(((Get-Date) - $lastProgressAt).TotalMinutes, 1)
        if ($inactiveMinutes -ge 20 -and (Get-Date) -ge $nextInactivityDiagnostic) {
            Write-SoftQueueState 'inactivity_diagnostic' "No monitored output changed for $inactiveMinutes minutes; worker PID $($process.Id) remains alive (CPU seconds=$([math]::Round($process.CPU,1))). Continuing without interrupting the fit." '' 0 $process.Id $lastProgressAt.ToString('o')
            $nextInactivityDiagnostic = (Get-Date).AddMinutes(5)
            $nextHeartbeat = (Get-Date).AddMinutes(5)
        } elseif ((Get-Date) -ge $nextHeartbeat) {
            Write-SoftQueueState 'running' "Worker PID $($process.Id) remains active; last monitored progress $inactiveMinutes minutes ago." '' 0 $process.Id $lastProgressAt.ToString('o')
            $nextHeartbeat = (Get-Date).AddMinutes(5)
        }
        Start-Sleep -Seconds 30
        $process.Refresh()
    }
    $process.WaitForExit()
    if ($process.ExitCode -ne 0) {
        throw "Soft full-recording runner exited with code $($process.ExitCode); inspect $stderrPath"
    }
    $campaignState = Get-Content -LiteralPath (Join-Path $output 'CAMPAIGN_STATE.json') -Raw | ConvertFrom-Json
    if ($campaignState.status -ne 'complete' -or $campaignState.completed -ne 24) {
        throw "Soft process exited zero but campaign state is $($campaignState.status), completed=$($campaignState.completed)/24."
    }
    Write-SoftQueueState 'complete' 'All 24 full-recording Soft fits verified complete.'
}
catch {
    Write-SoftQueueState 'stopped_on_guard_or_error' $_.Exception.Message
    exit 1
}
finally {
    if ($ownsMutex) { $mutex.ReleaseMutex() }
    $mutex.Dispose()
}
