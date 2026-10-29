$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$controllerPath = Join-Path $workspace 'results\rat_paper_fig1e_ed4_6seed_v1\GPU_QUEUE_RECOVERY_20261023.ps1'
$watcherPath = Join-Path $PSScriptRoot 'run_soft_fig1d_fullrecording_after_gpu_queue.ps1'

function Import-QueueFunction([string]$sourcePath, [string]$functionName) {
    $parseTokens = $null
    $parseErrors = $null
    $tree = [System.Management.Automation.Language.Parser]::ParseFile($sourcePath, [ref]$parseTokens, [ref]$parseErrors)
    if ($parseErrors.Count) { throw "Invalid PowerShell source: $sourcePath" }
    $definition = $tree.Find({ param($node)
        $node -is [System.Management.Automation.Language.FunctionDefinitionAst] -and $node.Name -eq $functionName
    }, $true)
    if (-not $definition) { throw "Missing function: $functionName" }
    return $definition.Extent.Text
}

# Load functions only: never run the controller or the training commands.
Invoke-Expression (Import-QueueFunction $controllerPath 'Write-QueueState')
Invoke-Expression (Import-QueueFunction $controllerPath 'Wait-ExpectedProcess')
Invoke-Expression (Import-QueueFunction $watcherPath 'Get-VerifiedProcess')
Invoke-Expression (Import-QueueFunction $watcherPath 'Write-SoftQueueState')
$probeRoot = Join-Path ([System.IO.Path]::GetTempPath()) ('AI_PhD_queue_test_' + [guid]::NewGuid().ToString('N'))
New-Item -ItemType Directory -Path $probeRoot | Out-Null
$campaign = $probeRoot
$tag = 'regression_probe'
$queueStatePath = Join-Path $probeRoot 'GPU_QUEUE_STATE.json'
$queueLogPath = Join-Path $probeRoot 'queue.log'
Write-QueueState 'probe' 'waiting' 'Atomic initial state test.'
Write-QueueState 'probe' 'running' 'Atomic replacement test.'
$state = Get-Content -LiteralPath $queueStatePath -Raw | ConvertFrom-Json
if ($state.status -ne 'running' -or $state.controller_pid -ne $PID) { throw 'Atomic state update regression' }
'PASS atomic state replacement on Windows PowerShell'

# Reproduce a reader that permits reads/writes but temporarily denies deletion.
# File.Replace must survive this real Windows sharing conflict for both writers.
$statePath = Join-Path $probeRoot 'SOFT_QUEUE_STATE.json'
Write-SoftQueueState 'waiting' 'Initial watcher state.'
foreach ($lockedPath in @($queueStatePath, $statePath)) {
    $reader = Start-Job -ArgumentList $lockedPath -ScriptBlock {
        param($lockedPath)
        $stream = [System.IO.File]::Open($lockedPath, [System.IO.FileMode]::Open,
            [System.IO.FileAccess]::Read, [System.IO.FileShare]::ReadWrite)
        try { 'reader_locked'; Start-Sleep -Milliseconds 1200 }
        finally { $stream.Dispose() }
    }
    try {
        $readerDeadline = (Get-Date).AddSeconds(15)
        while (@(Receive-Job -Job $reader -Keep) -notcontains 'reader_locked') {
            if ((Get-Date) -gt $readerDeadline -or $reader.State -eq 'Failed') { throw 'Reader-lock regression setup failed' }
            Start-Sleep -Milliseconds 20
        }
        $writeTimer = [System.Diagnostics.Stopwatch]::StartNew()
        if ($lockedPath -eq $queueStatePath) {
            Write-QueueState 'probe' 'running' 'Recovered from reader sharing conflict.'
        } else {
            Write-SoftQueueState 'running' 'Recovered from reader sharing conflict.'
        }
        $writeTimer.Stop()
        if ($writeTimer.ElapsedMilliseconds -lt 250) { throw 'Sharing-conflict test did not exercise the retry' }
        $state = Get-Content -LiteralPath $lockedPath -Raw | ConvertFrom-Json
        if ($state.status -ne 'running') { throw 'State replacement failed after reader released the file' }
        Wait-Job -Job $reader | Out-Null
    } finally {
        if ($reader.State -eq 'Running') { Stop-Job -Job $reader }
        Remove-Job -Job $reader
    }
}
'PASS controller and watcher recover from real Windows reader sharing conflicts'

$self = Get-Process -Id $PID
if (-not (Get-VerifiedProcess $PID $self.Path $self.StartTime.ToString('o'))) { throw 'Live process rejected' }
if (Get-VerifiedProcess $PID 'C:\not-the-worker.exe') { throw 'Reused executable identity accepted' }
if (Get-VerifiedProcess $PID $self.Path $self.StartTime.AddHours(-1).ToString('o')) { throw 'Reused PID start time accepted' }
'PASS live process and reused PID identity checks'

# Exercise 25 minutes of queue monitoring with a virtual clock: no actual waits,
# no real worker processes, and no possibility of stopping a fit.
function Get-Date { return $script:probeClock }
function Start-Sleep { param([int]$Seconds) $script:probeClock = $script:probeClock.AddSeconds($Seconds) }
function Stop-Process { throw 'Regression: queue attempted to terminate a worker' }
function Get-Process {
    param([int]$Id, $ErrorAction)
    if ($script:probeClock -ge $script:probeEnd) { return $null }
    return [pscustomobject]@{ Path = 'C:\probe\python.exe'; StartTime = $script:probeStart; CPU = 10 }
}
function Get-StageActivityFingerprint {
    param([string]$stage)
    if ($script:probeHasProgress) { return $script:probeClock.ToString('o') }
    return 'unchanged'
}
function Write-QueueState {
    param([string]$stage, [string]$status, [string]$detail, [int]$workerPid, [string]$workerPath)
    $script:probeEvents.Add([pscustomobject]@{ status = $status; at = $script:probeClock; detail = $detail })
}

foreach ($hasProgress in @($true, $false)) {
    $script:probeStart = [datetime]'2026-01-01T12:00:00'
    $script:probeClock = $script:probeStart
    $script:probeEnd = $script:probeStart.AddMinutes(25)
    $script:probeHasProgress = $hasProgress
    $script:probeEvents = [System.Collections.Generic.List[object]]::new()
    Wait-ExpectedProcess 123 'C:\probe\python.exe' 'conv_pivae_existing'
    $diagnostics = @($script:probeEvents | Where-Object status -eq 'inactivity_diagnostic')
    if ($hasProgress -and $diagnostics.Count) { throw 'Progressing fit incorrectly marked inactive' }
    if (-not $hasProgress -and ($diagnostics.Count -ne 1 -or $diagnostics[0].at -ne $script:probeStart.AddMinutes(20))) {
        throw 'Inactive fit did not produce its 20-minute diagnostic'
    }
    if ($script:probeClock -lt $script:probeEnd) { throw 'Long-running fit monitoring exited prematurely' }
}
'PASS inactivity diagnostic after 20 minutes; progress resets timer; long fits continue'
'ALL_QUEUE_REGRESSION_CHECKS_PASSED'
