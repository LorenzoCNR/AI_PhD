param(
    [Parameter(Mandatory=$true)][int]$OldControllerPid,
    [Parameter(Mandatory=$true)][string]$ExpectedStartedLocal,
    [Parameter(Mandatory=$true)][string]$AuditDirectory
)
$ErrorActionPreference = 'Stop'
$workspace = Split-Path -Parent $PSScriptRoot
$campaign = Join-Path $workspace 'results\rat_paper_fig1e_ed4_6seed_v1'
$queuePath = Join-Path $campaign 'GPU_QUEUE_STATE.json'
$statePath = Join-Path $AuditDirectory 'STABLE_HANDOVER_STATE.json'
$powershellPath = Join-Path $env:SystemRoot 'System32\WindowsPowerShell\v1.0\powershell.exe'

function Write-HandoverState([string]$status, [string]$detail, [int]$controllerPid = 0) {
    $state = [ordered]@{status=$status; detail=$detail; watcher_pid=$PID;
        previous_controller_pid=$OldControllerPid; next_controller_pid=$controllerPid;
        updated_local=(Get-Date).ToString('o')}
    [System.IO.File]::WriteAllText($statePath, ($state | ConvertTo-Json), [System.Text.UTF8Encoding]::new($false))
}

try {
    Write-HandoverState 'waiting_for_soft_boundary' 'Preserving the current controller and all active fit output streams; waiting for the legacy Soft preflight to reject the superseded two-version grid.'
    while ($true) {
        $previous = Get-Process -Id $OldControllerPid -ErrorAction SilentlyContinue
        if (-not $previous) { break }
        if ($previous.Path -ine $powershellPath -or
            [math]::Abs(($previous.StartTime - [datetime]::Parse($ExpectedStartedLocal).ToLocalTime()).TotalSeconds) -gt 1) {
            throw 'Previous controller PID identity changed; refusing automatic handover.'
        }
        Start-Sleep -Seconds 30
    }
    $final = Get-Content -LiteralPath $queuePath -Raw | ConvertFrom-Json
    if ([int]$final.controller_pid -ne $OldControllerPid -or
        $final.status -ne 'stopped_on_guard_or_error' -or
        $final.detail -notmatch '^Stage soft_dryrun_pre exited with code 1;') {
        throw "Controller did not stop at the expected Soft-only transition: $($final.status), $($final.detail)"
    }
    $scriptPath = Join-Path $campaign 'GPU_QUEUE_RECOVERY_20261023.ps1'
    $next = Start-Process -FilePath $powershellPath `
        -ArgumentList @('-NoProfile','-ExecutionPolicy','Bypass','-File',('"'+$scriptPath+'"')) `
        -WorkingDirectory $workspace -WindowStyle Hidden -PassThru `
        -RedirectStandardOutput (Join-Path $AuditDirectory 'stable_controller.stdout.log') `
        -RedirectStandardError (Join-Path $AuditDirectory 'stable_controller.stderr.log')
    [void]$next.Handle
    Write-HandoverState 'controller_restarted' 'Updated controller will re-audit completed CEBRA/piVAE stages and run only the approved 432 Stable Soft cells.' $next.Id
    # Remain alive to retain redirected stream readers for the new controller.
    $next.WaitForExit()
    if ($next.ExitCode -ne 0) { throw "Updated controller exited with code $($next.ExitCode)." }
    Write-HandoverState 'complete' 'Updated controller completed successfully.' $next.Id
} catch {
    Write-HandoverState 'stopped_on_guard_or_error' $_.Exception.Message
    exit 1
}
