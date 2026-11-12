<#
Finalizzazione esplicita della Figura 2c. NON avvia training o plot.
Esempio (solo dopo richiesta esplicita):
  powershell -File codice\figura_2c_trial_safe\finalize_2c.ps1 -ConfirmCleanup
La verifica statica del file non lo esegue. -ConfirmCleanup abilita le rimozioni.
#>
[CmdletBinding()]
param(
    [switch]$ConfirmCleanup,
    [ValidateRange(1, 3600)][int]$PollSeconds = 30
)
Set-StrictMode -Version Latest
$ErrorActionPreference = 'Stop'
if (-not $ConfirmCleanup) { throw 'Operazione distruttiva non attivata: specificare -ConfirmCleanup su richiesta esplicita.' }

$project = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..\..')).ProviderPath
$figure = Join-Path $project 'results\figura_2_2026-10-28'
$new2c = Join-Path $figure '2c'
$rootPrefix = $project.TrimEnd('\') + '\'
$rootPath = (Get-Item -LiteralPath $project -Force).FullName
if ($rootPath -ne $project) { throw 'Radice progetto non canonica.' }

function Assert-SafePath([string]$Path, [bool]$MustExist = $true) {
    $full = [IO.Path]::GetFullPath($Path)
    if (-not $full.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase)) { throw "Percorso fuori progetto: $full" }
    if ($MustExist -and -not (Test-Path -LiteralPath $full)) { throw "Percorso mancante: $full" }
    # Reject junctions, symbolic links and other reparse points on every existing ancestor.
    $part = $full
    while ($part.Length -ge $project.Length) {
        if (Test-Path -LiteralPath $part) {
            $item = Get-Item -LiteralPath $part -Force
            if (($item.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse point non consentito: $part" }
            $resolved = (Resolve-Path -LiteralPath $part).ProviderPath
            if (-not ($resolved -eq $project -or $resolved.StartsWith($rootPrefix, [StringComparison]::OrdinalIgnoreCase))) { throw "Percorso risolto fuori progetto: $resolved" }
        }
        if ($part -eq $project) { break }
        $part = Split-Path -Parent $part
    }
    return $full
}
function Read-Json([string]$Path) {
    $null = Assert-SafePath $Path
    return Get-Content -LiteralPath $Path -Raw -Encoding UTF8 | ConvertFrom-Json
}
function Check-Hash([string]$Path, [string]$Expected) {
    $null = Assert-SafePath $Path
    if ($Expected -notmatch '^[0-9a-fA-F]{64}$' -or (Get-FileHash -LiteralPath $Path -Algorithm SHA256).Hash -ne $Expected) { throw "Hash non valido: $Path" }
}
function Check-Entries([string]$Directory, $Entries, [string[]]$Required) {
    if ($null -eq $Entries) { throw "Manifest senza artefatti: $Directory" }
    $properties = @($Entries.PSObject.Properties)
    foreach ($name in $Required) { if ($name -notin $properties.Name) { throw "Artefatto richiesto assente: $Directory\$name" } }
    if ($properties.Count -eq 0) { throw "Manifest vuoto: $Directory" }
    foreach ($entry in $properties) {
        if ([IO.Path]::GetFileName($entry.Name) -ne $entry.Name -or $entry.Name -in @('.', '..')) { throw "Nome artefatto non sicuro: $($entry.Name)" }
        Check-Hash (Join-Path $Directory $entry.Name) ([string]$entry.Value)
    }
}

$null = Assert-SafePath $new2c
$plotState = Join-Path $new2c 'PLOT_STATE.json'
while ($true) {
    if (Test-Path -LiteralPath $plotState) {
        $state = Read-Json $plotState
        if ($state.status -eq 'error') { throw "Supervisore in errore: $($state.error)" }
        if ($state.status -eq 'complete') { break }
    }
    Start-Sleep -Seconds $PollSeconds
}
$scopes = @('split', 'full')
$rats = @('achilles', 'gatsby')
$models = @('cebra_cnn', 'soft_stable_cnn', 'soft_stable_transformer')
$conditions = @('position', 'direction', 'position_direction', 'position_shuffled', 'direction_shuffled', 'position_direction_shuffled')
$campaign = Read-Json (Join-Path $new2c 'CAMPAIGN_STATE.json')
if ($campaign.total -ne 72 -or $campaign.done -ne 72 -or -not $campaign.finished -or $null -ne $campaign.current) { throw 'Campagna non completata correttamente.' }
$allManifests = @(Get-ChildItem -LiteralPath (Join-Path $new2c 'fits') -Filter 'manifest.json' -File -Recurse -Force)
if ($allManifests.Count -ne 72) { throw "Attesi 72 manifest, trovati $($allManifests.Count)." }
$manifestCount = 0; $diagnosticCount = 0; $comparisonCount = 0
foreach ($scope in $scopes) { foreach ($rat in $rats) {
    foreach ($model in $models) {
        foreach ($condition in $conditions) {
            $dir = Join-Path $new2c "fits\$scope\$rat\$model\$condition"
            $manifest = Read-Json (Join-Path $dir 'manifest.json')
            if ($manifest.scope -ne $scope -or $manifest.rat -ne $rat -or $manifest.model -ne $model -or $manifest.condition -ne $condition -or $manifest.signature -ne $campaign.signature) { throw "Identita manifest errata: $dir" }
            $required = @('checkpoint.pt', 'training_history.csv')
            if ($scope -eq 'split') { $required += @('train_embedding.npz', 'valid_embedding.npz', 'test_embedding.npz') } else { $required += 'full_embedding.npz' }
            if ($model -like 'soft*') { $required += 'distance_scales.json' }
            Check-Entries $dir $manifest.artifacts $required
            $manifestCount++
        }
        $diag = Join-Path $new2c "plot\$scope\$rat\$model"
        $run = Read-Json (Join-Path $diag 'diagnostics_run.json')
        if ($run.scope -ne $scope -or $run.rat -ne $rat -or $run.model -ne $model) { throw "Identita diagnostica errata: $diag" }
        $complete = Read-Json (Join-Path $diag 'diagnostics_complete.json')
        $requiredDiag = @('ari.csv','silhouette.csv','cka.csv','partitions.csv','summary.json')
        if ($scope -eq 'split') { $requiredDiag += @('knn_metrics.csv', 'knn_test_predictions.csv') }
        Check-Entries $diag $complete $requiredDiag
        $diagnosticCount++
    }
    $comp = Join-Path $new2c "plot\$scope\$rat\confronto_modelli"
    $provenance = Read-Json (Join-Path $comp 'comparison_provenance.json')
    if ($provenance.scope -ne $scope -or $provenance.rat -ne $rat -or @($provenance.diagnostics).Count -ne 3) { throw "Provenienza confronto errata: $comp" }
    foreach ($model in $models) {
        $matching = @($provenance.diagnostics | Where-Object { $_.model -eq $model })
        if ($matching.Count -ne 1) { throw "Provenienza confronto incompleta: $comp/$model" }
        $diag = Join-Path $new2c "plot\$scope\$rat\$model"
        if ([IO.Path]::GetFullPath([string]$matching[0].path) -ne [IO.Path]::GetFullPath($diag)) { throw "Percorso diagnostico incoerente: $comp/$model" }
        Check-Hash (Join-Path $diag 'diagnostics_run.json') ([string]$matching[0].run_sha256)
        Check-Hash (Join-Path $diag 'diagnostics_complete.json') ([string]$matching[0].complete_sha256)
    }
    $comparisonFiles = @('ari.csv','silhouette.csv','cka.csv','support_audit.csv','ari.png','silhouette.png','cka.png')
    if ($scope -eq 'split') { $comparisonFiles += @('decoding.csv','test_accuracy.png','test_balanced_accuracy.png') }
    foreach ($name in $comparisonFiles) { $null = Assert-SafePath (Join-Path $comp $name) }
    $comparisonCount++
} }
if ($manifestCount -ne 72 -or $diagnosticCount -ne 12 -or $comparisonCount -ne 4) { throw 'Cardinalita non valida.' }

$referencesPath = Join-Path $figure 'RIFERIMENTI.md'
$null = Assert-SafePath $referencesPath
$utf8 = New-Object System.Text.UTF8Encoding($false, $true)
$referenceBytes = [IO.File]::ReadAllBytes($referencesPath)
$hasBom = $referenceBytes.Length -ge 3 -and $referenceBytes[0] -eq 239 -and $referenceBytes[1] -eq 187 -and $referenceBytes[2] -eq 191
$references = $utf8.GetString($referenceBytes)
$header = '## Aggiornamento 2c'
$index = $references.IndexOf($header, [StringComparison]::Ordinal)
if ($index -lt 0 -or $references.IndexOf($header, $index + 1, [StringComparison]::Ordinal) -ge 0) { throw 'Sezione 2c non identificabile univocamente in RIFERIMENTI.md.' }
$next = [regex]::Match($references.Substring($index + $header.Length), '(?m)^## ')
$end = if ($next.Success) { $index + $header.Length + $next.Index } else { $references.Length }
$updated = $references.Remove($index, $end - $index)
$updated = $updated.Replace('plot/b|c|d/<ratto>/', 'plot/b|d/<ratto>/')
$updated = $updated.Replace('Le tavole b e c usano le stime registrate nei riferimenti per grafico.', 'La tavola b usa le stime registrate nei riferimenti per grafico.')
$updated = [regex]::Replace($updated, ' In 2c alcune stime 8D[^\r\n]*?questi fit 10000\.', '')
if ($updated -match 'plot/b\|c\|d|tavole b e c|In 2c alcune stime 8D|figure2c_shuffle_diagnostic_2026-11-02|processo/provenienza/c/|plot/c/') { throw 'Riferimenti 2c precedenti non rimossi: revisione manuale necessaria.' }
$newline = if ($references.Contains("`r`n")) { "`r`n" } else { "`n" }
$updated = $updated.TrimEnd("`r", "`n") + $newline + $newline + '## Figura 2c — campagna trial-safe' + $newline + $newline + 'Risultati, fit e diagnostica: `2c/` e `2c/plot/`.' + $newline

$targets = @(
    (Join-Path $project 'results\figure2c_shuffle_diagnostic_2026-11-02'),
    (Join-Path $figure 'c'),
    (Join-Path $figure 'plot\c'),
    (Join-Path $figure 'processo\provenienza\c'),
    (Join-Path $figure 'processo\metriche\c')
)
$inventory = @()
foreach ($target in $targets) {
    $full = Assert-SafePath $target $false
    $exists = Test-Path -LiteralPath $full
    $bytes = 0; $files = 0
    if ($exists) {
        $item = Get-Item -LiteralPath $full -Force
        if (-not $item.PSIsContainer) { throw "La destinazione non e una cartella: $full" }
        # Refuse links anywhere inside the tree before measuring or removing it.
        $children = @(Get-ChildItem -LiteralPath $full -Force -Recurse)
        foreach ($child in $children) {
            if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse point nell'albero: $($child.FullName)" }
            $null = Assert-SafePath $child.FullName
            if (-not $child.PSIsContainer) { $bytes += $child.Length; $files++ }
        }
    }
    $inventory += [ordered]@{ path = $full; exists = $exists; files = $files; bytes = $bytes }
}
$logDir = Join-Path $new2c ('finalization_' + [datetime]::UtcNow.ToString('yyyyMMddTHHmmssfffZ') + '_' + [guid]::NewGuid().ToString('N'))
$null = Assert-SafePath $logDir $false
$null = New-Item -ItemType Directory -Path $logDir
[IO.File]::WriteAllBytes((Join-Path $logDir 'RIFERIMENTI.before.md'), $referenceBytes)
$logPath = Join-Path $logDir 'inventory.json'
$log = [ordered]@{ status = 'verified_before_removal'; utc = [datetime]::UtcNow.ToString('o'); manifests = $manifestCount; diagnostics = $diagnosticCount; comparisons = $comparisonCount; targets = $inventory; removed = @() }
[IO.File]::WriteAllText($logPath, ($log | ConvertTo-Json -Depth 8), $utf8)
try {
    foreach ($target in $inventory) {
        if ($target.exists) {
            $null = Assert-SafePath $target.path
            # Recheck the full tree immediately before recursive removal.
            foreach ($child in @(Get-ChildItem -LiteralPath $target.path -Force -Recurse)) {
                if (($child.Attributes -band [IO.FileAttributes]::ReparsePoint) -ne 0) { throw "Reparse point nell'albero: $($child.FullName)" }
                $null = Assert-SafePath $child.FullName
            }
            Remove-Item -LiteralPath $target.path -Recurse -Force
            $log.removed += $target.path
            [IO.File]::WriteAllText($logPath, ($log | ConvertTo-Json -Depth 8), $utf8)
        }
    }
    if ((Get-FileHash -LiteralPath $referencesPath -Algorithm SHA256).Hash -ne (Get-FileHash -LiteralPath (Join-Path $logDir 'RIFERIMENTI.before.md') -Algorithm SHA256).Hash) { throw 'RIFERIMENTI.md modificato durante la finalizzazione.' }
    [IO.File]::WriteAllText($referencesPath, $updated, $utf8)
    $log.status = 'complete'; $log['completed_utc'] = [datetime]::UtcNow.ToString('o')
    [IO.File]::WriteAllText($logPath, ($log | ConvertTo-Json -Depth 8), $utf8)
} catch {
    $log.status = 'error'; $log['error'] = $_.Exception.Message
    [IO.File]::WriteAllText($logPath, ($log | ConvertTo-Json -Depth 8), $utf8)
    throw
}
Write-Output "Finalizzazione completata: $logPath"
