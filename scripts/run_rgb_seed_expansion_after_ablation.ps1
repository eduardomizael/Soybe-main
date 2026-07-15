param(
    [string]$ConfirmationRunFile = "models/pipeline_runs/pipeline_run_20260714_210215.json",
    [int]$PollSeconds = 30
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfirmationPath = Join-Path $WorkspaceRoot $ConfirmationRunFile
$ExpansionConfig = "backend/training_jobs_rgb_seed_expansion_10.toml"
$ExpansionConfigPath = Join-Path $WorkspaceRoot $ExpansionConfig

if (-not (Test-Path -LiteralPath $ConfirmationPath)) {
    throw "Run de confirmacao nao encontrado: $ConfirmationPath"
}
if (-not (Test-Path -LiteralPath $ExpansionConfigPath)) {
    throw "Configuracao da expansao nao encontrada: $ExpansionConfigPath"
}

Set-Location $WorkspaceRoot
Write-Output "Aguardando a confirmacao da ablacao: $ConfirmationPath"

while ($true) {
    $run = Get-Content -Raw -LiteralPath $ConfirmationPath | ConvertFrom-Json
    $pending = @($run.jobs | Where-Object { $_.status -eq "pending" }).Count
    $running = @($run.jobs | Where-Object { $_.status -eq "running" }).Count
    $success = @($run.jobs | Where-Object { $_.status -eq "success" }).Count
    $errors = @($run.jobs | Where-Object { $_.status -eq "error" }).Count

    Write-Output "Confirmacao: success=$success running=$running pending=$pending error=$errors"
    if ($errors -gt 0) {
        throw "A confirmacao terminou com erro; a expansao nao sera iniciada."
    }
    if ($pending -eq 0 -and $running -eq 0) {
        if ($success -ne @($run.jobs).Count) {
            throw "A confirmacao terminou sem que todos os jobs fossem success."
        }
        break
    }
    Start-Sleep -Seconds ([Math]::Max(10, $PollSeconds))
}

$existingExpansionRuns = Get-ChildItem -LiteralPath (Join-Path $WorkspaceRoot "models/pipeline_runs") -Filter "pipeline_run_*.json" |
    ForEach-Object {
        try {
            $candidate = Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json
            if ([IO.Path]::GetFullPath([string]$candidate.config_path) -eq [IO.Path]::GetFullPath($ExpansionConfigPath)) {
                $candidate
            }
        }
        catch {
            Write-Warning "Ignorando run file invalido: $($_.FullName)"
        }
    }

if (@($existingExpansionRuns).Count -gt 0) {
    throw "Ja existe um pipeline_run para a expansao RGB; inicio duplicado bloqueado."
}

Write-Output "Confirmacao concluida sem erros. Iniciando a expansao RGB de 30 jobs."
& uv run --offline python -m backend.train_pipeline --config $ExpansionConfig
if ($LASTEXITCODE -ne 0) {
    throw "A fila de expansao terminou com codigo $LASTEXITCODE; analise nao executada."
}

Write-Output "Expansao concluida. Gerando consolidacao estatistica."
& uv run --offline python -m backend.analyze_rgb_seed_expansion
if ($LASTEXITCODE -ne 0) {
    throw "A consolidacao estatistica terminou com codigo $LASTEXITCODE."
}

Write-Output "Fluxo RGB concluido com sucesso."
