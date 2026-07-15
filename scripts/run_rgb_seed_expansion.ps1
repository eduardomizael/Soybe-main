param(
    [string]$ConfirmationRunFile = "models/pipeline_runs/pipeline_run_20260714_210215.json"
)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfirmationPath = Join-Path $WorkspaceRoot $ConfirmationRunFile
$ExpansionConfig = Join-Path $WorkspaceRoot "backend/training_jobs_rgb_seed_expansion_10.toml"
$PipelineRunsDir = Join-Path $WorkspaceRoot "models/pipeline_runs"

if (-not (Test-Path -LiteralPath $ConfirmationPath)) {
    throw "Run de confirmacao nao encontrado: $ConfirmationPath"
}
if (-not (Test-Path -LiteralPath $ExpansionConfig)) {
    throw "Configuracao da expansao nao encontrada: $ExpansionConfig"
}

$confirmation = Get-Content -Raw -LiteralPath $ConfirmationPath | ConvertFrom-Json
$pending = @($confirmation.jobs | Where-Object { $_.status -eq "pending" }).Count
$running = @($confirmation.jobs | Where-Object { $_.status -eq "running" }).Count
$success = @($confirmation.jobs | Where-Object { $_.status -eq "success" }).Count
$errors = @($confirmation.jobs | Where-Object { $_.status -eq "error" }).Count

if ($errors -gt 0) {
    throw "A confirmacao da ablacao tem $errors job(s) com erro. Corrija antes da expansao."
}
if ($pending -gt 0 -or $running -gt 0) {
    throw "A confirmacao ainda nao terminou: success=$success running=$running pending=$pending."
}
if ($success -ne @($confirmation.jobs).Count) {
    throw "A confirmacao terminou sem que todos os jobs fossem success."
}

Set-Location $WorkspaceRoot
$trainingExitCode = 0
$existingRun = Get-ChildItem -LiteralPath $PipelineRunsDir -Filter "pipeline_run_*.json" |
    Sort-Object LastWriteTime -Descending |
    ForEach-Object {
        try {
            $candidate = Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json
            if ([IO.Path]::GetFullPath([string]$candidate.config_path) -eq [IO.Path]::GetFullPath($ExpansionConfig)) {
                [PSCustomObject]@{ Path = $_.FullName; Run = $candidate }
            }
        }
        catch {
            Write-Warning "Ignorando run file invalido: $($_.FullName)"
        }
    } |
    Select-Object -First 1

if ($null -eq $existingRun) {
    Write-Output "Iniciando a expansao RGB de 30 jobs."
    & uv run --offline python -m backend.train_pipeline --config $ExpansionConfig
    $trainingExitCode = $LASTEXITCODE
}
else {
    $remaining = @($existingRun.Run.jobs | Where-Object { $_.status -ne "success" }).Count
    if ($remaining -gt 0) {
        Write-Output "Retomando a expansao existente: $($existingRun.Path)"
        & uv run --offline python -m backend.train_pipeline --run-file $existingRun.Path
        $trainingExitCode = $LASTEXITCODE
    }
    else {
        Write-Output "A expansao ja esta concluida; treinamento sera ignorado."
    }
}

if ($trainingExitCode -ne 0) {
    throw "A fila de expansao terminou com codigo $trainingExitCode; analise nao executada."
}

Write-Output "Gerando consolidacao estatistica."
& uv run --offline python -m backend.analyze_rgb_seed_expansion
if ($LASTEXITCODE -ne 0) {
    throw "A consolidacao estatistica terminou com codigo $LASTEXITCODE."
}

Write-Output "Fluxo RGB concluido com sucesso."
