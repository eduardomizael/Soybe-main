param([string]$RunFile)

$ErrorActionPreference = "Stop"
$WorkspaceRoot = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
$ConfigPath = Join-Path $WorkspaceRoot "backend/training_jobs_bateria_final_eduardo.toml"
$RunsDir = Join-Path $WorkspaceRoot "models/pipeline_runs"
Set-Location $WorkspaceRoot

if (-not [string]::IsNullOrWhiteSpace($RunFile)) {
    $Candidate = if ([IO.Path]::IsPathRooted($RunFile)) { $RunFile } else { Join-Path $WorkspaceRoot $RunFile }
    $Existing = [PSCustomObject]@{ Path = [IO.Path]::GetFullPath($Candidate); Run = (Get-Content -Raw -LiteralPath $Candidate | ConvertFrom-Json) }
} else {
    $Existing = Get-ChildItem -LiteralPath $RunsDir -Filter "pipeline_run_*.json" |
        Sort-Object LastWriteTime -Descending |
        ForEach-Object {
            $Run = Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json
            if ([IO.Path]::GetFullPath([string]$Run.config_path) -eq [IO.Path]::GetFullPath($ConfigPath)) {
                [PSCustomObject]@{ Path = $_.FullName; Run = $Run }
            }
        } | Select-Object -First 1
}

if ($null -eq $Existing) {
    Write-Output "Criando a fila da bateria final (180 jobs)."
    & uv run --offline python -m backend.train_pipeline --config $ConfigPath
} else {
    $Remaining = @($Existing.Run.jobs | Where-Object { $_.status -ne "success" }).Count
    if ($Remaining -eq 0) { Write-Output "Bateria final já concluída: $($Existing.Path)"; exit 0 }
    Write-Output "Retomando $Remaining job(s): $($Existing.Path)"
    & uv run --offline python -m backend.train_pipeline --run-file $Existing.Path
}
if ($LASTEXITCODE -ne 0) { throw "Pipeline terminou com código $LASTEXITCODE. Use o mesmo -RunFile para retomar." }
