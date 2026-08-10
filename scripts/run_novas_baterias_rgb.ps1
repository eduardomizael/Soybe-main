<#
Executa somente as novas baterias de treinamento:
Bloco 8 (40) -> Bloco 9 (40) -> Bloco 11 (20).
#>
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Runs = Join-Path $Root "models\pipeline_runs"
$StageDir = Join-Path $Root "backend\novas_baterias_stages"

Write-Host "[preparação] Gerando jobs das novas baterias..." -ForegroundColor Cyan
& uv run --offline python backend/generate_novas_baterias_rgb.py
if ($LASTEXITCODE -ne 0) { throw "Falha ao gerar os jobs." }

function Get-LatestRun([string]$ConfigPath) {
    $wanted = [IO.Path]::GetFullPath($ConfigPath)
    $files = @(Get-ChildItem -LiteralPath $Runs -Filter "pipeline_run_*.json" -File -ErrorAction SilentlyContinue |
        Sort-Object LastWriteTime -Descending)
    foreach ($file in $files) {
        try {
            $state = Get-Content -Raw -LiteralPath $file.FullName | ConvertFrom-Json
            if ([IO.Path]::GetFullPath([string]$state.config_path) -eq $wanted) {
                return $file.FullName
            }
        } catch { }
    }
    return $null
}

function Assert-Complete([string]$RunFile, [string]$Name) {
    $state = Get-Content -Raw -LiteralPath $RunFile | ConvertFrom-Json
    $pending = @($state.jobs | Where-Object { [string]$_.status -ne "success" })
    if ($pending.Count -gt 0) {
        $summary = ($pending | ForEach-Object { "$($_.id):$($_.status)" }) -join ", "
        throw "$Name incompleto: $summary"
    }
}

foreach ($stage in @("bloco8", "bloco9", "bloco11")) {
    $config = Join-Path $StageDir "$stage.toml"
    $run = Get-LatestRun $config
    if ($run) {
        Write-Host "[$stage] Retomando $run" -ForegroundColor Yellow
        & uv run --offline python -m backend.train_pipeline --run-file $run
    } else {
        Write-Host "[$stage] Iniciando $config" -ForegroundColor Green
        & uv run --offline python -m backend.train_pipeline --config $config
        $run = Get-LatestRun $config
    }
    if ($LASTEXITCODE -ne 0) { throw "$stage terminou com código $LASTEXITCODE." }
    if (-not $run) { throw "Run não encontrado para $stage." }
    Assert-Complete $run $stage
    Write-Host "[$stage] Consolidando..." -ForegroundColor Cyan
    & uv run --offline python backend/report_stage_results.py $run
    if ($LASTEXITCODE -ne 0) { throw "Consolidação falhou para $stage." }
}

Write-Host "Blocos 8, 9 e 11 concluídos e consolidados." -ForegroundColor Green
