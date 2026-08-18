<#
Executa e retoma a bateria completa do Bloco 13 (60 jobs).

Pré-requisito: preparar_arvore_bloco13.py e o pre-voo já concluídos com sucesso.
Os artefatos consolidados são copiados para reports/bloco13_aleatorio_limpo/;
os pesos permanecem somente em models/ e não entram no pacote versionável.
#>
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Config = Join-Path $Root "backend\bateria_final_stages\bloco13.toml"
$PreflightConfig = Join-Path $Root "backend\bateria_final_stages\bloco13_prevoo.toml"
$Runs = Join-Path $Root "models\pipeline_runs"
$FilteredData = Join-Path $Root "data\com_fundo_filtrado"
$Output = Join-Path $Root "reports\bloco13_aleatorio_limpo"
$ExpectedJobs = 60

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

function Read-Run([string]$RunFile) {
    return Get-Content -Raw -LiteralPath $RunFile | ConvertFrom-Json
}

if (-not (Test-Path -LiteralPath $FilteredData -PathType Container)) {
    throw "Árvore filtrada ausente. Execute primeiro: uv run python preparar_arvore_bloco13.py"
}
if (-not (Test-Path -LiteralPath $Config -PathType Leaf)) {
    throw "Configuração ausente: $Config"
}

$PreflightRun = Get-LatestRun $PreflightConfig
if (-not $PreflightRun) { throw "Pré-voo não encontrado. Execute bloco13_prevoo.toml antes da bateria." }
$PreflightState = Read-Run $PreflightRun
$PreflightPending = @($PreflightState.jobs | Where-Object { [string]$_.status -ne "success" })
if ($PreflightPending.Count -gt 0) { throw "Pré-voo incompleto; não iniciar a bateria." }
Write-Host "[prevoo] OK: $PreflightRun" -ForegroundColor Green

$run = Get-LatestRun $Config
if ($run) {
    Write-Host "[bloco13] Retomando $run" -ForegroundColor Yellow
    & uv run --offline python -m backend.train_pipeline --run-file $run
} else {
    Write-Host "[bloco13] Iniciando bateria completa: $Config" -ForegroundColor Green
    & uv run --offline python -m backend.train_pipeline --config $Config
    $run = Get-LatestRun $Config
}
if ($LASTEXITCODE -ne 0) { throw "Bloco 13 terminou com código $LASTEXITCODE." }
if (-not $run) { throw "Run-file do Bloco 13 não encontrado." }

$state = Read-Run $run
$total = @($state.jobs).Count
if ($total -ne $ExpectedJobs) { throw "Configuração inesperada: $total jobs; esperado $ExpectedJobs." }
$pending = @($state.jobs | Where-Object { [string]$_.status -ne "success" })
if ($pending.Count -gt 0) {
    $summary = ($pending | ForEach-Object { "$($_.id):$($_.status)" }) -join ", "
    throw "Bloco 13 incompleto ($($pending.Count) pendentes): $summary"
}

Write-Host "[bloco13] 60/60 concluídos. Consolidando..." -ForegroundColor Cyan
& uv run --offline python backend/report_stage_results.py $run
if ($LASTEXITCODE -ne 0) { throw "Consolidação do Bloco 13 falhou." }

$stamp = $state.run_id
$Analysis = Join-Path $Root "models\bloco13_analysis_$stamp"
$Package = Join-Path $Root "models\bloco13_package_$stamp"
New-Item -ItemType Directory -Force -Path (Join-Path $Output "resultados"), (Join-Path $Output "reports"), (Join-Path $Output "source") | Out-Null
Copy-Item -LiteralPath (Join-Path $Analysis "metricas_por_execucao.csv") -Destination (Join-Path $Output "resultados\metricas_por_execucao.csv") -Force
Copy-Item -LiteralPath (Join-Path $Analysis "resumo_por_arquitetura.csv") -Destination (Join-Path $Output "resultados\resumo_por_arquitetura.csv") -Force
Copy-Item -LiteralPath (Join-Path $Analysis "relatorio_consolidado.md") -Destination (Join-Path $Output "reports\relatorio_consolidado.md") -Force
Copy-Item -Path (Join-Path $Package "reports\*") -Destination (Join-Path $Output "reports") -Force
Copy-Item -Path (Join-Path $Package "predictions\*") -Destination (Join-Path $Output "resultados") -Force
Copy-Item -LiteralPath $run -Destination (Join-Path $Output "source") -Force
Copy-Item -LiteralPath $Config -Destination (Join-Path $Output "source") -Force

Write-Host "Bloco 13 concluído e consolidado em $Output" -ForegroundColor Green
