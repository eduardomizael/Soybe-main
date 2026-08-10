<#
Executa somente o Bloco 12 depois que a bateria antiga terminar.
Não toca nos pipeline_run existentes dos blocos anteriores.
#>
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Config = Join-Path $Root "backend\bateria_final_stages\bloco12_convnext_random.toml"
$Runs = Join-Path $Root "models\pipeline_runs"

& uv run --offline python backend/generate_bloco12_convnext_random.py
if ($LASTEXITCODE -ne 0) { throw "Falha ao gerar a configuração do Bloco 12." }

$wanted = [IO.Path]::GetFullPath($Config)
$run = $null
Get-ChildItem -LiteralPath $Runs -Filter "pipeline_run_*.json" -File |
    Sort-Object LastWriteTime -Descending |
    ForEach-Object {
        if ($run) { return }
        $state = Get-Content -Raw -LiteralPath $_.FullName | ConvertFrom-Json
        if ([IO.Path]::GetFullPath([string]$state.config_path) -eq $wanted) { $run = $_.FullName }
    }

if ($run) {
    & uv run --offline python -m backend.train_pipeline --run-file $run
} else {
    & uv run --offline python -m backend.train_pipeline --config $Config
    $run = Get-ChildItem -LiteralPath $Runs -Filter "pipeline_run_*.json" -File |
        Sort-Object LastWriteTime -Descending | Select-Object -First 1 -ExpandProperty FullName
}
if ($LASTEXITCODE -ne 0) { throw "Bloco 12 terminou com código $LASTEXITCODE." }

$state = Get-Content -Raw -LiteralPath $run | ConvertFrom-Json
$pending = @($state.jobs | Where-Object { [string]$_.status -ne "success" })
if ($pending.Count -gt 0) { throw "Bloco 12 incompleto; execute novamente para retomar." }

& uv run --offline python backend/report_stage_results.py $run
if ($LASTEXITCODE -ne 0) { throw "Consolidação do Bloco 12 falhou." }
Write-Host "Bloco 12 concluído e consolidado." -ForegroundColor Green
