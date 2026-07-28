param([ValidateSet("preflight","piloto","bloco1","bloco3","bloco4","bloco2","bloco5")][string]$Stage = "preflight", [string]$RunFile)
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Python = "uv run --offline python"
$stagePath = @{ piloto="backend/bateria_final_stages/piloto.toml"; bloco1="backend/bateria_final_stages/bloco1.toml"; bloco3="backend/bateria_final_stages/bloco3.toml"; bloco4="backend/bateria_final_stages/bloco4.toml"; bloco2="backend/bateria_final_stages/bloco2.toml"; bloco5="backend/bateria_final_stages/bloco5.toml" }
if ($Stage -eq "preflight") { & uv run --offline python backend/preflight_bateria_final_eduardo.py; exit $LASTEXITCODE }
if ($Stage -eq "piloto") { Write-Output "Executando somente o piloto exigido: ResNet50 / seed 42 / com_fundo." }
$Config = Join-Path $Root $stagePath[$Stage]
$Runs = Join-Path $Root "models/pipeline_runs"
$existing = $null
if ($RunFile) { $existing = [IO.Path]::GetFullPath((Join-Path $Root $RunFile)) }
if (-not $existing) {
  $existing = Get-ChildItem $Runs -Filter "pipeline_run_*.json" | Sort-Object LastWriteTime -Descending | ForEach-Object { $j=Get-Content -Raw $_.FullName|ConvertFrom-Json; if ([IO.Path]::GetFullPath([string]$j.config_path) -eq [IO.Path]::GetFullPath($Config)) { $_.FullName } } | Select-Object -First 1
}
if ($existing) { & uv run --offline python -m backend.train_pipeline --run-file $existing } else { & uv run --offline python -m backend.train_pipeline --config $Config }
if ($LASTEXITCODE -ne 0) { throw "Etapa $Stage terminou com código $LASTEXITCODE. Corrija/avalie antes da próxima etapa." }
if ($Stage -eq "piloto") { Write-Output "PILOTO CONCLUÍDO. Pare aqui e envie o CSV/resultado para avaliação antes de usar -Stage bloco1." }
