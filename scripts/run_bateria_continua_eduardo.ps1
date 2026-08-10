<#
.SYNOPSIS
    Executa a bateria final em sequência, com retomada automática e empacotamento.

.DESCRIPTION
    A ordem é a definida para a bateria final:
    piloto -> bloco1 -> bloco3 -> bloco4 -> bloco2 -> bloco5.

    Cada etapa usa o pipeline JSON existente. Ao ser executado novamente,
    o script reutiliza o último pipeline_run da etapa e o train_pipeline pula
    os jobs com status success.
#>

$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root

$Runs = Join-Path $Root "models\pipeline_runs"
$StageDir = Join-Path $Root "backend\bateria_final_stages"

$Stages = @(
    @{ Name = "piloto"; Config = (Join-Path $StageDir "piloto.toml") },
    @{ Name = "bloco1"; Config = (Join-Path $StageDir "bloco1.toml") },
    @{ Name = "bloco3"; Config = (Join-Path $StageDir "bloco3.toml") },
    @{ Name = "bloco4"; Config = (Join-Path $StageDir "bloco4.toml") },
    @{ Name = "bloco2"; Config = (Join-Path $StageDir "bloco2.toml") },
    @{ Name = "bloco5"; Config = (Join-Path $StageDir "bloco5.toml") },
    @{ Name = "bloco12_convnext_random"; Config = (Join-Path $StageDir "bloco12_convnext_random.toml") }
)

function Get-RunState($Path) {
    return Get-Content -Raw -LiteralPath $Path | ConvertFrom-Json
}

function Get-LatestRun([string]$ConfigPath) {
    if (-not (Test-Path -LiteralPath $Runs)) { return $null }

    $wanted = [IO.Path]::GetFullPath($ConfigPath)
    $matches = @(
        Get-ChildItem -LiteralPath $Runs -Filter "pipeline_run_*.json" -File |
            Sort-Object LastWriteTime -Descending |
            ForEach-Object {
                try {
                    $state = Get-RunState $_.FullName
                    if ([IO.Path]::GetFullPath([string]$state.config_path) -eq $wanted) {
                        $_
                    }
                } catch {
                    Write-Warning "Ignorando run JSON inválido: $($_.FullName)"
                }
            }
    )
    if ($matches.Count -gt 0) { return $matches[0].FullName }
    return $null
}

function Assert-StageComplete([string]$RunFile, [string]$StageName) {
    $state = Get-RunState $RunFile
    $jobs = @($state.jobs)
    $pending = @($jobs | Where-Object { [string]$_.status -ne "success" })
    if ($pending.Count -gt 0) {
        $summary = ($pending | ForEach-Object { "$($_.id):$($_.status)" }) -join ", "
        throw "A etapa $StageName não terminou com sucesso. Pendentes/falhos: $summary"
    }
}

Write-Host "[preflight] Validando a bateria final..." -ForegroundColor Cyan
& uv run --offline python backend/preflight_bateria_final_eduardo.py
if ($LASTEXITCODE -ne 0) { throw "Preflight falhou com código $LASTEXITCODE." }

foreach ($stage in $Stages) {
    $name = [string]$stage.Name
    $config = [string]$stage.Config
    if (-not (Test-Path -LiteralPath $config)) {
        throw "Configuração da etapa não encontrada: $config"
    }

    $runFile = Get-LatestRun $config
    if ($runFile) {
        Write-Host "[$name] Retomando: $runFile" -ForegroundColor Yellow
        & uv run --offline python -m backend.train_pipeline --run-file $runFile
    } else {
        Write-Host "[$name] Criando e executando a fila: $config" -ForegroundColor Green
        & uv run --offline python -m backend.train_pipeline --config $config
        $runFile = Get-LatestRun $config
    }

    if ($LASTEXITCODE -ne 0) {
        throw "A etapa $name terminou com código $LASTEXITCODE. Execute novamente para retomar após corrigir o problema."
    }
    if (-not $runFile) { throw "Não foi possível localizar o pipeline_run da etapa $name." }

    Assert-StageComplete $runFile $name

    Write-Host "[$name] Consolidando resultados..." -ForegroundColor Cyan
    & uv run --offline python backend/report_stage_results.py $runFile
    if ($LASTEXITCODE -ne 0) { throw "A consolidação da etapa $name falhou com código $LASTEXITCODE." }

    Write-Host "[$name] concluído e empacotado. Avançando para a próxima etapa." -ForegroundColor Green
}

Write-Host "Bateria final concluída: todas as etapas foram treinadas e consolidadas." -ForegroundColor Green
