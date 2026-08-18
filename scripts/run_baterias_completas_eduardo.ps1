<#
Supervisor da sequência:
Bloco 13 -> Bloco 8 -> Bloco 9 -> Bloco 11 -> Bloco 12.

Usa os pipeline_run existentes e não reinicia jobs success.
#>
$ErrorActionPreference = "Stop"
$Root = (Resolve-Path (Join-Path $PSScriptRoot "..")).Path
Set-Location $Root
$Runs = Join-Path $Root "models\pipeline_runs"

function Get-Run([string]$Config) {
    $wanted = [IO.Path]::GetFullPath($Config)
    foreach ($file in (Get-ChildItem -LiteralPath $Runs -Filter "pipeline_run_*.json" -File | Sort-Object LastWriteTime -Descending)) {
        try {
            $state = Get-Content -Raw -LiteralPath $file.FullName | ConvertFrom-Json
            if ([IO.Path]::GetFullPath([string]$state.config_path) -eq $wanted) { return $file.FullName }
        } catch { }
    }
    return $null
}

function Get-State([string]$RunFile) {
    return Get-Content -Raw -LiteralPath $RunFile | ConvertFrom-Json
}

function Test-TrainingProcess {
    $processes = @(Get-CimInstance Win32_Process -ErrorAction SilentlyContinue |
        Where-Object { $_.CommandLine -match "backend\.train_pipeline|backend/train_pipeline" })
    return $processes.Count -gt 0
}

function Wait-ForTrainingProcessToStop {
    while (Test-TrainingProcess) {
        Write-Host "[supervisor] Há um train_pipeline ativo; aguardando 60 segundos..." -ForegroundColor Yellow
        Start-Sleep -Seconds 60
    }
}

function Invoke-Stage([string]$Name, [string]$Config) {
    $run = Get-Run $Config
    Wait-ForTrainingProcessToStop
    if ($run) {
        Write-Host "[$Name] Retomando $run" -ForegroundColor Yellow
        & uv run --offline python -m backend.train_pipeline --run-file $run
    } else {
        Write-Host "[$Name] Iniciando $Config" -ForegroundColor Green
        & uv run --offline python -m backend.train_pipeline --config $Config
        $run = Get-Run $Config
    }
    if ($LASTEXITCODE -ne 0) { throw "$Name terminou com código $LASTEXITCODE." }
    if (-not $run) { throw "Run-file não encontrado para $Name." }
    $state = Get-State $run
    $pending = @($state.jobs | Where-Object { [string]$_.status -ne "success" })
    if ($pending.Count -gt 0) {
        throw "$Name ainda está incompleto ($($pending.Count) jobs). Execute novamente para retomar."
    }
    Write-Host "[$Name] 100% concluído; consolidando." -ForegroundColor Cyan
    & uv run --offline python backend/report_stage_results.py $run
    if ($LASTEXITCODE -ne 0) { throw "Consolidação falhou para $Name." }
}

Write-Host "Sequência: Bloco 13 -> Bloco 8 -> Bloco 9 -> Bloco 11 -> Bloco 12" -ForegroundColor Cyan

# O script próprio do Bloco 13 mantém sua validação e seu empacotamento especial.
Wait-ForTrainingProcessToStop
& powershell -NoProfile -ExecutionPolicy Bypass -File (Join-Path $PSScriptRoot "run_bloco13.ps1")
if ($LASTEXITCODE -ne 0) { throw "Bloco 13 não foi concluído." }

& uv run --offline python backend/generate_novas_baterias_rgb.py
if ($LASTEXITCODE -ne 0) { throw "Falha ao preparar os Blocos 8, 9 e 11." }
& uv run --offline python backend/generate_bloco12_convnext_random.py
if ($LASTEXITCODE -ne 0) { throw "Falha ao preparar o Bloco 12." }

Invoke-Stage "Bloco 8" (Join-Path $Root "backend\novas_baterias_stages\bloco8.toml")
Invoke-Stage "Bloco 9" (Join-Path $Root "backend\novas_baterias_stages\bloco9.toml")
Invoke-Stage "Bloco 11" (Join-Path $Root "backend\novas_baterias_stages\bloco11.toml")
Invoke-Stage "Bloco 12" (Join-Path $Root "backend\bateria_final_stages\bloco12_convnext_random.toml")

Write-Host "Todas as baterias foram concluídas e consolidadas." -ForegroundColor Green
