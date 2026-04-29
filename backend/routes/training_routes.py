"""
Training routes — REST + WebSocket para treinamento de modelos.
"""

import asyncio
import json
import os
from typing import Set

from fastapi import APIRouter, WebSocket, WebSocketDisconnect, HTTPException
from backend.schemas import TrainingConfig
from backend.services.training_service import (
    training_manager,
    scan_datasets,
    TRAINING_MODEL_CONFIGS,
    WORKSPACE_ROOT,
    MODELS_SAVE_DIR,
)

router = APIRouter()

# ──────────────────────── Diretórios permitidos para datasets ────────────────────────

ALLOWED_DATA_ROOT = os.path.join(WORKSPACE_ROOT, "data")

# ──────────────────────── WebSocket connections (thread-safe) ────────────────────────

_ws_clients: Set[WebSocket] = set()
_ws_lock = asyncio.Lock()


async def _broadcast(message: dict):
    """Envia mensagem para todos os clientes WebSocket conectados."""
    data = json.dumps(message)
    async with _ws_lock:
        dead = []
        for ws in _ws_clients:
            try:
                await ws.send_text(data)
            except Exception:
                dead.append(ws)
        for ws in dead:
            _ws_clients.discard(ws)


@router.websocket("/ws")
async def training_ws(websocket: WebSocket):
    """WebSocket para receber progresso do treinamento em tempo real."""
    await websocket.accept()
    async with _ws_lock:
        _ws_clients.add(websocket)
    try:
        while True:
            # Manter conexão aberta; o client pode enviar pings
            await websocket.receive_text()
    except WebSocketDisconnect:
        pass
    finally:
        async with _ws_lock:
            _ws_clients.discard(websocket)


# ──────────────────────── REST endpoints ────────────────────────


@router.get("/models")
async def list_models():
    """Lista os modelos disponíveis para treinamento."""
    models = []
    for key, cfg in TRAINING_MODEL_CONFIGS.items():
        if cfg.get("hidden"):
            continue
        models.append({
            "id": key,
            "name": key,
            "input_size": cfg["input_size"],
            "default_batch": cfg["default_batch"],
            "cpu_batch": cfg["cpu_batch"],
        })
    return models


@router.get("/datasets")
async def list_datasets():
    """Lista datasets disponíveis com labels e contagem de imagens."""
    datasets = scan_datasets()
    # Adiciona as classes detectadas para cada dataset
    for ds in datasets:
        ds["classes"] = sorted(ds["labels"].keys())
    return datasets


@router.get("/status")
async def get_status():
    """Retorna estado atual do treinamento."""
    return {
        "status": training_manager.status,
        "last_result": training_manager.last_result,
    }


def _validate_data_path(data_path: str) -> str:
    """Valida que data_path está dentro do diretório permitido (data/).

    Previne Path Traversal — rejeita caminhos como '/etc' ou '../../'.
    """
    resolved = os.path.realpath(data_path)
    allowed = os.path.realpath(ALLOWED_DATA_ROOT)

    if not resolved.startswith(allowed + os.sep) and resolved != allowed:
        raise HTTPException(
            status_code=400,
            detail=f"Caminho não permitido. O dataset deve estar dentro de '{ALLOWED_DATA_ROOT}'.",
        )

    if not os.path.isdir(resolved):
        raise HTTPException(
            status_code=400,
            detail=f"Diretório não encontrado: '{data_path}'.",
        )

    return resolved


def _validate_splits(train_split: float, val_split: float):
    """Valida que train_split + val_split < 1.0 (sobra espaço para test)."""
    total = train_split + val_split
    if total >= 0.95:
        raise HTTPException(
            status_code=400,
            detail=(
                f"train_split ({train_split}) + val_split ({val_split}) = {total:.2f}. "
                f"A soma deve ser < 0.95 para garantir um split de teste válido."
            ),
        )


@router.post("/start")
async def start_training(config: TrainingConfig):
    """Inicia treinamento com a configuração fornecida."""
    if training_manager.is_training:
        raise HTTPException(status_code=409, detail="Treinamento já em andamento.")

    # Validações de segurança
    validated_path = _validate_data_path(config.data_path)
    _validate_splits(config.train_split, config.val_split)

    # Fix 3: get_running_loop() em vez de get_event_loop()
    loop = asyncio.get_running_loop()

    def progress_callback(message: dict):
        """Callback chamado pela thread de treinamento — enfileira broadcast."""
        asyncio.run_coroutine_threadsafe(_broadcast(message), loop)

    training_config = {
        "model_name": config.model_name,
        "data_path": validated_path,
        "batch_size": config.batch_size,
        "num_epochs": config.num_epochs,
        "learning_rate": config.learning_rate,
        "patience": config.patience,
        "train_split": config.train_split,
        "val_split": config.val_split,
    }

    try:
        training_manager.start(training_config, progress_callback)
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc

    return {"status": "started", "config": training_config}


@router.post("/cancel")
async def cancel_training():
    """Cancela o treinamento em andamento."""
    if not training_manager.is_training:
        raise HTTPException(status_code=400, detail="Nenhum treinamento em andamento.")
    training_manager.cancel()
    return {"status": "cancelling"}

@router.post("/pause")
async def pause_training():
    """Pausa o treinamento em andamento."""
    if not training_manager.is_training:
        raise HTTPException(status_code=400, detail="Nenhum treinamento.")
    training_manager.pause()
    return {"status": "paused"}

@router.post("/resume")
async def resume_training():
    """Retoma o treinamento pausado."""
    if not training_manager.is_training:
        raise HTTPException(status_code=400, detail="Nenhum treinamento.")
    training_manager.resume()
    return {"status": "resumed"}

@router.post("/stop_early")
async def stop_early_training():
    """Finaliza o treinamente antecipadamente."""
    if not training_manager.is_training:
        raise HTTPException(status_code=400, detail="Nenhum treinamento.")
    training_manager.stop_early()
    return {"status": "stopping_early"}

@router.get("/history")
async def get_training_history():
    """Retorna o log de métricas e histórico de modelos já treinados"""
    history_path = os.path.join(MODELS_SAVE_DIR, "training_history.json")
    if not os.path.exists(history_path):
        return []
    
    try:
        with open(history_path, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as exc:
        raise HTTPException(status_code=500, detail=f"Erro ao ler histórico: {exc}")

@router.get("/model_versions/{model_name}")
async def get_model_versions(model_name: str):
    """Retorna a lista de arquivos .pth treinados disponíveis."""
    if not os.path.exists(MODELS_SAVE_DIR):
        return []
    
    prefix = f"soybean_model_{model_name.lower()}"
    files = []
    for f in os.listdir(MODELS_SAVE_DIR):
        if f.startswith(prefix) and f.endswith(".pth"):
            files.append(f)
            
    files.sort(reverse=True)
    return files
