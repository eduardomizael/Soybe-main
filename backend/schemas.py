# O arquivo schemas.py defini os modelos de dados usados na API

from pydantic import BaseModel
from typing import List

# BaseModel: modelo base para requisições e respostas
# Definição do contrato da API

# Aqui definimos o formato da requisição que a API espera receber do frontend
class InferenceRequest(BaseModel):
    model_name: str
    images: List[str]
    
# Aqui definimos o formato da resposta que a API vai enviar para o frontend
class InferenceResponse(BaseModel):
    filename: str
    classification: str
    confidence: float
    model_used: str


# Configuração de treinamento enviada pelo frontend
class TrainingConfig(BaseModel):
    model_name: str
    data_path: str
    batch_size: int = 16
    num_epochs: int = 20
    learning_rate: float = 1e-4
    fine_tune_learning_rate: float | None = None
    patience: int = 5
    early_stopping: bool = True
    split_strategy: str = "random"
    checkpoint_metric: str = "val_loss"
    sampler_strategy: str = "shuffle"
    loss_name: str = "cross_entropy"
    class_weight_strategy: str = "sqrt_inverse"
    label_smoothing: float = 0.0
    focal_gamma: float = 1.5
    effective_number_beta: float = 0.999
    augmentation_profile: str = "standard"
    train_split: float = 0.8
    val_split: float = 0.1
    seed: int = 42
    optimizer_name: str = "AdamW"
    weight_decay: float = 1e-4
    scheduler_factor: float = 0.5
    scheduler_patience: int = 2
    scheduler_min_lr: float = 1e-6
    accumulation_steps: int = 1
    freeze_backbone_epochs: int = 0
