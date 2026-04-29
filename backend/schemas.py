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
    patience: int = 5
    train_split: float = 0.8
    val_split: float = 0.1