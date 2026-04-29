import os
import cv2
import numpy as np
import torch
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from torchvision import models
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "../"))
WORKSPACE_ROOT = os.path.abspath(os.path.join(PROJECT_ROOT, "../"))

# Construiremos um dict global padrão para legados
classes = {
    0: "Broken soybeans",
    1: "Immature soybeans",
    2: "Intact soybeans",
    3: "Skin-damaged soybeans",
    4: "Spotted soybeans"
}
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

if device.type == "cuda":
    torch.backends.cudnn.benchmark = True
    print(f"Usando CUDA no backend: {torch.cuda.get_device_name(0)}")
else:
    cpu_threads = os.cpu_count() or 1
    torch.set_num_threads(cpu_threads)
    torch.set_num_interop_threads(max(1, cpu_threads // 2))
    print(f"Usando CPU no backend: {cpu_threads} threads")

IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

MODEL_CONFIGS = {
    "EfficientNetB0": {
        "builder": models.efficientnet_b0,
        "input_size": 224,
        "classifier_attr": "classifier",  # model.classifier[1]
        "weight_candidates": [
            os.path.join(PROJECT_ROOT, "network/models/efficientnet.pth"),
            os.path.join(PROJECT_ROOT, "models/efficientnet_b0.pth"),
            os.path.join(WORKSPACE_ROOT, "models/soybean_model_efficientnetb0.pth"),
        ],
    },
    "EfficientNetB7": {
        "builder": models.efficientnet_b7,
        "input_size": 600,
        "classifier_attr": "classifier",  # model.classifier[1]
        "weight_candidates": [
            os.path.join(PROJECT_ROOT, "network/models/efficientnet_b7.pth"),
            os.path.join(WORKSPACE_ROOT, "models/soybean_model_efficientnet_b7.pth"),
            os.path.join(WORKSPACE_ROOT, "models/soybean_model_efficientnetb7.pth"),
        ],
    },
    "ResNet50": {
        "builder": models.resnet50,
        "input_size": 224,
        "classifier_attr": "fc",  # model.fc (camada única)
        "weight_candidates": [
            os.path.join(PROJECT_ROOT, "network/models/resnet50.pth"),
            os.path.join(WORKSPACE_ROOT, "models/soybean_model_resnet50.pth"),
        ],
    },
    "MobileNetV3": {
        "builder": models.mobilenet_v3_large,
        "input_size": 224,
        "classifier_attr": "classifier",  # model.classifier[-1]
        "weight_candidates": [
            os.path.join(PROJECT_ROOT, "network/models/mobilenet_v3.pth"),
            os.path.join(WORKSPACE_ROOT, "models/soybean_model_mobilenet_v3.pth"),
            os.path.join(WORKSPACE_ROOT, "models/soybean_model_mobilenetv3.pth"),
        ],
    },
}

_MODEL_CACHE: dict[str, tuple[torch.nn.Module, transforms.Compose, dict[int, str]]] = {}


def _build_transform(input_size: int) -> transforms.Compose:
    return transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
    ])


def _resolve_weight_path(weight_candidates: list[str]) -> str:
    for path in weight_candidates:
        if os.path.exists(path):
            return path
    raise FileNotFoundError(f"Nenhum peso encontrado. Caminhos testados: {weight_candidates}")


def _replace_final_layer(model: torch.nn.Module, classifier_attr: str, num_out: int):
    """Substitui a última camada do modelo para o número de classes do projeto.

    - ResNet: model.fc (nn.Linear direto)
    - EfficientNet / MobileNet: model.classifier[1] ou model.classifier[-1] (nn.Sequential)
    """
    if classifier_attr == "fc":
        in_features = model.fc.in_features
        model.fc = torch.nn.Linear(in_features, num_out)
    else:
        # EfficientNet → classifier[1], MobileNet → classifier[-1]
        layer = getattr(model, classifier_attr)
        last_idx = -1
        # Percorre para encontrar a última Linear
        for idx in range(len(layer) - 1, -1, -1):
            if isinstance(layer[idx], torch.nn.Linear):
                last_idx = idx
                break
        in_features = layer[last_idx].in_features
        layer[last_idx] = torch.nn.Linear(in_features, num_out)


def _load_model(model_name: str, weight_filename: str = None) -> tuple[torch.nn.Module, transforms.Compose, dict[int, str]]:
    config = MODEL_CONFIGS[model_name]
    
    if weight_filename:
        path = os.path.join(WORKSPACE_ROOT, "models", weight_filename)
        if not os.path.exists(path):
            raise FileNotFoundError(f"Arquivo especificado não encontrado: {path}")
    else:
        path = _resolve_weight_path(config["weight_candidates"])
        
    print(f"Carregando pesos de {path} para {model_name}...")
    
    checkpoint = torch.load(path, map_location=device, weights_only=False)
    
    local_num_classes = 5
    local_classes = classes

    if isinstance(checkpoint, dict) and "state_dict" in checkpoint:
        state_dict = checkpoint["state_dict"]
        if "num_classes" in checkpoint:
            local_num_classes = checkpoint["num_classes"]
        if "class_names" in checkpoint:
            local_classes = {i: name for i, name in enumerate(checkpoint["class_names"])}
    else:
        state_dict = checkpoint
        # Infer shape from weights to support older raw checkpoints
        if config["classifier_attr"] == "fc" and "fc.weight" in state_dict:
            local_num_classes = state_dict["fc.weight"].shape[0]
        elif config["classifier_attr"] == "classifier":
            for k in reversed(list(state_dict.keys())):
                if "classifier" in k and "weight" in k:
                    local_num_classes = state_dict[k].shape[0]
                    break

    model = config["builder"](weights=None)
    _replace_final_layer(model, config["classifier_attr"], local_num_classes)

    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()

    transform = _build_transform(config["input_size"])
    return model, transform, local_classes


def get_model_and_transform(model_name: str, weight_filename: str = None) -> tuple[torch.nn.Module, transforms.Compose, dict[int, str]]:
    if model_name not in MODEL_CONFIGS:
        modelos = ", ".join(MODEL_CONFIGS.keys())
        raise ValueError(f"Modelo {model_name} não suportado. Opções: {modelos}")

    cache_key = f"{model_name}_{weight_filename}" if weight_filename else model_name

    if cache_key not in _MODEL_CACHE:
        _MODEL_CACHE[cache_key] = _load_model(model_name, weight_filename)

    return _MODEL_CACHE[cache_key]

# Função chamada pelo Inference_Service (endpoint)
def classify_image(image_bytes: bytes, model_name: str, weight_filename: str = None) -> dict:
    """
    Recebe bytes da imagem e retorna a classificação
    Bytes -> Numpy array -> Imagem PIL
    """
    model, transform, model_classes = get_model_and_transform(model_name, weight_filename)

    nparr = np.frombuffer(image_bytes, np.uint8)
    img_cv = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

    # Erro de decodificação
    if img_cv is None:
        raise ValueError("Não foi possível decodificar a imagem.")
    
    # Converter OpenCV (BGR) para PIL (RGB)
    img_rgb = cv2.cvtColor(img_cv, cv2.COLOR_BGR2RGB)
    image = Image.fromarray(img_rgb)

    # Pré-processamento
    input_tensor = transform(image).unsqueeze(0).to(device, non_blocking=device.type == "cuda")

    # Inferência
    with torch.inference_mode():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0][predicted_class].item()

    return {
        "predicted_class": predicted_class,
        "confidence": float(confidence),
        "class_name": model_classes.get(predicted_class, f"Classe {predicted_class}")
    }