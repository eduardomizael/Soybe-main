import os
import torch
from torchvision import models
import torchvision.transforms as transforms
from PIL import Image
import torch.nn.functional as F
from torchvision.models import EfficientNet_B7_Weights


def configure_runtime() -> tuple[torch.device, bool]:
    """Configura device e threads para inferência."""
    cpu_threads = os.cpu_count() or 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.set_float32_matmul_precision("high")
        print(f"Usando CUDA: {torch.cuda.get_device_name(0)}")
    else:
        torch.set_num_threads(cpu_threads)
        torch.set_num_interop_threads(max(1, cpu_threads // 2))
        print(f"CUDA indisponivel. Usando CPU com {cpu_threads} threads logicas.")

    return device, device.type == "cuda"

# Classes
classes = {
    0: "Broken soybeans",
    1: "Immature soybeans",
    2: "Intact soybeans",
    3: "Skin-damaged soybeans",
    4: "Spotted soybeans"
}

# Configurações
num_classes = 5
device, pin_memory = configure_runtime()

# Carregando EfficientNet-B0 pré-treinado
# model = models.efficientnet_b0(pretrained=False)  # False se você tiver pesos salvos

# Carregando EfficientNet-B7 pré-treinado
weights = EfficientNet_B7_Weights.IMAGENET1K_V1
model = models.efficientnet_b7(weights=weights)  # Carrega pesos pré-treinados do ImageNet

# Substitui a última camada fully connected pelo número de classes
model.classifier[1] = torch.nn.Linear(model.classifier[1].in_features, num_classes)
PATH = "./models/soybean_model_efficientnet_b7.pth"

# Carrega pesos
if os.path.exists(PATH):
    print(f"Arquivo de pesos encontrado. Carregando {PATH}...")
    model.load_state_dict(torch.load(PATH, map_location=device))
else:
    raise FileNotFoundError(f"Arquivo de pesos não encontrado: {PATH}")

model = model.to(device)
model.eval()

# Transformação da imagem (pré-processamento)
data_transforms = transforms.Compose([
    transforms.Resize((600, 600)),
    transforms.ToTensor(),
    transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

# Função de inferência
def predict_single_image(image_path):
    image = Image.open(image_path).convert('RGB')
    input_tensor = data_transforms(image).unsqueeze(0).to(device, non_blocking=pin_memory)

    with torch.no_grad():
        output = model(input_tensor)
        predicted_class = torch.argmax(output, dim=1).item()
        probabilities = F.softmax(output, dim=1)
        confidence = probabilities[0][predicted_class].item()

    return predicted_class, confidence

# Teste
if __name__ == "__main__":
    image_path = "./src/models/test/immature_test.jpg"

    if not os.path.exists(image_path):
        print(f"Arquivo {image_path} não encontrado!")
    else:
        predicted_class, confidence = predict_single_image(image_path)
        print(f"Classe predita: {classes[predicted_class]}, Confiança: {confidence:.4f}")
