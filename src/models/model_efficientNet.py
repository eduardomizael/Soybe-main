import os
import time
import torch
from torch import nn, optim
from torch.utils.data import DataLoader, random_split
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import torchvision
import torchvision.transforms as T
from torchvision.datasets import ImageFolder
from torchvision.models import EfficientNet_B7_Weights
from torchmetrics.classification import Accuracy, Precision, Recall, F1Score
from tqdm import tqdm
import numpy as np

try:
    from torch.utils.tensorboard import SummaryWriter
except ModuleNotFoundError:
    SummaryWriter = None


def configure_runtime() -> tuple[torch.device, int, bool, bool, int]:
    """Configura device e paralelismo para CPU/GPU com foco em estabilidade de RAM."""
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

    # Em CPU com 16GB, muitos workers aumentam muito o uso de RAM.
    num_workers = min(4, max(1, cpu_threads // 4)) if device.type == "cpu" else min(8, cpu_threads)
    pin_memory = device.type == "cuda"
    persistent_workers = device.type == "cuda" and num_workers > 0

    # Prefetch baixo para reduzir picos de memória dos workers.
    prefetch_factor = 1 if num_workers > 0 else 2

    print(f"DataLoader workers: {num_workers}")
    print(f"prefetch_factor: {prefetch_factor}")

    return device, num_workers, pin_memory, persistent_workers, prefetch_factor

# # Implementando metricas da matriz de confusão
# import sklearn
# from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay

# Classificação Supervisionada de Imagens de Soja usando CNN

# TensorBoard opcional
writer = SummaryWriter() if SummaryWriter is not None else None

# Hiperparâmetros
batch_size = 8
num_epochs = 20
num_class = 5 # Broken, Immature, Intact, Skin-damaged, Spotted
patience = 5
best_val_loss = float('inf')
epochs_no_improve = 0
learning_rate = 1e-4
PATH = "./models/soybean_model_efficientnet_b7.pth"
data_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../data/processed"))
device, num_workers, pin_memory, persistent_workers, prefetch_factor = configure_runtime()

# Em CPU integrada/16GB, B7 com entrada 600x600 exige batch menor para não estourar RAM.
if device.type == "cpu":
    batch_size = 1
    print("CPU mode: batch_size ajustado para 1 para evitar crash de memória.")

class TransformSubset(torch.utils.data.Dataset):
    def __init__(self, subset, transform=None):
        self.subset = subset
        self.transform = transform
        
    def __getitem__(self, index):
        x, y = self.subset[index]
        if self.transform:
            x = self.transform(x)
        return x, y
        
    def __len__(self):
        return len(self.subset)

train_transforms = T.Compose([
    T.RandomResizedCrop(600, scale=(0.8, 1.0)),
    T.RandomHorizontalFlip(),
    T.RandomRotation(15),
    T.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

val_transforms = T.Compose([
    T.Resize((600, 600)),
    T.ToTensor(),
    T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
])

image_dataset = ImageFolder(root=data_root, transform=None)

if len(image_dataset.classes) != num_class:
    raise ValueError(
        f"num_class={num_class}, mas o dataset possui {len(image_dataset.classes)} classes: {image_dataset.classes}"
    )

train_size = int(0.8 * len(image_dataset))
val_size = int(0.1 * len(image_dataset))
test_size = len(image_dataset) - train_size - val_size

train_dataset_raw, val_dataset_raw, test_dataset_raw = random_split(image_dataset, [train_size, val_size, test_size])

train_dataset = TransformSubset(train_dataset_raw, transform=train_transforms)
val_dataset = TransformSubset(val_dataset_raw, transform=val_transforms)
test_dataset = TransformSubset(test_dataset_raw, transform=val_transforms)

train_indices = train_dataset_raw.indices
targets = [image_dataset.targets[i] for i in train_indices]
class_counts = np.bincount(targets, minlength=num_class)
weights = 1.0 / np.sqrt(class_counts + 1e-8)
weights = (weights / np.sum(weights)) * num_class
class_weights = torch.FloatTensor(weights).to(device)

# Loader -> Remover gargalo da CPU em preparar os dados
train_loader = DataLoader(
    train_dataset,
    batch_size=batch_size,
    shuffle=True,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
) # Embaralha os dados
val_loader = DataLoader(
    val_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
) # Não embaralha os dados
test_loader = DataLoader(
    test_dataset,
    batch_size=batch_size,
    shuffle=False,
    num_workers=num_workers,
    pin_memory=pin_memory,
    persistent_workers=persistent_workers,
    prefetch_factor=prefetch_factor if num_workers > 0 else None,
) # Não embaralha os dados

# Modelo EfficientNet-B7 pré-treinado
weights = EfficientNet_B7_Weights.IMAGENET1K_V1
model = torchvision.models.efficientnet_b7(weights=weights)
num_features = model.classifier[1].in_features
model.classifier[1] = nn.Linear(num_features, num_class)  # Ajusta a camada final
model = model.to(device)

criterion = nn.CrossEntropyLoss(weight=class_weights)
optimizer = optim.Adam(model.parameters(), lr=learning_rate)

# Treinamento do Modelo
def train_model(num_epochs):
    best_val_loss = float('inf')
    epochs_no_improve = 0
    use_amp = device.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    
    history = {'train_loss': [], 'val_loss': []}
    min_delta = 0.001
    
    best_path = PATH.replace('.pth', '_best.pth')
    last_path = PATH.replace('.pth', '_last.pth')

    try:
        for epoch in range(num_epochs):
            model.train()
            running_loss = 0.0

            for data, targets in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
                data = data.to(device, non_blocking=pin_memory)
                targets = targets.to(device, non_blocking=pin_memory)
                optimizer.zero_grad()
                with torch.autocast(device_type=device.type, enabled=use_amp):
                    outputs = model(data)
                    loss = criterion(outputs, targets)

                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                running_loss += loss.item()

            epoch_train_loss = running_loss / len(train_loader)

            # Validação
            model.eval()
            val_loss = 0.0
            with torch.no_grad():
                for data, targets in val_loader:
                    data = data.to(device, non_blocking=pin_memory)
                    targets = targets.to(device, non_blocking=pin_memory)
                    outputs = model(data)
                    loss = criterion(outputs, targets)
                    val_loss += loss.item()

            epoch_val_loss = val_loss / len(val_loader)
            
            history['train_loss'].append(epoch_train_loss)
            history['val_loss'].append(epoch_val_loss)

            # Logs para TensorBoard
            if writer is not None:
                writer.add_scalar("Loss/Treino", epoch_train_loss, epoch)
                writer.add_scalar("Loss/Validação", epoch_val_loss, epoch)
                
            loss_diff = epoch_train_loss - epoch_val_loss
            print(f"Epoch {epoch+1}, Train Loss: {epoch_train_loss:.4f}, Val Loss: {epoch_val_loss:.4f} (Diff: {loss_diff:.4f})")

            # Salvar último modelo (checkpoint de recuperação)
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'val_loss': epoch_val_loss,
                'best_val_loss': best_val_loss
            }, last_path)

            # Early stopping check
            if epoch_val_loss < (best_val_loss - min_delta):
                best_val_loss = epoch_val_loss
                epochs_no_improve = 0
                
                # Salvar melhor modelo
                torch.save({
                    'epoch': epoch + 1,
                    'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                    'best_val_loss': best_val_loss
                }, best_path)
                print(f"⭐ Novo melhor modelo salvo! Val Loss reduziu para {best_val_loss:.4f}")
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    print(f"🛑 Early stopping ativado na época {epoch+1}. Sem melhorias estruturais de {min_delta} nas ultimas {patience} épocas.")
                    break

    except KeyboardInterrupt:
        print("\n⚠️ Treinamento interrompido pelo usuário (Ctrl+C)!")
        interrupted_path = PATH.replace('.pth', '_interrupted.pth')
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'best_val_loss': best_val_loss,
            'reason': 'keyboard_interrupt'
        }, interrupted_path)
        print(f"💾 Último estado salvo com segurança em: {interrupted_path}")

    finally:
        if writer is not None:
            writer.close()
            
    return history

if __name__ == "__main__":
    start = time.time()
    train_model(num_epochs)
    end = time.time()
    print(end - start)

    # Melhores pesos salvos
    best_path = PATH.replace('.pth', '_best.pth')
    if os.path.exists(best_path):
        checkpoint = torch.load(best_path, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Pesos do melhor modelo (Epoca {checkpoint.get('epoch', '?')}) carregados para avaliação.")
    else:
        print(f"Aviso: {best_path} não encontrado, avaliando modelo no estado atual.")
    model.eval()

    # Avaliação do Modelo -> Garante verificar as métricas após o treinamento.
    acc = Accuracy(task="multiclass", num_classes=num_class, average="macro").to(device)  # macro para acurácia agregada
    precision = Precision(task="multiclass", num_classes=num_class, average=None).to(device)
    recall = Recall(task="multiclass", num_classes=num_class, average=None).to(device)
    f1 = F1Score(task="multiclass", num_classes=num_class, average=None).to(device)

    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device, non_blocking=pin_memory)
            labels = labels.to(device, non_blocking=pin_memory)
            outputs = model(images)
            _, preds = torch.max(outputs, 1)

            # Atualiza as métricas
            acc.update(preds, labels)
            precision.update(preds, labels)
            recall.update(preds, labels)
            f1.update(preds, labels)

            # Matriz de Confusão
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())

    # computa métricas
    accuracy = acc.compute().item()
    prec_per_class = precision.compute().cpu().numpy()  # vetor de tamanho num_class
    rec_per_class = recall.compute().cpu().numpy()
    f1_per_class = f1.compute().cpu().numpy()

    print(f"Acurácia (macro): {accuracy:.4f}\n")

    class_names = image_dataset.classes if hasattr(image_dataset, "classes") else [f"class_{i}" for i in range(num_class)]

    for i, cname in enumerate(class_names):
        print(f"Classe: {cname:15s}  Precision: {prec_per_class[i]:.4f}  Recall: {rec_per_class[i]:.4f}  F1: {f1_per_class[i]:.4f}")

    print("\n\nClassification Report (sklearn):\n")
    print(classification_report(all_labels, all_preds, target_names=class_names, digits=4))

    cm = confusion_matrix(all_labels, all_preds)
    print("\nConfusion Matrix (raw counts):\n", cm)

    cm_norm = cm.astype('float') / (cm.sum(axis=1)[:, np.newaxis] + 1e-8)
    print("\nConfusion Matrix (normalized by true class / recall):\n", np.round(cm_norm, 3))
    print("Conteúdo salvo...")
    torch.save(model.state_dict(), PATH)
