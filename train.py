import torch
from torch import nn
from torchvision.datasets import ImageFolder
import torchvision.transforms as T
import wandb # Aggiunto per la visualizzazione

# ==========================================
# 1. DEFINIZIONE DEL MODELLO
# ==========================================
class CustomNet(nn.Module):
    def __init__(self):
        super(CustomNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.conv3 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        
        self.pool = nn.MaxPool2d(2, 2)
        
        self.batch_norm1 = nn.BatchNorm2d(64)
        self.batch_norm2 = nn.BatchNorm2d(128)
        self.batch_norm3 = nn.BatchNorm2d(256)
        
        self.adaptive_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc1 = nn.Linear(256, 200)

    def forward(self, x):
        x = self.pool(torch.relu(self.batch_norm1(self.conv1(x))))
        x = self.pool(torch.relu(self.batch_norm2((self.conv2(x)))))
        x = self.pool(torch.relu(self.batch_norm3(self.conv3(x))))
        x = self.adaptive_pool(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        return x

# ==========================================
# 2. FUNZIONI DI TRAINING E VALIDATION
# ==========================================
def train(epoch, model, train_loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for batch_idx, (inputs, targets) in enumerate(train_loader):
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        _, predicted = outputs.max(1)
        total += targets.size(0)
        correct += predicted.eq(targets).sum().item()

    train_loss = running_loss / len(train_loader)
    train_accuracy = 100. * correct / total
    print(f'Train Epoch: {epoch} Loss: {train_loss:.6f} Acc: {train_accuracy:.2f}%')
    
    # Log su WandB per i grafici
    wandb.log({"train_loss": train_loss, "train_acc": train_accuracy, "epoch": epoch})

def validate(epoch, model, val_loader, criterion, device):
    model.eval()
    val_loss = 0.0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(val_loader):
            inputs, targets = inputs.to(device), targets.to(device)

            outputs = model(inputs)
            loss = criterion(outputs, targets)
            val_loss += loss.item()

            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

    val_loss = val_loss / len(val_loader)
    val_accuracy = 100. * correct / total

    print(f'Validation Loss: {val_loss:.6f} Acc: {val_accuracy:.2f}%')
    
    # Log su WandB per i grafici
    wandb.log({"val_loss": val_loss, "val_acc": val_accuracy, "epoch": epoch})
    
    return val_accuracy

# ==========================================
# 3. BLOCCO DI ESECUZIONE PRINCIPALE
# ==========================================
if __name__ == '__main__':
    # Setup del device (GPU se disponibile, altrimenti CPU)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Sto usando il device: {device}")

    # Inizializza WandB per i grafici
    wandb.init(project="faimdl-lab3", name="custom-net-run")

    # PREPARAZIONE DATI
    transform = T.Compose([
        T.Resize((64, 64)),
        T.ToTensor(),
        T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # Assicurati che i path siano corretti per Colab
    tiny_imagenet_dataset_train = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/train', transform=transform)
    tiny_imagenet_dataset_val = ImageFolder(root='tiny-imagenet/tiny-imagenet-200/val', transform=transform)

    train_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_train, batch_size=64, shuffle=True, num_workers=2)
    val_loader = torch.utils.data.DataLoader(tiny_imagenet_dataset_val, batch_size=64, shuffle=False, num_workers=2)

    # SETUP MODELLO, LOSS E OTTIMIZZATORE
    model = CustomNet().to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

    # CICLO DI TRAINING
    best_acc = 0
    num_epochs = 50
    
    for epoch in range(1, num_epochs + 1):
        train(epoch, model, train_loader, criterion, optimizer, device)
        val_accuracy = validate(epoch, model, val_loader, criterion, device)
        best_acc = max(best_acc, val_accuracy)

    print(f'Best validation accuracy: {best_acc:.2f}%')
    
    # Chiude WandB alla fine
    wandb.finish()