import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
import torchvision.transforms.functional as TF
from sklearn.metrics import confusion_matrix, classification_report
from tabulate import tabulate

# -------------------------------------------------------------
# 1. REPRODUCIBILITY (SEED SABİTLEME)
# -------------------------------------------------------------
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(SEED)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"[*] Çalışma Cihazı: {device}")

# -------------------------------------------------------------
# 2. MODEL MİMARİSİ (Lightweight CNN)
# -------------------------------------------------------------
class RobustCNN(nn.Module):
    """
    3 bloklu, Batch Normalization ve Dropout içeren hafif CNN.
    CIFAR-10 için hızlı yakınsama ve kontrollü kapasite sunar.
    """
    def __init__(self, num_classes=10):
        super(RobustCNN, self).__init__()
        self.features = nn.Sequential(
            # Blok 1 (32x32 -> 16x16)
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.2),

            # Blok 2 (16x16 -> 8x8)
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.3),

            # Blok 3 (8x8 -> 4x4)
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2, 2),
            nn.Dropout2d(0.4)
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 4 * 4, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size(0), -1)
        x = self.classifier(x)
        return x

# -------------------------------------------------------------
# 3. SENTETİK DOMAIN SHIFT DÖNÜŞÜMLERİ
# -------------------------------------------------------------
class AddGaussianNoise(object):
    def __init__(self, std=0.1):
        self.std = std

    def __call__(self, tensor):
        noise = torch.randn_like(tensor) * self.std
        noisy_tensor = tensor + noise
        return torch.clamp(noisy_tensor, 0.0, 1.0)

class ApplyGaussianBlur(object):
    def __init__(self, kernel_size=3, sigma=1.0):
        self.kernel_size = kernel_size
        self.sigma = sigma

    def __call__(self, tensor):
        return TF.gaussian_blur(tensor, kernel_size=self.kernel_size, sigma=self.sigma)

class ShiftBrightnessContrast(object):
    def __init__(self, brightness_factor=1.0, contrast_factor=1.0):
        self.b_factor = brightness_factor
        self.c_factor = contrast_factor

    def __call__(self, tensor):
        out = TF.adjust_brightness(tensor, self.b_factor)
        out = TF.adjust_contrast(out, self.c_factor)
        return torch.clamp(out, 0.0, 1.0)

# -------------------------------------------------------------
# 4. VERİ SETİ HAZIRLIĞI
# -------------------------------------------------------------
# CIFAR-10 standart normalizasyon sabitleri
NORM_MEAN = (0.4914, 0.4822, 0.4465)
NORM_STD = (0.2023, 0.1994, 0.2010)

# Baseline Eğitimi: Sadece standart kırpma ve çevirme
baseline_train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

# Müdahale Eğitimi: Dağılım kaymalarına karşı zenginleştirilmiş (Robust Augmentation)
intervention_train_tf = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2),
    transforms.RandomApply([transforms.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))], p=0.4),
    transforms.ToTensor(),
    transforms.RandomApply([AddGaussianNoise(std=0.15)], p=0.4),
    transforms.Normalize(NORM_MEAN, NORM_STD)
])

# Ham test seti (Bozulmalar [0, 1] tensör üzerinde uygulanıp ardından normalize edilir)
raw_test_data = datasets.CIFAR10(root="./data", train=False, download=True, transform=transforms.ToTensor())
CLASSES = raw_test_data.classes

# -------------------------------------------------------------
# 5. EĞİTİM VE TEST FONKSİYONLARI
# -------------------------------------------------------------
def train_model(model, train_loader, epochs=15, lr=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)

    model.train()
    for epoch in range(1, epochs + 1):
        running_loss = 0.0
        correct, total = 0, 0
        for inputs, targets in train_loader:
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += targets.size(0)
            correct += predicted.eq(targets).sum().item()

        scheduler.step()
        epoch_acc = 100.0 * correct / total
        print(f"Epoch [{epoch:02d}/{epochs:02d}] - Loss: {running_loss/total:.4f} - Acc: %{epoch_acc:.2f}")
    return model

def evaluate_on_shift(model, dataset, corruption_fn=None):
    """
    Test kümesine belirtilen bozulmayı uygular, normalize eder ve doğruluk hesaplar.
    """
    model.eval()
    correct, total = 0, 0
    all_preds, all_targets = [], []
    normalize = transforms.Normalize(NORM_MEAN, NORM_STD)

    loader = DataLoader(dataset, batch_size=256, shuffle=False, num_workers=2)

    with torch.no_grad():
        for inputs, targets in loader:
            inputs = inputs.to(device)
            targets = targets.to(device)

            if corruption_fn is not None:
                # Toplu tensör üzerinde bozulmayı işlet
                inputs = corruption_fn(inputs)

            # Normalizasyon adımı
            inputs = normalize(inputs)

            outputs = model(inputs)
            _, preds = outputs.max(1)

            total += targets.size(0)
            correct += preds.eq(targets).sum().item()

            all_preds.extend(preds.cpu().numpy())
            all_targets.extend(targets.cpu().numpy())

    accuracy = 100.0 * correct / total
    return accuracy, np.array(all_preds), np.array(all_targets)

# -------------------------------------------------------------
# 6. DENEY PROTOKOLÜ VE ÇALIŞTIRMA
# -------------------------------------------------------------
if __name__ == "__main__":
    BATCH_SIZE = 128
    EPOCHS = 15

    # 1. Baseline Model Eğitimi
    print("\n" + "="*50)
    print("[1/2] Baseline Model Eğitimi Başlatılıyor...")
    print("="*50)
    train_data_base = datasets.CIFAR10(root="./data", train=True, download=True, transform=baseline_train_tf)
    train_loader_base = DataLoader(train_data_base, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    baseline_model = RobustCNN().to(device)
    baseline_model = train_model(baseline_model, train_loader_base, epochs=EPOCHS)

    # 2. Müdahale Modeli Eğitimi (Data Augmentation)
    print("\n" + "="*50)
    print("[2/2] Müdahale (Data Augmentation) Modeli Eğitimi Başlatılıyor...")
    print("="*50)
    train_data_interv = datasets.CIFAR10(root="./data", train=True, download=True, transform=intervention_train_tf)
    train_loader_interv = DataLoader(train_data_interv, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
    interv_model = RobustCNN().to(device)
    interv_model = train_model(interv_model, train_loader_interv, epochs=EPOCHS)

    # 3. Bozulma Parametreleri (3 Bozulma x 3 Şiddet)
    corruptions = {
        "Clean (Temiz Veri)": {
            "Yok": None
        },
        "Gaussian Noise": {
            "Hafif (σ=0.08)": AddGaussianNoise(std=0.08),
            "Orta (σ=0.18)": AddGaussianNoise(std=0.18),
            "Ağır (σ=0.32)": AddGaussianNoise(std=0.32)
        },
        "Gaussian Blur": {
            "Hafif (k=3, σ=0.8)": ApplyGaussianBlur(kernel_size=3, sigma=0.8),
            "Orta (k=5, σ=1.5)": ApplyGaussianBlur(kernel_size=5, sigma=1.5),
            "Ağır (k=7, σ=2.5)": ApplyGaussianBlur(kernel_size=7, sigma=2.5)
        },
        "Parlaklık & Kontrast": {
            "Hafif (b=0.8, c=0.8)": ShiftBrightnessContrast(0.8, 0.8),
            "Orta (b=0.5, c=0.6)": ShiftBrightnessContrast(0.5, 0.6),
            "Ağır (b=0.25, c=0.35)": ShiftBrightnessContrast(0.25, 0.35)
        }
    }

    # 4. Değerlendirme Döngüsü
    print("\n" + "="*50)
    print("[*] Dağılım Kayması Testleri Başlatılıyor...")
    print("="*50)

    results_table = []
    cm_storage = {}

    for c_name, severities in corruptions.items():
        for s_name, c_fn in severities.items():
            acc_base, preds_b, targets = evaluate_on_shift(baseline_model, raw_test_data, c_fn)
            acc_interv, preds_i, _ = evaluate_on_shift(interv_model, raw_test_data, c_fn)
            diff = acc_interv - acc_base

            results_table.append([c_name, s_name, f"%{acc_base:.2f}", f"%{acc_interv:.2f}", f"{diff:+.2f} puan"])

            # İnceleme için ağır gürültü tahminlerini kaydet
            if c_name == "Gaussian Noise" and "Ağır" in s_name:
                cm_storage["base"] = (targets, preds_b)
                cm_storage["interv"] = (targets, preds_i)

    print("\n" + tabulate(results_table, headers=["Bozulma Türü", "Şiddet", "Baseline Doğruluk", "Müdahale Doğruluk", "Fark"], tablefmt="github"))

    # 5. Sınıf Bazlı Hata Analizi (Ağır Gaussian Noise Altında)
    print("\n" + "="*50)
    print("[*] Sınıf Bazlı Analiz: Ağır Gaussian Noise Altında Karışıklık Matrisi")
    print("="*50)
    targets, preds_b = cm_storage["base"]
    cm_base = confusion_matrix(targets, preds_b)
    
    print("\nSınıf Doğrulukları (Baseline - Ağır Gürültü):")
    class_acc_base = cm_base.diagonal() / cm_base.sum(axis=1) * 100
    for cls_name, acc in zip(CLASSES, class_acc_base):
        print(f" - {cls_name:<12}: %{acc:.1f}")

    # En çok karışan çiftleri tespit et
    np.fill_diagonal(cm_base, 0)
    top_confusion_idx = np.unravel_index(np.argsort(cm_base.ravel())[-3:], cm_base.shape)
    print("\nEn Çok Karıştırılan Sınıf Çiftleri (True -> Predicted):")
    for r, c in zip(reversed(top_confusion_idx[0]), reversed(top_confusion_idx[1])):
        print(f" • Gerçek: {CLASSES[r]} --> Tahmin: {CLASSES[c]} ({cm_base[r, c]} örnek)")
