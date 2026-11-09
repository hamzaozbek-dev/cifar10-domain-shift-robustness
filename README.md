# cifar10-domain-shift-robustness
AIRLAB: AI RESEARCH TEAM – ALIM GÖREVİ 2026 BÖLÜM II
# Mitigating Domain Shift on CIFAR-10 via Data Augmentation

Bu çalışma, görüntü sınıflandırma modellerinin test verisinde karşılaştığı sentetik dağılım kaymalarına (domain shift) karşı dayanıklılığını ölçmeyi ve hedeflenmiş veri artırma (data augmentation) yöntemiyle bu performans kaybını minimize etmeyi amaçlar.

---

## 1. Metrik Seçimi Gerekçesi
CIFAR-10 test kümesi sınıf başına tam olarak 1.000 görüntü içeren dengeli (balanced) bir veri kümesi olduğundan, modellerin genel genelleme kapasitesini doğrudan ve tarafsız biçimde yansıtan metrik **Top-1 Doğruluk (Accuracy)** olarak belirlenmiş; sınıflar arası bozulma asimetrisini tespit etmek için **Sınıf Bazlı Doğruluk (Class-wise Accuracy)** ve **Hata Matrisi (Confusion Matrix)** ile desteklenmiştir.

---

## 2. Kurulum ve Çalıştırma

Gerekli bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
