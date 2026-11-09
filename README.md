# Mitigating Domain Shift on CIFAR-10 via Data Augmentation
*AIRLAB: AI Research Team – Alım Görevi 2026 | Bölüm II*

Bu çalışma, görüntü sınıflandırma modellerinin test verisinde karşılaştığı sentetik dağılım kaymalarına (domain shift) karşı dayanıklılığını ölçmeyi ve hedeflenmiş veri artırma (data augmentation) yöntemiyle bu performans kaybını minimize etmeyi amaçlar.

---

## 1. Metrik Seçimi Gerekçesi
CIFAR-10 test kümesi sınıf başına tam olarak 1.000 görüntü içeren dengeli (balanced) bir veri kümesi olduğundan, modellerin genel genelleme kapasitesini doğrudan ve tarafsız biçimde yansıtan metrik **Top-1 Doğruluk (Accuracy)** olarak belirlenmiş; sınıflar arası bozulma asimetrisini tespit etmek için **Sınıf Bazlı Doğruluk (Class-wise Accuracy)** ve **Hata Matrisi (Confusion Matrix)** ile desteklenmiştir.

---

## 2. Kurulum ve Çalıştırma

Gerekli bağımlılıkları yükleyin:
```bash
pip install -r requirements.txt
```

Baseline eğitimi, müdahale modeli eğitimi ve dağılım kayması testlerini çalıştırmak için:
```bash
python run_experiment.py
```
> **Not:** Sonuçların birebir tekrarlanabilmesi (reproducibility) için rastgelelik tohumu (`seed=42`) olarak sabitlenmiştir.

---

## 3. Deneysel Karşılaştırma Sonuçları

Sentetik bozulmalar test setine üç farklı seviyede uygulanmış, baseline ve müdahale modeli karşılaştırılmıştır:

| Bozulma Türü | Şiddet Seviyesi | Baseline Doğruluk | Müdahale (Data Aug.) Doğruluk | Net Değişim |
| :--- | :--- | :--- | :--- | :--- |
| **Temiz Veri (Clean Test)** | Standart | **%82.40** | %80.95 | -1.45 puan |
| **Gaussian Noise** | Hafif ($\sigma=0.08$)<br>Orta ($\sigma=0.18$)<br>Ağır ($\sigma=0.32$) | %64.12<br>%39.45<br>%21.30 | %76.80<br>%66.25<br>%51.10 | **+12.68 puan**<br>**+26.80 puan**<br>**+29.80 puan** |
| **Gaussian Blur** | Hafif ($k=3, \sigma=0.8$)<br>Orta ($k=5, \sigma=1.5$)<br>Ağır ($k=7, \sigma=2.5$) | %73.80<br>%54.20<br>%31.50 | %77.40<br>%69.15<br>%55.40 | **+3.60 puan**<br>**+14.95 puan**<br>**+23.90 puan** |
| **Parlaklık & Kontrast** | Hafif ($b=0.8, c=0.8$)<br>Orta ($b=0.5, c=0.6$)<br>Ağır ($b=0.25, c=0.35$) | %77.60<br>%61.10<br>%34.75 | %78.20<br>%71.85<br>%58.30 | **+0.60 puan**<br>**+10.75 puan**<br>**+23.55 puan** |

---

## 4. Temel Bulgular ve Hata Analizi
* **Doku vs. Geometri:** Ağır gürültü altında belirgin küresel hatlara sahip araç sınıfları (*ship* %72.4, *automobile* %68.1) güçlü kalırken; yüksek uzamsal frekanslı doku detaylarına dayanan canlı sınıfları (*cat* %18.2, *dog* %22.4) ciddi doğruluk kaybı yaşamış ve birbirine karışmıştır.
* **Robustness vs. Accuracy Trade-off:** Müdahale modeli ağır gürültüde **+29.80 puanlık** bir direnç kazanırken, temiz test verisinde **1.45 puanlık** marjinal bir düşüş göstermiştir (%82.40 $\rightarrow$ %80.95). Bu durum, modelin yüksek frekanslı ince piksel detayları yerine daha genel formlara odaklanmaya zorlanmasının beklenen bir sonucudur.
