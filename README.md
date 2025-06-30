# **Ürün Başlığından Açıklama ve Kategori Tahmini Projesi**

Bu proje, e-ticaret ürün başlıklarından hem kategori tahmini hem de açıklama üretimi yapmak için makine öğrenmesi ve üretken yapay zeka tekniklerini kullanan kapsamlı bir çalışmadır.

## 📋 **Proje Özeti**

Proje iki ana bölümden oluşmaktadır:

1. **Gözetimli Öğrenme ile Kategori Tahmini**: Ürün başlıklarından kategori sınıflandırması
2. **Üretken AI ile Açıklama Üretimi**: T5 modelleri kullanarak ürün açıklaması oluşturma

## 🎯 **Hedefler**

* E-ticaret ürün başlıklarından otomatik kategori belirleme
* Ürün başlıklarından anlamlı ve tutarlı açıklamalar üretme
* Farklı makine öğrenmesi modellerinin performansını karşılaştırma
* T5 model varyantlarının etkinliğini değerlendirme

## 🔧 **Kullanılan Teknolojiler**

### Makine Öğrenmesi Modelleri:

* **LogisticRegression**
* **RandomForestClassifier**
* **XGBoostClassifier**

### Üretken AI Modelleri:

* **flan-t5-small**
* **flan-t5-base**

### Kütüphaneler:

```python
pandas, numpy, matplotlib, seaborn, scikit-learn
transformers, datasets, torch, accelerate, sentencepiece
```

## 📊 **Metodoloji**

### 1\. Veri Seti Hazırlığı

* E-ticaret ürün verilerinin toplanması ve temizlenmesi
* Keşifsel veri analizi (EDA) ile veri yapısının incelenmesi
* Metin ön işleme ve feature engineering

### 2\. Gözetimli Öğrenme Yaklaşımı

* **3 Kategori ile Eğitim**: İlk aşamada farklı kategoriler seçilerek model eğitimi
* **5 Kategori ile Eğitim**: Daha karmaşık sınıflandırma için kategori sayısı artırılması
* **Model Karşılaştırması**: Farklı algoritmaların performans analizi

### 3\. Üretken AI Yaklaşımı

* FLAN-T5 model varyantlarının test edilmesi(FLAN-T5-small ve FLAN-T5-base)
* Prompt engineering ve model optimizasyonu
* Çıktı kalitesinin değerlendirilmesi

## 📈 **Ana Bulgular ve Sonuçlar**

### Gözetimli Öğrenme Sonuçları

#### LogisticRegression Performansı:

* **3 Kategori**:

  * Doğruluk (Accuracy): **93.02%**
  * Ağırlıklı F1 Skoru: **93.02%**
![image](https://github.com/user-attachments/assets/3af156a5-2cfa-4388-9c2b-3e24ac377c3f)


* **5 Kategori**:

  * Doğruluk (Accuracy): **81.87%**
  * Ağırlıklı F1 Skoru: **81.87%**
![image](https://github.com/user-attachments/assets/e65acd30-6b9f-47b0-aa8e-fef9d79d811e)


> \*\*Not\*\*: Kategori sayısı arttıkça doğruluk oranında beklenen düşüş gözlenmiştir.

#### Model Karşılaştırması:

* **LogisticRegression**: En yüksek başlangıç performansı
* **RandomForestClassifier**: Dengeli sonuçlar
* **XGBoostClassifier**: Karmaşık veri setleri için optimize edilmiş sonuçlar

#### TF-IDF + KMeans Kümeleme:
![image](https://github.com/user-attachments/assets/c0528396-0611-4175-99a5-5afe54db9109)

### Üretken AI Model Karşılaştırması

## 🔍 **T5 Model Analizi ve Karşılaştırma**

### **t5-small Modeli Değerlendirmesi**

t5- modelleri **talimatları anlamak için değil, desenleri (pattern) devam ettirmek için eğitilmiştir**. Bu modeller görev için tamamen uygun değildir.

### **flan-t5: Talimat Ayarlı Model**

Bu modelleri çok daha iyidir, çünkü **görevin ne olduğunu anlayabilmektedir**.

### **Detaylı Model Karşılaştırması {flan-t5-small ve flan-t5-base}**

| \*\*Kriter\*\* | \*\*flan-t5-small (Daha Zayıf)\*\* | \*\*flan-t5-base (Daha Güçlü)\*\* |
|------------|--------------------------------|-------------------------------|
| \*\*Tutarlılık\*\* | Çok Düşük. 10 üründen sadece 2-3 tanesi kabul edilebilir. | Orta-Yüksek. 10 üründen 7-8 tanesi kabul edilebilir veya iyi. |
| \*\*Cümle Kalitesi\*\* | Genellikle yarım, eksik veya sadece başlığın tekrarı. | Çoğunlukla tam ve dilbilgisi açısından doğru cümleler kuruyor. |
| \*\*Özellik Çıkarma\*\* | En fazla tek bir özellik çıkarabiliyor, onu da nadiren yapıyor. | Genellikle birden fazla özelliği tek bir cümlede birleştirebiliyor. |
| \*\*Yaratıcılık / Dil\*\* | Neredeyse hiç yok. Sadece kopyalama veya çok basit ifadeler. | Ürünün faydasını anlatan sıfatlar ve daha akıcı bir dil kullanmaya başlıyor. |
| \*\*Halüsinasyon Riski\*\* | Yüksek. Prompt'taki örnekle gerçek ürünü birbirine karıştırdı. | Düşük. Girdideki bilgilere sadık kalıyor, bariz bir halüsinasyon yok. |




## 🚀 \*\*Önemli Çıkarımlar\*\*

1. \*\*Model Seçimi Kritik\*\*: flan-t5-base, flan-t5-small'a göre önemli ölçüde daha iyi performans göstermektedir.

2. \*\*Kategori Sayısı Etkisi\*\*: Sınıflandırılacak kategori sayısı arttıkça model performansında düşüş gözlenmektedir.

3. \*\*Talimat Ayarlı Modeller Üstün\*\*: Vanilla T5 modelleri yerine FLAN-T5 gibi talimat ayarlı modellerin kullanılması önemli performans artışı sağlamaktadır.




