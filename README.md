# Online Alışveriş Davranışı Tahmini

Siteye giren herkes alışveriş yapmıyor; kimi bakıp çıkıyor, kimi sepeti dolduruyor. Bu projede bir ziyaretçinin oturum içindeki davranışına bakarak (kaç sayfa gezmiş, ne kadar kalmış, hangi ay gelmiş, yeni mi eski ziyaretçi mi) "bu kişi satın alacak mı" sorusunun cevabını tahmin eden bir sınıflandırma modeli kuruldu.

## Veri seti

Kaynak, UCI Machine Learning Repository'deki **Online Shoppers Purchasing Intention** veri seti (id=468, Sakar & Kastro, 2018). 12.330 oturum, 17 özellik ve hedef değişken olarak `Revenue` var (oturum satın alma ile mi bitti). Eksik değer yok. Pozitif sınıf, yani gerçekten satın alma yapılan oturumlar, toplamın sadece %15.47'si. Veri bu yüzden belirgin şekilde dengesiz ve accuracy tek başına bir şey ifade etmiyor.

Özelliklerin bir kısmı oturum içi gezinme sayıları ve süreleri (`Administrative`, `ProductRelated`, `Informational` ve bunların süre karşılıkları), bir kısmı Google Analytics kaynaklı sayfa kalitesi göstergeleri (`BounceRates`, `ExitRates`, `PageValues`), kalanı da ay, işletim sistemi, tarayıcı, bölge ve trafik kaynağı gibi bağlamsal bilgiler.

## Ön işleme

`Month` ve `VisitorType` one-hot kodlandı; ikisi de anlamsal olarak nominal ve düşük kardinaliteli (10 ve 3 seviye). `OperatingSystems`, `Browser`, `Region`, `TrafficType` de aslında nominal kimlik kodları, aralarında sıra anlamı yok, ama kardinaliteleri yüksek (8-20 seviye arası). Bunları one-hot açmak KNN'de gereksiz bir boyut patlamasına yol açacağından sayısal bırakıldı ve StandardScaler ile ölçeklendi. Karar ağacı ve XGBoost için bu seçimin bir önemi yok, KNN için teorik olarak küçük bir hata payı var; bu veri setinin yayınlanmış analizlerinde de sık görülen bir sadeleştirme.

Eğitim/test ayrımı %80/%20, sınıf oranı korunacak şekilde (stratify) yapıldı: 9.864 eğitim, 2.466 test örneği.

## Modeller

Üç model karşılaştırıldı, hepsi `RandomizedSearchCV` (15-20 deneme) ve 5 katlı çapraz doğrulama ile ayarlandı. Tuning metriği accuracy değil ROC-AUC, çünkü veri dengesiz.

KNN'de komşu sayısı, ağırlıklandırma ve mesafe metriği (Manhattan/Öklid) tarandı. Karar ağacında derinlik, yaprak/bölünme eşikleri ve `class_weight=balanced` seçeneği denendi. XGBoost'ta ağaç sayısı, derinlik, öğrenme oranı, örnekleme oranları ve pozitif sınıfa verilen ağırlık (`scale_pos_weight`) tarandı.

### Sonuçlar (test seti)

| Model | Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|
| XGBoost | 0.861 | 0.532 | 0.840 | 0.652 | 0.930 |
| Karar Ağacı | 0.829 | 0.471 | 0.851 | 0.606 | 0.919 |
| KNN | 0.872 | 0.784 | 0.238 | 0.365 | 0.865 |

XGBoost en dengeli sonucu veriyor ve ROC-AUC'de açık ara önde. KNN'de dikkat çekici bir tuzak var: accuracy'de en yüksek görünüyor (%87.2) ama recall'u sadece 0.238, yani satın alan ziyaretçilerin dörtte üçünü kaçırıyor ve büyük ölçüde "hayır" demeyi öğrenmiş durumda. Dengesiz veride accuracy'nin neden tek başına yanıltıcı olduğunun iyi bir örneği bu. Karar ağacı `class_weight=balanced` sayesinde recall'u yüksek tutmuş (0.851) ama bunun bedelini precision'da ödüyor (0.471).

En iyi model XGBoost'un confusion matrix'i şöyle: 1908 gerçek satın alımın 321'ini doğru yakalıyor, 61'ini kaçırıyor, buna karşılık 282 yanlış alarm veriyor. Bir e-ticaret sitesi için pratik anlamı şu: model potansiyel alıcıların yaklaşık %84'ünü önceden işaretleyip onlara kampanya gösterebilir, karşılığında bir miktar gereksiz gösterim maliyetini kabul eder.

## Görseller

Tüm görseller `gorseller/` klasöründe, PNG ve Plotly olanlarda ayrıca etkileşimli HTML olarak duruyor.

`01_satin_alma_hunisi.png`, ziyaretten satın almaya dönüşüm hunisini gösteriyor. Veri setinde hazır bir huni aşaması alanı olmadığı için dört aşama kurgulandı: toplam ziyaret, ürün sayfası gezinme (`ProductRelated>0`), değerli sayfa görüntüleme (`PageValues>0`, Google Analytics'te bu alan bir sayfanın nihai dönüşüme katkısını gösteriyor) ve satın alma. Son iki aşama birebir iç içe değil; satın alan 1908 kişinin 1538'i (%80.6) PageValues>0 idi, ama huni azalan sırada olduğu için genel eğilimi doğru yansıtıyor.

`02_ziyaretci_tipi_donusum.png`'de yeni ziyaretçilerin dönüşüm oranı (%24.9) geri dönen ziyaretçilerden (%13.9) belirgin şekilde yüksek çıkıyor. İlk bakışta ters gibi görünse de mantıklı bir açıklaması var: geri dönenler büyük çoğunlukla zaten göz atmaya gelmiş ve karar aşamasını geride bırakmış oluyor, yeni ziyaretçiler ise genelde bir arama sonucu veya reklamdan doğrudan alım niyetiyle geliyor.

`03_aylik_donusum_orani.png`'de Kasım ayı %25.4 ile en yüksek dönüşüm oranına sahip (muhtemelen Black Friday etkisi), Şubat %1.6 ile en düşük. Veri setinde Ocak ve Nisan ayları hiç yok, bu kaynağın kendi eksiği.

`04_korelasyon_isi_haritasi.png`'de `PageValues`, `Revenue` ile en yüksek korelasyona sahip sayısal değişken (0.493), açık farkla. `BounceRates` ve `ExitRates` kendi aralarında çok yüksek korele (0.91); ikisi de sitede terk davranışını ölçüyor.

`05_karar_agaci.png` için dtreeviz denendi ama bu makinede graphviz'in `dot` komutu kurulu olmadığından (sadece Python paketi kurulu, sistem binary'si yok) otomatik olarak `sklearn.tree.plot_tree`'ye düşüldü. Görselde sınırlı derinlikte (max_depth=3) ayrı bir ağaç kullanıldı, karşılaştırma tablosundaki tuned ağaçtan farklı.

`06_confusion_matrix.png`, en iyi model olan XGBoost için üretildi.

## Belirsizlikler ve alınan kararlar

Huni kurgusu veride hazır gelmiyor, dört aşama `ProductRelated` ve `PageValues` üzerinden mantıksal olarak türetildi (güven: MEDIUM, eğilimi doğru yansıtıyor ama üretici sistemdeki gerçek huni tanımından farklı olabilir).

Yüksek kardinaliteli kategorik alanlar (`OperatingSystems`, `Browser`, `Region`, `TrafficType`) one-hot yerine sayısal bırakıldı; KNN için teorik bir dezavantaj ama boyut patlamasından kaçınmak için bilinçli bir tercih (güven: MEDIUM).

dtreeviz için sistemde graphviz `dot` binary'si yok, kod başından itibaren `plot_tree`'ye otomatik düşecek şekilde try/except ile yazıldı; bu yüzden iş yarım kalmadı.

## Kullanılan kütüphaneler

- [pandas](https://pandas.pydata.org/) ve [numpy](https://numpy.org/): veri işleme
- [scikit-learn](https://scikit-learn.org/): KNN, karar ağacı, pipeline, RandomizedSearchCV, metrikler
- [xgboost](https://xgboost.readthedocs.io/): gradyan artırmalı ağaç modeli
- [matplotlib](https://matplotlib.org/) ve [seaborn](https://seaborn.pydata.org/): korelasyon ısı haritası, karar ağacı görseli
- [plotly](https://plotly.com/python/): huni, bar ve confusion matrix görselleri (kaleido ile PNG'ye aktarım)
- [dtreeviz](https://github.com/parrt/dtreeviz): denendi, graphviz eksikliğinden dolayı plot_tree'ye düşüldü
- [ucimlrepo](https://github.com/uci-ml-repo/ucimlrepo): UCI veri seti indirme
- [jupytext](https://jupytext.readthedocs.io/) ve [nbconvert](https://nbconvert.readthedocs.io/): `.py` dosyasını `.ipynb`'ye çevirme ve çalıştırma

## Dosya yapısı

```
08_online_alisveris/
├── proje.py                           ana script, notebook buradan üretildi
├── proje.ipynb                        çalıştırılmış notebook, tüm hücre çıktıları dolu
├── model_karsilastirma_sonuclari.csv  3 modelin tam metrik tablosu ve en iyi parametreler
├── ozet_metrikler.json                README'deki sayıların kaynağı
├── requirements.txt
├── veri/online_shoppers_intention.csv UCI'dan indirilen ham veri, git'e gitmiyor
└── gorseller/                         6 görsel, PNG ve HTML
```
