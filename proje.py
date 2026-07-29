# %% [markdown]
# # Online Alışveriş Davranışı Tahmini
#
# Siteye giren herkes alışveriş yapmıyor; kimi bakıp çıkıyor, kimi sepeti dolduruyor.
# Bu projede bir ziyaretçinin oturum içindeki davranışına (kaç sayfa gezdi, ne kadar
# kaldı, hangi ay geldi, yeni mi eski ziyaretçi mi...) bakıp "bu kişi satın alacak mı?"
# sorusunun cevabını tahmin eden bir sınıflandırma modeli kuruyoruz.
#
# Veri seti: **Online Shoppers Purchasing Intention** (UCI Machine Learning Repository,
# id=468, Sakar & Kastro, 2018). 12.330 oturum, 17 özellik, hedef değişken `Revenue`
# (oturum satın alma ile mi bitti).

# %%
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
import plotly.express as px

from sklearn.model_selection import train_test_split, StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix,
)
from xgboost import XGBClassifier

warnings.filterwarnings("ignore")

# Path(__file__) proje.ipynb -> proje.py dönüşümünde / nbconvert içinde sorun
# çıkarabiliyor; bu yüzden __file__ yerine bu güvenli desen kullanılıyor.
PROJE_KOK = Path(__file__).resolve().parent if "__file__" in dir() else Path.cwd()
VERI_YOLU = PROJE_KOK / "veri" / "online_shoppers_intention.csv"
GORSEL_KOK = PROJE_KOK / "gorseller"
GORSEL_KOK.mkdir(exist_ok=True)

RASSAL_TOHUM = 42
np.random.seed(RASSAL_TOHUM)

# --- Modern/şık renk paleti (dataviz skill referans paletinden) ---
RENK_MAVI = "#2a78d6"
RENK_TURUNCU = "#eb6834"
RENK_AKUA = "#1baf7a"
RENK_SARI = "#eda100"
RENK_MOR = "#4a3aa7"
RENK_KIRMIZI = "#e34948"
RENK_YESIL = "#008300"
YUZEY = "#fcfcfb"
METIN_ANA = "#0b0b0b"
METIN_IKINCIL = "#52514e"
IZGARA = "#e1e0d9"
KATEGORIK_PALET = [RENK_MAVI, RENK_TURUNCU, RENK_AKUA, RENK_SARI, RENK_MOR, RENK_KIRMIZI]
MAVI_SIRALI = ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"]

PLOTLY_TEMA = dict(
    plot_bgcolor=YUZEY,
    paper_bgcolor=YUZEY,
    font=dict(family="system-ui, -apple-system, Segoe UI, sans-serif", color=METIN_ANA, size=13),
    title_font=dict(size=17, color=METIN_ANA),
)

print("=" * 70)
print("ADIM 1: VERİ YÜKLEME VE KEŞİF")
print("=" * 70)

# %%
df = pd.read_csv(VERI_YOLU)
print(f"Veri boyutu: {df.shape[0]} satır, {df.shape[1]} sütun")
print(f"Eksik değer sayısı: {df.isna().sum().sum()}")
print(f"Sütunlar: {df.columns.tolist()}")
print(f"\nHedef değişken (Revenue) dağılımı:")
print(df["Revenue"].value_counts())
pozitif_oran = df["Revenue"].mean()
print(f"Pozitif sınıf (satın alma) oranı: %{pozitif_oran*100:.2f}")

# %% [markdown]
# ## Görsel 1 — Satın Alma Hunisi
#
# Veri setinde doğrudan bir "huni aşaması" alanı yok; aşamaları şu mantıkla kurguladık:
#
# 1. **Toplam ziyaret**: veri setindeki tüm oturumlar.
# 2. **Ürün sayfası gezinme**: `ProductRelated > 0` — en az bir ürün sayfasına bakan oturumlar.
# 3. **Değerli sayfa görüntüleme**: `PageValues > 0` — Google Analytics'in bu alanı, bir
#    sayfanın nihai dönüşüme (satın almaya) ne kadar katkı sağladığını gösteriyor;
#    0'ın üzerinde olması oturumun dönüşüme yaklaştığının güçlü bir işareti.
# 4. **Satın alma**: `Revenue == True`.
#
# Not: 3. ve 4. aşama birebir iç içe geçmiş (nested) değil — satın alan 1908 oturumun
# 1538'i (%80.6) PageValues>0 idi, geri kalanı PageValues=0 olsa da satın almış.
# Yani PageValues güçlü bir sinyal ama kesin bir kapı değil; huni sayıları azalan
# sırada olduğu için genel eğilimi doğru yansıtıyor.

# %%
asama_isimleri = ["Toplam Ziyaret", "Ürün Sayfası Gezinme", "Değerli Sayfa Görüntüleme", "Satın Alma"]
asama_sayilari = [
    len(df),
    int((df["ProductRelated"] > 0).sum()),
    int(((df["ProductRelated"] > 0) & (df["PageValues"] > 0)).sum()),
    int((df["Revenue"] == True).sum()),
]
print("Huni aşama sayıları:", dict(zip(asama_isimleri, asama_sayilari)))

fig_huni = go.Figure(go.Funnel(
    y=asama_isimleri,
    x=asama_sayilari,
    textinfo="value+percent initial",
    hovertemplate="<b>%{y}</b><br>Oturum sayısı: %{x}<br>Başlangıca göre oran: %{percentInitial:.1%}<extra></extra>",
    marker=dict(color=MAVI_SIRALI[1:5]),
    connector=dict(line=dict(color=IZGARA, width=1)),
))
fig_huni.update_layout(
    title="Ziyaretten Satın Almaya Dönüşüm Hunisi",
    xaxis_title="Oturum Sayısı",
    **PLOTLY_TEMA,
    height=450, width=800,
)
fig_huni.write_image(str(GORSEL_KOK / "01_satin_alma_hunisi.png"), scale=2)
fig_huni.write_html(str(GORSEL_KOK / "01_satin_alma_hunisi.html"))
print("Kaydedildi: gorseller/01_satin_alma_hunisi.png (+.html)")

# %% [markdown]
# ## Görsel 2 — Ziyaretçi Tipine Göre Dönüşüm Oranı

# %%
ziyaretci_donusum = df.groupby("VisitorType", observed=True)["Revenue"].mean().sort_values(ascending=False) * 100
ziyaretci_adet = df["VisitorType"].value_counts()
print("Ziyaretçi tipine göre dönüşüm oranı (%):")
print(ziyaretci_donusum)

turkce_isim = {"New_Visitor": "Yeni Ziyaretçi", "Returning_Visitor": "Geri Dönen Ziyaretçi", "Other": "Diğer"}
fig_ziyaretci = go.Figure(go.Bar(
    x=[turkce_isim[i] for i in ziyaretci_donusum.index],
    y=ziyaretci_donusum.values,
    marker_color=KATEGORIK_PALET[:len(ziyaretci_donusum)],
    text=[f"%{v:.1f}" for v in ziyaretci_donusum.values],
    textposition="outside",
    hovertemplate="<b>%{x}</b><br>Dönüşüm oranı: %{y:.1f}%<extra></extra>",
))
fig_ziyaretci.update_layout(
    title="Ziyaretçi Tipine Göre Satın Alma Dönüşüm Oranı",
    yaxis_title="Dönüşüm Oranı (%)", xaxis_title="",
    **PLOTLY_TEMA, height=450, width=750,
)
fig_ziyaretci.update_yaxes(gridcolor=IZGARA)
fig_ziyaretci.write_image(str(GORSEL_KOK / "02_ziyaretci_tipi_donusum.png"), scale=2)
fig_ziyaretci.write_html(str(GORSEL_KOK / "02_ziyaretci_tipi_donusum.html"))
print("Kaydedildi: gorseller/02_ziyaretci_tipi_donusum.png (+.html)")

# %% [markdown]
# ## Görsel 3 — Aylara Göre Dönüşüm Oranı
#
# Not: Veri setinde Ocak ve Nisan ayları hiç yok (kaynak veri setinin kendi eksiği,
# bizim temizliğimizden değil) — bu yüzden grafikte 10 ay görünüyor.

# %%
ay_sirasi = ["Feb", "Mar", "May", "June", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]
ay_turkce = {"Feb": "Şub", "Mar": "Mar", "May": "May", "June": "Haz", "Jul": "Tem",
             "Aug": "Ağu", "Sep": "Eyl", "Oct": "Eki", "Nov": "Kas", "Dec": "Ara"}
ay_donusum = df.groupby("Month", observed=True)["Revenue"].mean().reindex(ay_sirasi) * 100
print("Aylara göre dönüşüm oranı (%):")
print(ay_donusum)

fig_ay = go.Figure(go.Bar(
    x=[ay_turkce[a] for a in ay_sirasi],
    y=ay_donusum.values,
    marker_color=RENK_MAVI,
    text=[f"%{v:.1f}" for v in ay_donusum.values],
    textposition="outside",
    hovertemplate="<b>Ay: %{x}</b><br>Dönüşüm oranı: %{y:.1f}%<extra></extra>",
))
fig_ay.update_layout(
    title="Aylara Göre Satın Alma Dönüşüm Oranı",
    yaxis_title="Dönüşüm Oranı (%)", xaxis_title="Ay",
    **PLOTLY_TEMA, height=450, width=850,
)
fig_ay.update_yaxes(gridcolor=IZGARA)
fig_ay.write_image(str(GORSEL_KOK / "03_aylik_donusum_orani.png"), scale=2)
fig_ay.write_html(str(GORSEL_KOK / "03_aylik_donusum_orani.html"))
print("Kaydedildi: gorseller/03_aylik_donusum_orani.png (+.html)")

# %% [markdown]
# ## Görsel 4 — Korelasyon Isı Haritası

# %%
sayisal_sutunlar = df.select_dtypes(include=[np.number]).columns.tolist()
korelasyon = df[sayisal_sutunlar + ["Revenue"]].assign(Revenue=df["Revenue"].astype(int)).corr()

# Orijinal sütun adları (İngilizce, veri setinin kendi isimleri) koda dokunulmadan
# kalıyor; sadece görselde okunabilirlik için yanına Türkçe açıklama ekleniyor.
turkce_aciklama = {
    "Administrative": "yönetimsel sayfa sayısı",
    "Administrative_Duration": "yönetimsel sayfada geçirilen süre",
    "Informational": "bilgilendirme sayfası sayısı",
    "Informational_Duration": "bilgilendirme sayfasında geçirilen süre",
    "ProductRelated": "ürün sayfası sayısı",
    "ProductRelated_Duration": "ürün sayfasında geçirilen süre",
    "BounceRates": "hemen çıkma oranı",
    "ExitRates": "çıkış oranı",
    "PageValues": "sayfa değeri skoru",
    "SpecialDay": "özel güne yakınlık",
    "OperatingSystems": "işletim sistemi kodu",
    "Browser": "tarayıcı kodu",
    "Region": "bölge kodu",
    "TrafficType": "trafik türü kodu",
    "Revenue": "satın alma (hedef)",
}
eksen_etiketleri = [f"{c}\n({turkce_aciklama.get(c, '')})" if c in turkce_aciklama else c
                     for c in korelasyon.columns]

plt.figure(figsize=(13, 10.5))
ozel_cmap = matplotlib.colors.LinearSegmentedColormap.from_list(
    "mavi_kirmizi_diverging", [RENK_MAVI, "#f0efec", RENK_KIRMIZI]
)
sns.heatmap(korelasyon, annot=True, fmt=".2f", cmap=ozel_cmap, center=0,
            linewidths=0.5, linecolor=IZGARA, square=True,
            xticklabels=eksen_etiketleri, yticklabels=eksen_etiketleri,
            annot_kws={"size": 8},
            cbar_kws={"shrink": 0.8, "label": "Korelasyon Katsayısı"})
plt.title("Sayısal Değişkenler Arası Korelasyon", fontsize=14, color=METIN_ANA, pad=15)
plt.xticks(rotation=45, ha="right", fontsize=8)
plt.yticks(fontsize=8)
plt.tight_layout()
plt.savefig(GORSEL_KOK / "04_korelasyon_isi_haritasi.png", dpi=150, facecolor=YUZEY)
plt.close()
print("Kaydedildi: gorseller/04_korelasyon_isi_haritasi.png")
print(f"Revenue ile en yüksek korelasyon: {korelasyon['Revenue'].drop('Revenue').abs().idxmax()} "
      f"({korelasyon['Revenue'].drop('Revenue').abs().max():.3f})")

# %% [markdown]
# ## Ön İşleme
#
# **Kodlama kararı (Not):** `Month` ve `VisitorType` anlamsal olarak nominal ve düşük
# kardinaliteli (10 ve 3 seviye) olduğu için one-hot kodlanıyor. `OperatingSystems`,
# `Browser`, `Region`, `TrafficType` de aslında nominal kimlik kodları (1, 2, 3...
# aralarında sıra anlamı yok) ama kardinaliteleri yüksek (8-20 seviye); tamamını
# one-hot açmak KNN için boyut patlamasına yol açacağından, bu dört alan sayısal
# olarak bırakılıp StandardScaler ile ölçekleniyor. Bu, bu veri setinin yayınlanmış
# analizlerinde de sık kullanılan bir sadeleştirme; karar ağacı/XGBoost için zaten
# fark etmiyor, KNN için teorik olarak küçük bir hata payı var — güven: MEDIUM.

# %%
hedef = "Revenue"
onehot_sutunlar = ["Month", "VisitorType"]
sayisal_sutunlar_x = [c for c in df.columns if c not in onehot_sutunlar + [hedef, "Weekend"]]
# Weekend bool -> int
df["Weekend"] = df["Weekend"].astype(int)
sayisal_sutunlar_x = [c for c in sayisal_sutunlar_x]  # Weekend hariç, ayrı ekleniyor

X = df.drop(columns=[hedef])
y = df[hedef].astype(int)

on_isleyici = ColumnTransformer(transformers=[
    ("kategorik", OneHotEncoder(drop="first", handle_unknown="ignore"), onehot_sutunlar),
    ("sayisal", StandardScaler(), [c for c in X.columns if c not in onehot_sutunlar]),
])

X_egitim, X_test, y_egitim, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RASSAL_TOHUM, stratify=y
)
print(f"Eğitim seti: {X_egitim.shape[0]} örnek | Test seti: {X_test.shape[0]} örnek")
print(f"Eğitimde pozitif oran: %{y_egitim.mean()*100:.2f} | Testte: %{y_test.mean()*100:.2f}")

kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=RASSAL_TOHUM)

# %%
print("=" * 70)
print("ADIM 2: MODEL EĞİTİMİ (RandomizedSearchCV + 5 katlı CV)")
print("=" * 70)

sonuclar = []
en_iyi_modeller = {}


def modeli_degerlendir(isim, pipeline, param_grid, n_iter=15):
    print(f"\n--- {isim} ---")
    arama = RandomizedSearchCV(
        pipeline, param_distributions=param_grid, n_iter=n_iter,
        scoring="roc_auc", cv=kfold, random_state=RASSAL_TOHUM,
        n_jobs=2, verbose=0,
    )
    arama.fit(X_egitim, y_egitim)
    en_iyi = arama.best_estimator_
    print(f"En iyi parametreler: {arama.best_params_}")
    print(f"CV en iyi ROC-AUC (eğitim): {arama.best_score_:.4f}")

    y_tahmin = en_iyi.predict(X_test)
    y_olasilik = en_iyi.predict_proba(X_test)[:, 1]

    metrikler = {
        "Model": isim,
        "Accuracy": accuracy_score(y_test, y_tahmin),
        "Precision": precision_score(y_test, y_tahmin),
        "Recall": recall_score(y_test, y_tahmin),
        "F1": f1_score(y_test, y_tahmin),
        "ROC_AUC": roc_auc_score(y_test, y_olasilik),
        "CV_ROC_AUC": arama.best_score_,
        "En_Iyi_Parametreler": str(arama.best_params_),
    }
    print(f"Test seti -> Accuracy: {metrikler['Accuracy']:.4f} | Precision: {metrikler['Precision']:.4f} | "
          f"Recall: {metrikler['Recall']:.4f} | F1: {metrikler['F1']:.4f} | ROC-AUC: {metrikler['ROC_AUC']:.4f}")

    sonuclar.append(metrikler)
    en_iyi_modeller[isim] = en_iyi
    return en_iyi, y_tahmin, y_olasilik


# --- 1) KNN ---
knn_pipeline = Pipeline([("onisleme", on_isleyici), ("model", KNeighborsClassifier())])
knn_grid = {
    "model__n_neighbors": [5, 7, 9, 11, 15, 21, 25, 31],
    "model__weights": ["uniform", "distance"],
    "model__p": [1, 2],
}
modeli_degerlendir("KNN", knn_pipeline, knn_grid, n_iter=15)

# --- 2) Karar Ağacı ---
agac_pipeline = Pipeline([("onisleme", on_isleyici), ("model", DecisionTreeClassifier(random_state=RASSAL_TOHUM))])
agac_grid = {
    "model__max_depth": [3, 4, 5, 6, 8, 10, 12, None],
    "model__min_samples_split": [2, 5, 10, 20, 40],
    "model__min_samples_leaf": [1, 2, 5, 10, 20],
    "model__criterion": ["gini", "entropy"],
    "model__class_weight": [None, "balanced"],
}
modeli_degerlendir("Karar Ağacı", agac_pipeline, agac_grid, n_iter=20)

# --- 3) XGBoost ---
pozitif_agirlik = (y_egitim == 0).sum() / (y_egitim == 1).sum()
xgb_pipeline = Pipeline([("onisleme", on_isleyici),
                          ("model", XGBClassifier(random_state=RASSAL_TOHUM, eval_metric="logloss", n_jobs=2))])
xgb_grid = {
    "model__n_estimators": [100, 200, 300, 400],
    "model__max_depth": [3, 4, 5, 6, 8],
    "model__learning_rate": [0.01, 0.03, 0.05, 0.1, 0.2],
    "model__subsample": [0.7, 0.8, 0.9, 1.0],
    "model__colsample_bytree": [0.6, 0.8, 1.0],
    "model__scale_pos_weight": [1, pozitif_agirlik],
}
modeli_degerlendir("XGBoost", xgb_pipeline, xgb_grid, n_iter=20)

# %%
sonuc_df = pd.DataFrame(sonuclar).sort_values("ROC_AUC", ascending=False).reset_index(drop=True)
print("\n" + "=" * 70)
print("MODEL KARŞILAŞTIRMA TABLOSU")
print("=" * 70)
print(sonuc_df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC", "CV_ROC_AUC"]].to_string(index=False))

sonuc_df.to_csv(PROJE_KOK / "model_karsilastirma_sonuclari.csv", index=False)
print("\nKaydedildi: model_karsilastirma_sonuclari.csv")

en_iyi_model_adi = sonuc_df.iloc[0]["Model"]
print(f"\nEn iyi model (ROC-AUC bazında): {en_iyi_model_adi}")

# %% [markdown]
# ## Görsel 5 — Karar Ağacı Görselleştirmesi
#
# Not: Tuning sonucu seçilen karar ağacı çok derin olabileceğinden (okunaksız olur),
# görselleştirme için AYRI ve sınırlı derinlikte (max_depth=3) bir ağaç eğitildi.
# Bu sadece görsel amaçlıdır; karşılaştırma tablosundaki gerçek (tuned) ağaçtan farklı
# olabilir ve performans karşılaştırmasında kullanılmaz.

# %%
X_egitim_islenmis = on_isleyici.fit_transform(X_egitim, y_egitim)
ozellik_isimleri = on_isleyici.get_feature_names_out()

gorsel_agac = DecisionTreeClassifier(max_depth=3, random_state=RASSAL_TOHUM, class_weight="balanced")
gorsel_agac.fit(X_egitim_islenmis, y_egitim)

dtreeviz_basarili = False
try:
    import dtreeviz as dtv
    viz_model = dtv.model(
        gorsel_agac, X_train=X_egitim_islenmis, y_train=y_egitim,
        target_name="Revenue (satın alma)", feature_names=list(ozellik_isimleri),
        class_names=["Hayır", "Evet"],
    )
    viz = viz_model.view()
    viz.save(str(GORSEL_KOK / "05_karar_agaci.svg"))
    dtreeviz_basarili = True
    print("Kaydedildi: gorseller/05_karar_agaci.svg (dtreeviz)")
except Exception as hata:
    print(f"dtreeviz başarısız oldu ({type(hata).__name__}: {hata}), sklearn.tree.plot_tree'ye düşülüyor "
          f"(sistemde graphviz 'dot' komutu kurulu değil).")
    # graphviz, 'dot' binary'sini çağırmadan önce uzantısız bir .dot kaynak dosyası bırakıyor;
    # bu kalıntıyı temizle ki gorseller/ klasörü karışmasın.
    kalinti_dosya = GORSEL_KOK / "05_karar_agaci"
    if kalinti_dosya.exists():
        kalinti_dosya.unlink()

if not dtreeviz_basarili:
    plt.figure(figsize=(20, 10))
    plot_tree(gorsel_agac, feature_names=list(ozellik_isimleri), class_names=["Hayır", "Evet"],
              filled=True, rounded=True, fontsize=9,
              impurity=True, proportion=False)
    # sklearn'in plot_tree fonksiyonu düğüm metinlerini (gini/samples/value/class) ve
    # kök düğümün True/False dal etiketlerini sabit İngilizce üretiyor; görselde her
    # şeyin Türkçe olması için üretilen Text nesneleri burada elle çevriliyor
    # (özellik adları ve eşik değerleri —gerçek sütun adları— değişmeden kalıyor).
    for metin in plt.gca().texts:
        yeni_metin = (
            metin.get_text()
            .replace("gini =", "gini katsayısı =")
            .replace("entropy =", "entropi =")
            .replace("log_loss =", "log kaybı =")
            .replace("samples =", "örnek sayısı =")
            .replace("value =", "değer =")
            .replace("class =", "sınıf =")
            .replace("True", "Doğru")
            .replace("False", "Yanlış")
        )
        metin.set_text(yeni_metin)
    plt.title("Karar Ağacı (görselleştirme amaçlı, max_depth=3)", fontsize=14, color=METIN_ANA)
    plt.tight_layout()
    plt.savefig(GORSEL_KOK / "05_karar_agaci.png", dpi=150, facecolor=YUZEY)
    plt.close()
    print("Kaydedildi: gorseller/05_karar_agaci.png (plot_tree)")

# %% [markdown]
# ## Görsel 6 — Karışıklık Matrisi (En İyi Model)

# %%
en_iyi_model = en_iyi_modeller[en_iyi_model_adi]
y_tahmin_final = en_iyi_model.predict(X_test)
cm = confusion_matrix(y_test, y_tahmin_final)
cm_etiket = [["Doğru Negatif", "Yanlış Pozitif"], ["Yanlış Negatif", "Doğru Pozitif"]]
cm_metin = [[f"{cm[i][j]}<br>{cm_etiket[i][j]}" for j in range(2)] for i in range(2)]

fig_cm = go.Figure(go.Heatmap(
    z=cm, x=["Tahmin: Hayır", "Tahmin: Evet"], y=["Gerçek: Hayır", "Gerçek: Evet"],
    text=cm_metin, texttemplate="%{text}", textfont=dict(size=14, color=METIN_ANA),
    hovertemplate="%{y}<br>%{x}<br>Oturum sayısı: %{z}<extra></extra>",
    colorscale=[[0, "#cde2fb"], [0.5, "#3987e5"], [1, "#0d366b"]],
    colorbar=dict(title="Oturum Sayısı"),
    showscale=True,
))
fig_cm.update_layout(
    title=f"Karışıklık Matrisi — {en_iyi_model_adi} (En İyi Model)",
    **PLOTLY_TEMA, height=500, width=600,
)
fig_cm.update_yaxes(autorange="reversed")
fig_cm.write_image(str(GORSEL_KOK / "06_confusion_matrix.png"), scale=2)
fig_cm.write_html(str(GORSEL_KOK / "06_confusion_matrix.html"))
print("Kaydedildi: gorseller/06_confusion_matrix.png (+.html)")
print(f"\nConfusion matrix ({en_iyi_model_adi}):\n{cm}")

# %%
print("=" * 70)
print("ÖZET BİLGİLER (README için)")
print("=" * 70)
ozet = {
    "veri_satir_sayisi": int(len(df)),
    "veri_sutun_sayisi": int(df.shape[1]),
    "pozitif_oran": float(pozitif_oran),
    "egitim_boyutu": int(len(X_egitim)),
    "test_boyutu": int(len(X_test)),
    "en_iyi_model": en_iyi_model_adi,
    "en_yuksek_korelasyon_degisken": str(korelasyon["Revenue"].drop("Revenue").abs().idxmax()),
    "en_yuksek_korelasyon_deger": float(korelasyon["Revenue"].drop("Revenue").abs().max()),
    "huni_asamalari": dict(zip(asama_isimleri, asama_sayilari)),
    "ziyaretci_donusum": ziyaretci_donusum.to_dict(),
    "en_yuksek_donusum_ayi": str(ay_donusum.idxmax()),
    "en_yuksek_donusum_ay_orani": float(ay_donusum.max()),
    "en_dusuk_donusum_ayi": str(ay_donusum.idxmin()),
    "en_dusuk_donusum_ay_orani": float(ay_donusum.min()),
    "sonuc_tablosu": sonuc_df[["Model", "Accuracy", "Precision", "Recall", "F1", "ROC_AUC"]].to_dict("records"),
    "dtreeviz_basarili": dtreeviz_basarili,
}
with open(PROJE_KOK / "ozet_metrikler.json", "w", encoding="utf-8") as f:
    json.dump(ozet, f, ensure_ascii=False, indent=2)
print("Kaydedildi: ozet_metrikler.json")
print(json.dumps(ozet, ensure_ascii=False, indent=2))

print("\nTÜM ADIMLAR TAMAMLANDI.")
