# Segmentation sémantique Cityscapes — Veille & Amélioration

Benchmark de 7 architectures de segmentation sémantique sur le dataset Cityscapes (8 classes), du CNN classique au foundation model auto-supervisé. Dashboard interactif Streamlit pour explorer les résultats.

Ce dépôt fait suite au [projet initial de segmentation embarquée](https://github.com/El-GuiGui/P8-Traitez-les-images-pour-le-systeme-embarquer-d-une-voiture-autonome) (U-Net from scratch, VGG16, ResNet50, API FastAPI, app Streamlit de prédiction). Ici, on reprend la même base de code et le même dataset pour y ajouter des architectures récentes et un dashboard de comparaison.

---

## Ce qui change par rapport au projet initial

Le projet initial avait mis en place les baselines (scratch, VGG16, ResNet50), une API de prédiction et une app de démo.

Ce projet ajoute :

- **3 nouvelles architectures** : ConvNeXt Tiny, SegFormer (MiT-B0 et MiT-B5), EoMT/DINOv2
- **27 runs benchmarkés** avec variation de résolution, batch size, augmentation, loss, mode d'entraînement
- **Un dashboard Streamlit de comparaison** (EDA, comparaison des runs, accessibilité WCAG)
- **Un notebook de quickbench 256 vs 512** pour mesurer l'impact de la résolution
- **Un CSV** (comparison_all_runs.csv) regroupant toutes les métriques

L'API FastAPI et l'app de prédiction du projet initial restent fonctionnelles dans ce dépôt (dossiers `api/` et `app/`) mais ne sont pas le livrable principal ici.

---

## Dataset

[Cityscapes](https://www.cityscapes-dataset.com/) — 5 000 images haute résolution (2048×1024), caméras embarquées, 50 villes européennes.

Split : 2 975 train / 500 val / 1 525 test.

Les 32 sous-catégories sont regroupées en 8 classes :

| Indice | Catégorie    | Exemples                  | Couleur    |
| ------ | ------------ | ------------------------- | ---------- |
| 0      | void         | Non-étiqueté, ego-vehicle | Noir       |
| 1      | flat         | Route, trottoir           | Violet     |
| 2      | construction | Bâtiments, murs, clôtures | Gris foncé |
| 3      | object       | Poteaux, panneaux, feux   | Gris clair |
| 4      | nature       | Végétation, terrain       | Vert       |
| 5      | sky          | Ciel                      | Bleu       |
| 6      | human        | Piétons, cyclistes        | Rouge      |
| 7      | vehicle      | Voitures, camions, bus    | Bleu foncé |

Remapping via lookup table dans `scripts/preprocessing.py`. Pixels ambigus = `IGNORE_LABEL=255`.

---

## Architectures

### Baselines (projet initial)

| Modèle             | Encodeur | Pré-entraînement | Framework |
| ------------------ | -------- | ---------------- | --------- |
| U-Net from scratch | —        | Non              | TF/Keras  |
| U-Net + VGG16      | VGG16    | ImageNet         | TF/Keras  |
| U-Net + ResNet50   | ResNet50 | ImageNet         | TF/Keras  |

### Nouveaux modèles (ce projet)

| Modèle                    | Encodeur           | Pré-entraînement          | Framework    |
| ------------------------- | ------------------ | ------------------------- | ------------ |
| U-Net + ConvNeXt Tiny     | ConvNeXt Tiny      | ImageNet                  | TF/Keras     |
| SegFormer MiT-B0          | MiT-B0 (~3.8M)     | Cityscapes (keras-hub)    | TF/Keras     |
| SegFormer MiT-B5          | MiT-B5 (~82M)      | Cityscapes (keras-hub)    | TF/Keras     |
| EoMT DINOv2 (tête dense)  | DINOv2-Base (~86M) | Auto-supervisé (142M img) | PyTorch + HF |
| EoMT DINOv2 (multi-scale) | DINOv2-Base (~86M) | Auto-supervisé (142M img) | PyTorch + HF |

---

## Résultats

Métrique de référence : **val_mIoU** (Mean IoU sur validation, sans augmentation).

### Top runs à 256×256

| Modèle                   | Val mIoU | Mode     | Époques | Durée    |
| ------------------------ | -------- | -------- | ------- | -------- |
| ConvNeXt Tiny FT (50+30) | 0.7598   | finetune | 30      | 277 min  |
| U-Net Scratch            | 0.7539   | scratch  | 40      | 199 min  |
| ResNet50 FT              | 0.7471   | finetune | 30      | 257 min  |
| VGG16 FT                 | 0.7465   | finetune | 30      | 167 min  |
| EoMT MultiScale + Cosine | 0.7338   | finetune | 80      | 168 min  |
| SegFormer MiT-B5         | 0.7213   | finetune | 15      | 148 min  |
| SegFormer MiT-B0 (50+30) | 0.7140   | finetune | 80      | 273 min  |
| EoMT DINOv2 tête dense   | 0.7119   | finetune | 80      | ~280 min |

### Top runs à 512×512 (quickbench, 20 époques)

| Modèle               | Val mIoU | Gain vs 256 |
| -------------------- | -------- | ----------- |
| ConvNeXt Tiny FT     | 0.8256   | +5.2 pts    |
| VGG16 FT             | 0.7970   | +5.1 pts    |
| ConvNeXt Tiny Frozen | 0.7965   | +3.9 pts    |
| SegFormer B0 Frozen  | 0.7805   | +6.7 pts    |
| ResNet50 Frozen      | 0.7754   | +3.1 pts    |

L'ensemble des 27 runs est dans `out/compare/comparison_all_runs.csv` .

### Observations principales

- **La résolution est le levier le plus puissant** : passer de 256 à 512 donne +3 à +6 pts selon le modèle.
- **L'augmentation est indispensable** : le scratch sans augmentation plafonne à 0.6477, avec augmentation il monte à 0.7539 (+10 pts).
- **CE+Dice > CE seule** : les runs en CE seule tournent à 0.63–0.65, CE+Dice dépasse 0.75.
- **ConvNeXt Tiny domine** le classement à 256 et à 512.
- **MiT-B5 converge 2× plus vite que MiT-B0** (148 min vs 273 min) pour un meilleur score.
- **EoMT/DINOv2** produit les prédictions les mieux calibrées (loss de test la plus basse du benchmark).
- Les classes **object** (IoU 0.30–0.39) et **human** (0.57–0.63) restent en retrait sur tous les modèles à cause du déséquilibre du dataset.

---

## Architecture du projet

```
PROJ9/
├── data/
│   └── raw/cityscapes/
│       ├── leftImg8bit/                  # Images RGB
│       └── gtFine/                       # Masques d'annotation
│
├── notebooks/
│   ├── 1_exploration.ipynb               # EDA du dataset
│   ├── 2_Benchmark_Unet_from_scratch.ipynb
│   ├── 3_Benchmark_Unet_VGG16.ipynb
│   ├── 4_Benchmark_Unet_Resnet.ipynb
│   ├── 5_Benchmark_Unet_ConvNeXt.ipynb        # Nouveau
│   ├── 6_Benchmark_SegFormer_MiTB0.ipynb       # Nouveau
│   ├── 6d_Benchmark_SegFormer_MiTB5.ipynb      # Nouveau
│   ├── 7_Benchmark_EoMT_DINOv2.ipynb           # Nouveau
│   ├── 7d_Benchmark_EoMT_MultiScale_Cosine.ipynb  # Nouveau
│   ├── 8_Resolution_Comparison_256_vs_512.ipynb # Nouveau
│   └── 0_Comparaison.ipynb              # Agrégation des runs → CSV
│
├── scripts/
│   ├── config.py             # Chemins, détection auto de la racine
│   ├── preprocessing.py      # Remapping 8 classes, colorisation, overlay
│   ├── datagen.py            # CityscapesSequence (Keras) + DataLoader (PyTorch)
│   ├── augmentations.py      # Pipeline Albumentations
│   ├── losses_metrics.py     # CE+Dice, MeanIoUArgmax
│   ├── models.py             # Toutes les architectures (scratch, VGG16, ResNet50, ConvNeXt, SegFormer)
│   ├── training.py           # Boucles d'entraînement automatisées
│   ├── inference.py          # Prédiction + téléchargement modèle HuggingFace
│   ├── viz.py                # Visualisation des prédictions
│   └── seed.py               # Fix seed pour reproductibilité
│
├── out/
│   ├── experiments/          # Un sous-dossier par run : summary.json, history.json, *.png
│   ├── compare/
│   │   └── comparison_all_runs.csv    # 27 runs × 33 colonnes
│   └── cityscapes_split_testXtrainXval.csv
│
├── app_dashboard/
│   ├── streamlit_app.py      # Dashboard de comparaison (livrable principal)
│   └── eda_cache/            # Stats pré-calculées pour le déploiement
│       ├── stats.json
│       └── samples/          # 30 images échantillon
│
├── api/                      # API FastAPI (hérité du projet initial)
│   └── main.py
│
├── app/                      # App Streamlit de prédiction (hérité du projet initial)
│   └── streamlit_app.py
│
├── tests/
│   └── test_dashboard_unit.py    # 12 tests unitaires
│
├── precompute_eda.py         # Pré-calcul des stats EDA (à lancer en local)
├── start.sh                  # Lance API + dashboard
├── requirements.txt
├── .replit
├── .github/workflows/ci.yml  # Lint flake8 + tests pytest
└── .gitignore
```

---

## Pipeline d'entraînement

### Protocole commun

Tous les modèles avec transfer learning suivent le même protocole en 2 phases :

| Phase | Encodeur | Époques max | LR   | Objectif                                  |
| ----- | -------- | ----------- | ---- | ----------------------------------------- |
| 1     | Gelé     | 50          | 1e-3 | La tête s'adapte aux features du backbone |
| 2     | Dégelé   | 30          | 1e-4 | Fine-tuning de l'ensemble du réseau       |

Pour EoMT (PyTorch) : AdamW avec LR différentiel (encodeur = 0.1×LR tête), gradient accumulation sur 2 steps, CosineAnnealingWarmRestarts au lieu de ReduceLROnPlateau.

### Loss

Cross-Entropy + 0.5 × Dice, avec masquage des pixels `IGNORE_LABEL`. Le coefficient λ=0.5 a été validé comme supérieur à la CE seule lors du projet initial.

### Callbacks

- `ModelCheckpoint` : sauvegarde le meilleur modèle sur `val_mIoU`
- `EarlyStopping` : patience 8–10 époques
- `ReduceLROnPlateau` : factor 0.5, patience 4

### Augmentation (Albumentations)

Flip horizontal, rotation légère, ShiftScaleRotate, luminosité/contraste, bruit gaussien, flou. Appliquée conjointement sur l'image et le masque.

### Reproductibilité

Seed fixée à 42 pour tous les runs. Session Keras réinitialisée entre chaque entraînement. Chaque run sauvegarde automatiquement `summary.json`, `history.json`, `pred_grid.png`, `loss.png`, `miou.png`.

---

## Dashboard (livrable principal)

Le dashboard Streamlit (`app_dashboard/streamlit_app.py`) est le livrable central de ce projet. Il est déployé sur Replit.

### Page EDA

Présentation du dataset : exemples d'images + masques par split et par ville, comptages pixels par classe, visualisation des transformations (equalisation, flou, augmentations).

Fonctionne en **mode cache** (stats pré-calculées dans `eda_cache/`) ou en **mode live** (avec le dataset local).

### Page Comparaison

Charge les résultats depuis `out/experiments/` (summary.json, history.json, images) et permet de :

- Filtrer et trier les 27 runs par encodeur, résolution, mode, score
- Afficher un bar chart des top N, scatter encodeur vs val_mIoU, boxplot par groupe
- Tracer les courbes de loss et mIoU de n'importe quel run (Plotly interactif)
- Voir les grilles de prédictions et les courbes sauvegardées de chaque run

### Page À propos

Méthodologie, modèles utilisés, palette des 8 classes, critères WCAG couverts.

### Accessibilité WCAG

Mode accessible activable dans la sidebar : police 18px, hachures sur les barres (1.4.1), bordures scatter (1.4.11), palette haut contraste (1.4.3), focus clavier orange (2.4.7), descriptions textuelles (1.1.1).

---

## Environnements

Le projet utilise deux environnements virtuels séparés :

| Environnement | Framework                                    | Modèles concernés                             | Python |
| ------------- | -------------------------------------------- | --------------------------------------------- | ------ |
| `.env0`       | TensorFlow / Keras                           | Scratch, VGG16, ResNet50, ConvNeXt, SegFormer | 3.12   |
| `.env_eomt`   | PyTorch 2.0.1+cu117 + HF transformers 4.44.2 | EoMT / DINOv2                                 | 3.11   |

La GTX 1080 Ti (sm_61, Pascal) impose PyTorch cu117. Les versions cu12x ne prennent plus en charge cette génération de GPU.

---

## Installation

### Prérequis

- Python 3.10+
- GPU recommandé (entraînements réalisés sur NVIDIA GTX 1080 Ti, 11 Go VRAM)
- TensorFlow 2.x et/ou PyTorch 2.0.1+cu117

### Dépendances

```bash
pip install -r requirements.txt
```

Pour EoMT/DINOv2 (dans l'environnement `.env_eomt`) :

```bash
pip install torch==2.0.1+cu117 torchvision==0.15.2+cu117 --index-url https://download.pytorch.org/whl/cu117
pip install transformers==4.44.2
```

### Données

Télécharger Cityscapes (`leftImg8bit_trainvaltest.zip` et `gtFine_trainvaltest.zip`) depuis le [site officiel](https://www.cityscapes-dataset.com/) et extraire dans :

```
data/raw/cityscapes/leftImg8bit/
data/raw/cityscapes/gtFine/
```

### Lancer le dashboard

```bash
streamlit run app_dashboard/streamlit_app.py
```

Ou avec l'API (projet initial) :

```bash
bash start.sh
```

### Pré-calculer le cache EDA (avant déploiement)

```bash
python precompute_eda.py
```

Ce script scanne le dataset et sauvegarde les stats dans `app_dashboard/eda_cache/`, ce qui permet au dashboard de fonctionner sans le dataset de ~10 Go sur le cloud.

---

## GPU & contraintes matérielles

Tous les entraînements ont été réalisés sur une **NVIDIA GTX 1080 Ti** (11 Go VRAM, compute capability sm_61). Cette contrainte impose :

- Les variantes légères des modèles (ConvNeXt Tiny, MiT-B0, DINOv2-Base)
- La résolution 256×256 pour les entraînements longs (512 possible en quickbench)
- PyTorch cu117 (cu12x ne supporte plus sm_61)
- Pas de mixed precision (AMP non disponible sur génération gpu Pascal)
