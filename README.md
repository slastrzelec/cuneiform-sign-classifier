# Cuneiform Sign Classifier

[![tests](https://github.com/slastrzelec/cuneiform-sign-classifier/actions/workflows/ci.yml/badge.svg)](https://github.com/slastrzelec/cuneiform-sign-classifier/actions/workflows/ci.yml)

Classification of individual cuneiform signs (Sumerian/Akkadian) on images of
3D-rendered clay tablets, using transfer learning. A portfolio project
demonstrating a full ML workflow: exploring a niche, challenging dataset,
methodologically sound training, and deep error analysis connecting model
results with domain knowledge (paleography).

![t-SNE of the model's feature space, coloured by historical period](eda_outputs/embeddings_period_drift.png)

## Why this project

Cuneiform is one of the oldest writing systems in the world and still an
active area of digital humanities research — a recent (2026) paper describes
an end-to-end OCR pipeline for this script
([arXiv:2606.22608](https://arxiv.org/abs/2606.22608)), which served as a
reference point. Instead of another cats-vs-dogs classifier, this project
tackles real challenges: a niche, imbalanced dataset, inconsistencies in the
metadata, and domain-specific data augmentation.

## Results at a glance

| Metric | Result |
|---|---|
| Number of classes | Top 30 most frequent signs |
| Test set | 1,699 sign images, grouped by tablet |
| Test accuracy | 90.4% (95 % Wilson CI 88.9–91.7 %) |
| Test macro-F1 | 0.896 |
| Model | ResNet18 (transfer learning), CPU-only |
| Dataset | [MaiCuBeDa Hilprecht](https://doi.org/10.11588/DATA/QSNIQ2) |

The size of the test set and the confidence interval are printed by `python evaluate.py`;
the per-class analysis is in [Detailed Results](#detailed-results) below. The test split is
grouped by tablet (see Methodology). Model selection (which checkpoint to keep) used the
validation macro-F1 only; the test set was not used for tuning. It is a single grouped split,
so the interval reflects sampling noise on these 1,699 images, not variation between
different splits.

## Dataset

**MaiCuBeDa Hilprecht** (Mainz Cuneiform Benchmark Dataset), Homburg & Mara
2023, released under **CC BY-SA 4.0**. Contains ~28.7k annotated individual
cuneiform signs, rendered from 3D models of tablets from the Hilprecht
Collection (MSII rendering — a curvature-based filter, reported in the
literature to work better for sign detection/classification than plain
lighting).

**Realistic scope:** instead of full OCR of entire texts (a PhD-scale
problem), this project focuses on **classifying a single, already-cropped
sign** — the top 30 most frequent classes, selected based on the actual
distribution in the data (each class has at least 203 examples after
filtering, covering ~44% of the full annotation set).

### Data quality issue encountered

The `Filename` field in the metadata CSV (`translitmetadata.csv`) **does not
match the actual filenames** in the published image archive (a different
number of fields — likely a dataset versioning artifact). Solution: instead
of matching by the filename from the CSV, `build_dataset.py` **parses
filenames directly from disk** and maps each sign's transliteration to its
class name via a separate lookup table built from the CSV (923 unique
readings, only 18 ambiguous, resolved by majority vote).

## Methodology

- **Train/val/test split (70/15/15) grouped by tablet**, not by individual
  image — the same sign from the same tablet never ends up in two splits at
  once. The splitting code is unit-tested on synthetic tablets (no overlap, every tablet
  assigned once, deterministic for a seed), and the manifest of the real split is
  checked by `tests/test_no_leakage.py` when the dataset is present locally.
- **Class weights** (inverse frequency) in the loss function — moderate
  imbalance (~5x between the most and least frequent of the top-30 classes).
- **Model selected by macro-F1 on val, not accuracy** — to avoid favoring
  frequent classes.
- **Augmentation without mirror flips.** A standard `RandomHorizontalFlip`
  would be a methodological error here — mirroring a cuneiform sign changes
  its identity. Only small rotation (±8°) and light brightness/contrast
  jitter were used.
- **Transfer learning with partial freezing** — ResNet18's `conv1`/`layer1`/`layer2`
  receive no gradient updates (general low-level features), `layer3`/`layer4`/`fc`
  are trained. A quality/training-time trade-off for CPU training. "Frozen" means no
  gradient updates: the BatchNorm running statistics of those layers still adapt in
  train mode.

## Detailed Results

### Confusion matrix

![Confusion matrix](eda_outputs/confusion_matrix.png)

A clear diagonal — the model is systematically correct, with errors that are
rare and concentrated. The top confusions (`LUGAL→LU2`, `A↔MIN_(2)`) are
explainable by visual similarity of the signs' graphical components, not
random — confirmed geometrically in the embedding analysis below.

### Discovery: paleographic drift of the sign `U`

EDA on sample images revealed that the sign `U` (the number "10") has a
**physically different form** depending on the historical period: a round
indentation in the oldest tablets (ED IIIa/b, ~2500-2600 BC) vs. a clear
triangular wedge in later periods. This is a documented paleographic
phenomenon — early cuneiform partly used a round stylus for writing numbers.

**Indicative, from a small sample** (`evaluate.py`, accuracy per period):

| Period | Accuracy for `U` |
|---|---|
| Ur III, Old Assyrian, Old Babylonian, Old Akkadian, Early OB | 100% |
| **ED IIIb (ca. 2500-2340 BC)** | **43% (3/7; 95 % Wilson CI 16-75 %)** |

**Verified geometrically** (`embeddings.py`, t-SNE on 512-dim feature
vectors): the sign `U` forms **two completely separate clusters** in the
model's feature space, corresponding to the two graphical variants — whereas,
e.g., `ASZ` (100% accuracy regardless of period) forms a single coherent
cluster. The model "sees" the same discrepancy a human notices visually.

With only 7 ED IIIb examples in the test set the interval is wide: the accuracy drop is a strong hint, not a
proof, and the t-SNE picture is the stronger evidence. The likely cause is the imbalance in the
training data (64% of it is Ur III) rather than an architectural flaw — a hypothesis that
would need more ED IIIb examples to confirm.

### Interpretability (Grad-CAM)

The demo app shows a Grad-CAM overlay for every prediction. It is a qualitative
sanity check — on the examples I looked at, the highlighted regions lie on the sign
rather than on the clay background — not a quantitative proof that the model never
uses background cues.

## Demo (Streamlit)

An interactive app: pick an example from the test gallery → top-1 prediction
+ top-3 with confidence → a warning on low confidence → Grad-CAM
visualization.

```bash
streamlit run app.py
```

**Live demo:** https://cuneiform-sign-classifier.streamlit.app/

### How the app gets its model and images

The trained checkpoint (`checkpoints/best_model.pt`, ~123 MB) and the full test
set (`data/processed/`) are gitignored — too large, and regenerable. So that the
app runs from a fresh clone (which is what Streamlit Community Cloud does):

- a small curated gallery of test images is committed under `demo_samples/`
  (~120 images, a handful per class); `app.py` falls back to it whenever
  `data/processed/test` is not present;
- the checkpoint is fetched at runtime from the **Hugging Face Hub**
  (`HF_REPO_ID` in `app.py`; `HF_REVISION` can pin it to an exact commit of the
  model repo).

**Safe loading.** A PyTorch checkpoint is a pickle file, and loading an untrusted
pickle can run arbitrary code. Every `torch.load` in this project uses
`weights_only=True` (a restricted unpickler that accepts only tensors and plain
containers), so a tampered checkpoint is rejected instead of executed. A test
builds a malicious checkpoint and checks that its payload does not run.

To use your own checkpoint: upload `checkpoints/best_model.pt` to a model repo on
the Hub, set `HF_REPO_ID` (and optionally `HF_REVISION`) in `app.py`, then deploy
the repo on [share.streamlit.io](https://share.streamlit.io) with `app.py` as the
main file. The first load downloads the file; `st.cache_resource` keeps the model in
memory afterwards.

## Testing

**46 automated tests** (`pytest`, plus 8 optional dataset checks), run on every push in GitHub Actions together with a
`ruff` lint check; a separate informational job runs `pip-audit` on the pinned
dependencies. The suite uses only **synthetic data** — no dataset, no trained
checkpoint, no network — so `pytest` works on a fresh clone in about ten seconds.

| Area | What is verified |
|---|---|
| Train/val/test split | `group_split`: no tablet in two splits, every tablet assigned exactly once, proportions close to 70/15/15, deterministic for a seed; filename parser incl. readings with underscores |
| Augmentation | the training pipeline contains **no mirror flips** (a flipped sign is a different sign) and only a small rotation; the evaluation pipeline is deterministic; output shape and normalisation |
| Class weighting | inverse-frequency weights on a synthetic dataset; train/val/test must share one class-to-index mapping, otherwise loading fails |
| Model | output shape; only `layer3`/`layer4`/`fc` are trainable and the frozen layers do not change after an optimiser step |
| Grad-CAM | map shape and [0, 1] range on a random-weight model, different target classes give different maps, hooks are removed after each prediction (no leak on the cached model) |
| Checkpoint loading | save → load gives identical logits; missing keys and a wrong class count are rejected; a **malicious pickle is rejected and its payload does not run** (with a control showing that the unrestricted loader would run it) |
| App (Streamlit `AppTest`) | renders, lists every gallery class, shows top-3 with confidences, shows the low-confidence warning, re-runs after changing the example |
| Statistics | Wilson interval helper (known value, bounds, narrower with more data, invalid input) |
| Repository hygiene | no model binaries or dataset files tracked by git, no hard-coded local paths, no `weights_only=False` |

I checked that the tests can fail: re-introducing a flip, unfreezing `layer1`,
switching the loader to `weights_only=False` or leaving the Grad-CAM hooks attached
each turns the relevant test red.

**What the tests do not cover:** the accuracy of the real trained model (it needs the
dataset and the checkpoint — the numbers above come from `python evaluate.py` run
locally), the visual quality of Grad-CAM, the download from the Hugging Face Hub, and
the Docker image build. The 8 manifest checks in `tests/test_no_leakage.py` run only
when `data/processed/manifest.csv` exists locally; on a clean clone they are reported as
skipped, not passed.

## Project structure

```
cuneiform-sign-classifier/
├── src/
│   ├── data.py              # DataLoaders, augmentation, class weights
│   ├── model.py             # ResNet18 + freeze policy
│   ├── gradcam.py           # Grad-CAM and heat-map overlay
│   ├── inference.py         # safe checkpoint loading, single-image prediction
│   └── metrics.py           # Wilson interval
├── tests/                   # synthetic-data tests (see Testing)
├── build_dataset.py         # Dataset parsing, top-N filtering, grouped split
├── eda.py                   # Class/period distribution, image size, samples
├── train.py                 # Training (transfer learning, resumable)
├── evaluate.py              # Test metrics + CI, confusion matrix, per-period analysis
├── embeddings.py            # t-SNE visualization of the feature space
├── app.py                   # Streamlit demo + Grad-CAM
├── demo_samples/            # small CC BY-SA gallery for the demo (own LICENSE)
├── eda_outputs/             # Generated plots (kept in repo for README)
├── requirements*.txt        # runtime / training / dev dependencies
├── Dockerfile, .dockerignore
└── .github/                 # CI and Dependabot
```

## Reproducing the project from scratch

```bash
conda create -n TABL python=3.11 -y
conda activate TABL
pip install -r requirements-dev.txt        # CPU PyTorch, Streamlit, pytest, ruff

# 0. Run the tests (no data needed)
pytest

# 1. Download MaiCuBeDa Hilprecht: https://doi.org/10.11588/DATA/QSNIQ2
#    (translitmetadata.csv -> files/, plus one of the image zips, e.g. MSII)
#    Paths default to the project folder; override with
#    CUNEIFORM_TRANSLIT_CSV / CUNEIFORM_CHAR_IMAGES / CUNEIFORM_OUTPUT_DIR
# 2. Build the dataset
python build_dataset.py
# 3. Verify the manifest of the real split (optional extra checks)
pytest tests/test_no_leakage.py -v
# 4. EDA (optional)
python eda.py
# 5. Training (resumable - safe to interrupt and re-run)
python train.py
# 6. Evaluation (prints n and a 95 % interval) and error analysis
python evaluate.py
python embeddings.py
# 7. Demo
streamlit run app.py
```

### Docker

The image contains the app and the demo gallery; the checkpoint is downloaded from the
Hugging Face Hub on first start, so it builds from a fresh clone and runs as a non-root
user. (The image build is not part of the CI checks.)

```bash
docker build -t cuneiform-sign-classifier .
docker run -p 8501:8501 cuneiform-sign-classifier
```

## Limitations and future directions

- **Only the top-30 classes** — the full cuneiform sign inventory has
  hundreds of signs with a strong long tail (93 classes have only 1 example
  in the entire dataset). Extending this would require either few-shot
  learning or oversampling rare classes.
- **Imbalance across historical periods** (64% Ur III) limits generalization
  to older graphical variants — directly documented via the sign `U`.
- **Classification, not detection** — the model assumes the sign has already
  been cropped from the tablet. A natural extension: a detection +
  classification pipeline on the full tablet (cf. eBL, arXiv:2606.22608).
- **The test split is a single grouped split** (one seed): the reported numbers carry
  split-to-split variance that a repeated or cross-validated grouped split would quantify.
- The **MSII** rendering was used for training; the dataset also provides
  `VirtualLight` — whether combining renderings would improve generalization
  remains untested.

## Citation / data sources

- Homburg, T., Mara, H. (2023). *MaiCuBeDa Hilprecht — Mainz Cuneiform
  Benchmark Dataset*. Heidelberg University. CC BY-SA 4.0.
  https://doi.org/10.11588/DATA/QSNIQ2
- Mara, H. (2019). *HeiCuBeDa Hilprecht*. https://doi.org/10.11588/data/IE8CCN
- Automated sign detection across the Electronic Babylonian Library (2026).
  arXiv:2606.22608

## License

- **Code:** MIT, see [LICENSE](LICENSE).
- **Data and demo images:** the MaiCuBeDa dataset is CC BY-SA 4.0 (Homburg & Mara 2023). The
  sign crops in `demo_samples/` are a selection from it and stay under CC BY-SA 4.0 — see
  [demo_samples/LICENSE](demo_samples/LICENSE) for attribution.
- **Trained weights:** derived from CC BY-SA 4.0 data, so they are shared under the same terms
  (attribution, share-alike).

## Author

Sławomir Strzelec — AI/ML Engineer & Data Scientist, Kraków
[Portfolio](https://slastrzelec.github.io/portfolio/) ·
[GitHub](https://github.com/slastrzelec) ·
[LinkedIn](https://linkedin.com/in/sławomir-strzelec)
