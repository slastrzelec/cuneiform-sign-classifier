# Cuneiform Sign Classifier

Klasyfikacja pojedynczych znaków pisma klinowego (sumeryjskiego/akadyjskiego) na
zdjęciach 3D-renderowanych tabliczek glinianych, z wykorzystaniem transfer
learningu. Projekt portfolio pokazujący pełny cykl pracy ML: od eksploracji
niszowego, trudnego datasetu, przez metodologicznie poprawny trening, po
głęboką analizę błędów łączącą wyniki modelu z wiedzą domenową (paleografią).

![Grad-CAM demo](eda_outputs/embeddings_period_drift.png)

## Dlaczego ten projekt

Pismo klinowe to jedna z najstarszych form pisma na świecie i wciąż aktywny
obszar badań cyfrowej humanistyki — istnieje świeży (2026) paper opisujący
end-to-end pipeline OCR dla tego pisma
([arXiv:2606.22608](https://arxiv.org/abs/2606.22608)), który posłużył jako
punkt odniesienia. Zamiast kolejnego klasyfikatora kotów i psów, ten projekt
mierzy się z realnymi wyzwaniami: niszowym, niezbalansowanym datasetem,
niespójnościami w metadanych i domenowo specyficzną augmentacją danych.

## Wyniki w skrócie

| Metryka | Wynik |
|---|---|
| Liczba klas | 30 najczęstszych znaków |
| Test accuracy | 90,4% |
| Test macro-F1 | 0,896 |
| Model | ResNet18 (transfer learning), CPU-only |
| Dataset | [MaiCuBeDa Hilprecht](https://doi.org/10.11588/DATA/QSNIQ2) |

Pełny `classification_report` per-klasa w `evaluate.py` / sekcja
[Wyniki](#wyniki-szczegółowe) poniżej.

## Dataset

**MaiCuBeDa Hilprecht** (Mainz Cuneiform Benchmark Dataset), Homburg & Mara
2023, udostępniony na licencji **CC BY-SA 4.0**. Zawiera ~28,7 tys.
anotowanych pojedynczych znaków klinowych, wyrenderowanych z modeli 3D
tabliczek Kolekcji Hilprechta (technika renderingu MSII — filtr krzywizny,
lepszy do detekcji/klasyfikacji znaków niż zwykłe oświetlenie wg literatury).

**Realistyczny zakres problemu:** zamiast pełnego OCR całych tekstów (temat
na doktorat), projekt skupia się na **klasyfikacji pojedynczego, już
wyciętego znaku** — top-30 najczęstszych klas, wybranych na podstawie
rzeczywistego rozkładu w danych (każda klasa ma min. 203 przykłady po
przefiltrowaniu, pokrycie ~44% całego zbioru anotacji).

### Napotkany problem jakości danych

Pole `Filename` w metadanych CSV (`translitmetadata.csv`) **nie odpowiada
rzeczywistym nazwom plików** w opublikowanym archiwum obrazów (inny schemat
liczby pól — prawdopodobnie artefakt wersjonowania datasetu). Rozwiązanie:
zamiast dopasowywać po nazwie pliku z CSV, `build_dataset.py` **parsuje nazwy
plików bezpośrednio z dysku** i mapuje transliterację znaku na nazwę klasy
przez osobną tabelę zbudowaną z CSV (923 unikalne odczyty, tylko 18
niejednoznacznych, rozwiązanych większością głosów).

## Metodologia

- **Podział train/val/test (70/15/15) grupowany po tabliczce**, nie po
  pojedynczym obrazie — ten sam znak z tej samej tabliczki nigdy nie trafia
  jednocześnie do dwóch zbiorów. Zweryfikowane testami jednostkowymi
  (`tests/test_no_leakage.py`, 8 testów, w tym explicit sprawdzenie braku
  nakładania się zbiorów tabliczek).
- **Class weights** (odwrotność częstości) w funkcji straty — umiarkowany
  imbalance (~5x między najczęstszą a najrzadszą z top-30 klas).
- **Model wybierany wg macro-F1 na val, nie accuracy** — żeby uniknąć
  faworyzowania częstych klas.
- **Augmentacja bez odbić lustrzanych.** Standardowe `RandomHorizontalFlip`
  byłoby tu błędem metodologicznym — odbicie znaku klinowego zmienia jego
  tożsamość. Użyto tylko małej rotacji (±8°) i lekkiego jitteru
  jasności/kontrastu.
- **Transfer learning z częściowym zamrożeniem** — `layer1`/`layer2`
  ResNet18 zamrożone (ogólne cechy niskiego poziomu), `layer3`/`layer4`/`fc`
  trenowane. Kompromis jakość/czas treningu na CPU.

## Wyniki szczegółowe

### Confusion matrix

![Confusion matrix](eda_outputs/confusion_matrix.png)

Wyraźna przekątna — model systematycznie poprawny, błędy nieliczne i
skoncentrowane. Top pomyłki (`LUGAL→LU2`, `A↔MIN_(2)`) są wytłumaczalne
wizualnym podobieństwem komponentów graficznych znaków, nie są losowe —
potwierdzone geometrycznie w analizie embeddingów poniżej.

### Odkrycie: dryf paleograficzny znaku `U`

EDA na przykładowych obrazach ujawniło, że znak `U` (liczba "10") ma
**fizycznie różną formę** w zależności od okresu historycznego: okrągłe
wgłębienie w najstarszych tabliczkach (ED IIIa/b, ~2500-2600 BC) vs wyraźny
trójkątny klin w młodszych okresach. To udokumentowane zjawisko
paleograficzne — wczesne pismo klinowe częściowo używało okrągłego rylca do
zapisu liczb.

**Zweryfikowane ilościowo** (`evaluate.py`, accuracy per okres):

| Okres | Accuracy dla `U` |
|---|---|
| Ur III, Old Assyrian, Old Babylonian, Old Akkadian, Early OB | 100% |
| **ED IIIb (ca. 2500-2340 BC)** | **43% (3/7)** |

**Zweryfikowane geometrycznie** (`embeddings.py`, t-SNE na wektorach cech
512-wym.): znak `U` tworzy **dwa całkowicie oddzielone skupiska** w
przestrzeni cech modelu, odpowiadające dwóm wariantom graficznym — podczas
gdy np. `ASZ` (100% accuracy niezależnie od okresu) tworzy jedno spójne
skupisko. Model "widzi" ten sam błąd, który człowiek widzi na oko.

To pokazuje ograniczenie modelu wynikające wprost z niezbalansowania
danych treningowych (64% danych to Ur III) — nie z wady architektury.

### Interpretowalność (Grad-CAM)

Aplikacja demo wizualizuje Grad-CAM dla każdej predykcji — potwierdza, że
model opiera decyzje na samym znaku (nie na teksturze gliny w tle).

## Demo (Streamlit)

Interaktywna aplikacja: wybór przykładu z galerii testowej → predykcja top-1
+ top-3 z pewnością → ostrzeżenie przy niskiej pewności → wizualizacja
Grad-CAM.

```bash
streamlit run app.py
```

lub przez Docker (patrz niżej).

## Struktura projektu

```
cuneiform-sign-classifier/
├── src/
│   ├── __init__.py
│   └── data.py              # DataLoadery, augmentacja, class weights
├── tests/
│   └── test_no_leakage.py   # 8 testów integralności datasetu
├── build_dataset.py          # Parsowanie datasetu, filtrowanie top-N, split
├── eda.py                    # Rozkład klas/okresów, wymiary obrazów, próbki
├── train.py                  # Trening (transfer learning, wznawianie)
├── evaluate.py                # Confusion matrix, analiza błędów per-okres
├── embeddings.py              # Wizualizacja t-SNE przestrzeni cech
├── app.py                     # Demo Streamlit + Grad-CAM
├── Dockerfile / .dockerignore
├── requirements.txt
└── eda_outputs/                # Wygenerowane wykresy (w repo, do README)
```

## Odtworzenie projektu od zera

```bash
conda create -n TABL python=3.11 -y
conda activate TABL
conda install pytorch torchvision cpuonly -c pytorch -y
conda install -c conda-forge pandas numpy pillow matplotlib jupyter scikit-learn seaborn tqdm -y
pip install pytest streamlit

# 1. Pobierz MaiCuBeDa Hilprecht: https://doi.org/10.11588/DATA/QSNIQ2
#    (translitmetadata.csv + jeden z zipow z obrazami, np. MSII)
# 2. Zbuduj dataset (dostosuj sciezki w CONFIG na gorze pliku)
python build_dataset.py
# 3. Zweryfikuj brak wycieku danych
pytest tests/test_no_leakage.py -v
# 4. EDA (opcjonalnie)
python eda.py
# 5. Trening (wznawialny - bezpiecznie przerwac i uruchomic ponownie)
python train.py
# 6. Ewaluacja i analiza bledow
python evaluate.py
python embeddings.py
# 7. Demo
streamlit run app.py
```

### Docker

**Uwaga:** budowanie obrazu wymaga lokalnie już wytrenowanego
`checkpoints/best_model.pt` (checkpoint nie jest częścią repozytorium —
patrz `.gitignore` — więc najpierw przejdź przez krok 5 powyżej, `python
train.py`).

```bash
docker build -t cuneiform-sign-classifier .
docker run -p 8501:8501 cuneiform-sign-classifier
```

## Ograniczenia i dalsze kierunki

- **Tylko top-30 klas** — pełny sylabariusz klinowy ma setki znaków z silnym
  long-tail (93 klasy z tylko 1 przykładem w całym datasecie). Rozszerzenie
  wymagałoby albo few-shot learningu, albo oversamplingu rzadkich klas.
- **Niezbalansowanie okresów historycznych** (64% Ur III) ogranicza
  generalizację na starsze warianty graficzne — udokumentowane wprost na
  przykładzie znaku `U`.
- **Klasyfikacja, nie detekcja** — model zakłada, że znak jest już wycięty z
  tabliczki. Naturalne rozszerzenie: pipeline detekcji + klasyfikacji na
  całej tabliczce (por. eBL, arXiv:2606.22608).
- Rendering **MSII** użyty do treningu; dataset udostępnia też
  `VirtualLight` — nieprzetestowane pytanie, czy łączenie renderingów
  poprawiłoby generalizację.

## Cytowanie / źródła danych

- Homburg, T., Mara, H. (2023). *MaiCuBeDa Hilprecht — Mainz Cuneiform
  Benchmark Dataset*. Heidelberg University. CC BY-SA 4.0.
  https://doi.org/10.11588/DATA/QSNIQ2
- Mara, H. (2019). *HeiCuBeDa Hilprecht*. https://doi.org/10.11588/data/IE8CCN
- Automated sign detection across the Electronic Babylonian Library (2026).
  arXiv:2606.22608

## Autor

Sławomir Strzelec — AI/ML Engineer & Data Scientist, Kraków
[Portfolio](https://slastrzelec.github.io/portfolio/) ·
[GitHub](https://github.com/slastrzelec) ·
[LinkedIn](https://linkedin.com/in/sławomir-strzelec)
