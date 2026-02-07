\# Porównanie skuteczności lekkich sieci CNN w zadaniu klasyfikacji



Projekt zaliczeniowy z widzenia komputerowego: zaprojektowanie, wytrenowanie i porównanie konwolucyjnych sieci neuronowych (CNN) dla klasyfikacji wieloklasowej, z uwzględnieniem danych rzeczywistych i syntetycznych oraz różnych strategii augmentacji.



\## Cel

\- Porównać 2 architektury CNN:

&nbsp; - \*\*EfficientNet-B0\*\*

&nbsp; - \*\*ShuffleNetV2 x1.0\*\*

\- Zbadać wpływ:

&nbsp; - rodzaju i liczebności danych treningowych (\*\*E1–E4\*\*),

&nbsp; - augmentacji (\*\*A1–A4\*\*),

\- Ocenić modele na stałym zbiorze testowym przy użyciu metryk:

&nbsp; - \*\*accuracy\*\*

&nbsp; - \*\*macro-F1\*\*

&nbsp; - \*\*macierzy pomyłek (confusion matrix)\*\*



\## Dane i podział

Zbiór danych zawiera \*\*220 obrazów\*\* podzielonych na \*\*5 klas\*\*.



Stały zbiór walidacyjno-testowy:

\- \*\*20 obrazów rzeczywistych\*\* (po \*\*4 na klasę\*\*) – używany identycznie we wszystkich eksperymentach (\*\*nie trafia do treningu\*\*).



Pula treningowa (razem 200 obrazów):

\- dane rzeczywiste i syntetyczne zgodnie ze scenariuszem eksperymentu.



\### Scenariusze treningowe (E1–E4)

\- \*\*E1\*\* – wszystkie dostępne dane treningowe (mix real + synthetic) — `train\_len = 200`

\- \*\*E2\*\* – tylko dane syntetyczne — `train\_len = 130`

\- \*\*E3\*\* – tylko dane rzeczywiste — `train\_len = 70`

\- \*\*E4\*\* – wyrównane (bez miksu): trening na typie liczniejszym, przyciętym do liczności typu mniej licznego — `train\_len = 70`



Test (stały):

\- `test\_len = 20` dla wszystkich E1–E4



> Uwaga: Zbiór testowy ma tylko 20 próbek, więc pojedyncza pomyłka zmienia accuracy o 5 p.p. (wyniki mogą wykazywać większą wariancję).



\## Augmentacje (A1–A4)

Dla każdego scenariusza E1–E4 wykonano trening dla czterech wariantów augmentacji:

\- \*\*A1\*\* – brak / minimalna augmentacja (baseline)

\- \*\*A2\*\* – umiarkowana augmentacja

\- \*\*A3\*\* – umiarkowana augmentacja (wariant)

\- \*\*A4\*\* – agresywna augmentacja



Szczegółowa definicja transformacji znajduje się w kodzie (moduł loaderów / augmentacji).



\## Konfiguracja treningu

\- Python: \*\*3.10.11\*\*

\- Framework: \*\*PyTorch\*\*

\- Sprzęt: \*\*NVIDIA RTX 4070\*\*

\- Epoki: \*\*20\*\*

\- Batch size: \*\*128\*\*

\- LR: \*\*1e-4\*\*

\- Optymalizacje DataLoader: `pin\_memory=True`, `persistent\_workers=True`



\## Wyniki

Wyniki wszystkich 32 uruchomień (2 modele × 4 scenariusze × 4 augmentacje) zapisują się do:

\- `results/results.csv`

\- `results/results.xlsx`

\- `results/confusion\_matrices/\*.csv`



Skrypt analizy generuje wykresy i automatyczną analizę do:

\- `results/plots/\*.png`

\- `results/plots/summary.txt`

\- `results/plots/analysis.txt`



\## Struktura projektu

Przykładowa struktura katalogów:

\- `src/` – kod (trening, modele, loadery, analiza)

\- `data/prepared/` – przygotowany dataset (train\_real, train\_synthetic, test\_fixed)

\- `results/` – wyniki, macierze pomyłek

\- `venv/` – środowisko wirtualne (lokalnie)



\## Uruchomienie



\### 1) Instalacja zależności

```bash

python -m venv venv

\# Windows PowerShell:

.\\venv\\Scripts\\activate

python -m pip install -r requirements.txt





Trening pełnej siatki eksperymentów (32 runy)

python -m src.run\_all



Analiza wyników i wykresy

python -m src.analyze\_results





Reprodukowalność



Stały zbiór testowy test\_fixed jest identyczny we wszystkich eksperymentach.



Każdy run jest identyfikowany przez run\_id (model + E + A).



Dla każdego runu zapisywana jest macierz pomyłek.



Autor



Bartek Orliński (projekt zaliczeniowy – widzenie komputerowe)





