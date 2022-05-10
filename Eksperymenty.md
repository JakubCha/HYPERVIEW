# Experiments

## 28.03.2022

 * Wygenerowano wartości średnie oraz odchylenia standardowe dla wszystkich kanałów na podstawie zbioru treningowego. Wartości te zostały wykorzystane do znormalizowania obrazów hiperspektralnych.
 * Stworzono kod w pytorch do eksperymentów (HyperView_Torch_DITRY.ipynb) 
 * Przetestowano optymalizatory `SGD`, `Adam`, `AdamW`. Najlepsze wyniki otrzymano z `AdamW`, model był w stanie przeuczyć się na pojedynczym batchu z tym optymalizatorem i wartością `learning-rate = 0.01`.
 * Wygenerowano wynik dla **ResNet34** z wykorzystaniem modelu z torchvision. W Weights&Biases jest to model *valiant-surf-10* i otrzymany wynik dla zbioru testowego to **0.941**.

***
## 01.04.2022

 * Wygenerowano wynik dla **ResNext50** z wykorzystaniem modelu z torchvision. W Weights&Biases jest to model *atomic-cloud-11* i otrzymany wynik dla zbioru testowego to **0.9755**.
 * Wygenerowano wynik dla **ResNext100** z wykorzystaniem modelu z torchvision. W Weights&Biases jest to model *light-oath-12* i otrzymany wynik dla zbioru testowego to **0.9702**.
 * Wygenerowano wynik dla **ResNet50** z wykorzystaniem modelu z torchvision. W Weights&Biases jest to model *peach-resonance-14* i otrzymany wynik dla zbioru testowego to **1.00205**.
 
***
## 02.04.2022

 * **Znaleziono błąd w przetwarzaniu danych w torch.dataset**. Nie odwracano maski z numpy co powodowało błędne maskowanie danych wejściowych. Błąd naprawiono.
 * Wygenerowano wynik dla **ResNet50** z wykorzystaniem modelu z torchvision. W Weights&Biases jest to model *swift-eon-15* i otrzymany wynik dla zbioru testowego to **0.933795**.
 
## 03.04.2022

 * Wygenerowano wynik dla **ResNet101** z wykorzystaniem modelu z torchvision. W Weights&Biases jest to model *leafy-oath-19* i otrzymany wynik dla zbioru testowego to **...**.
 
## 07.05.2022

 * Wykonanie analizy błędów najlepszego modelu **ResNet50**. Wyniki znajdują się w /plots/ResNet50_4classes_07_05_2022
 
 ## 08.05.2022 i 09.05.2022
 
 * Trening modelu **ResNet50** oddzielnie dla każdego parametru. Proporcje trening/walidacja dla parametrów **P, K, Mg** wyniosły 1400/332, natomiast dla parametru **pH** 1000/732. Zauważono, że **ResNet50** szybko zbiega dla danych treningowych na parametrze **pH** ale jakość na danych walidacyjnych jest niska (`validation_loss` ma peaki wysokich wartości). Dla parametrów **P, K, Mg** błąd treningu jest dużo wyższy (zbiega ale bardzo powoli), na danych walidacyjnych błąd praktycznie nie spada.
 * Model **ResNet50**, parametr: **pH** --> W&B: `soft-tree-32`
 * Model **ResNet50**, parametr: **Mg** --> W&B: `smooth-frog-30`
 * Model **ResNet50**, parametr: **K** --> W&B: `denim-dawn-29`
 * Model **ResNet50**, parametr: **P** --> W&B: `young-forest-28`

 ## 10.05.2022
 * Stworzono notebook do predykcji na danych testowych z modeli pojedynczych parametrów (08 i 09.05.2022): ``HyperView_Torch_PredictTest_1ParamModels_ResNets_10_05_2022.ipynb``
 * Uzyskany wynik: ``ResNet50_10052022_1ParamModels``: **0.94449**
 
# Eksperymenty do wykonania

 * Skalowanie predykowanych wartości P, K, Mg, pH. Przy liczeniu MSE (i logowaniu do W&B) wartości predykowane i rzeczywiste powinny mieć wartości po inwersji skalowania (do bazowych wartości). Pytania: Parametry skalowania powinny zostać obliczone na całym zbiorze treningowym czy tylko na danych treningowych po wydzieleniu zbioru trening/walidacja? (Jeżeli druga opcja to trzeba podzielić zbiór treningowy na trening/walidację) 
 
 * Wykorzystanie RMSLELoss do przy wysokich wartościach (nieskalowanych) parametrów gleby: P, K, Mg
 * Wypróbowanie modeli z biblioteki timm (https://github.com/rwightman/pytorch-image-models) oraz modelu DeiT (https://github.com/facebookresearch/deit/blob/main/models.py). Biblioteka timm daje duże możliwości customizacji - warto przeczytać https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#9388 oraz  https://fastai.github.io/timmdocs/models#So-how-is-timm-able-to-load-these-weights? 
 * Wykonanie crop np. 5 metrów środkowych. Domyślamy się, że zazwyczaj w tym obszarze były wykonane pomiary terenowe. Dodatkowo, może to zmniejszyć ilośc danych do trenowania (?szum informacyjny?). Mozna to uzyskać z wykorzystaniem scipy.ndimage.center_of_mass na masce numpy.
 * 
 
 
 