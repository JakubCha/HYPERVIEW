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
 
 ## 11.05.2022
 * Wyznaczono wartości skalujące parametry predykowane P, K, Mg, pH. Zostały one zapisane przez pickle.dump w /train_data. Parametry skalowania wyznaczono na całym zbiorze treningowym. Pytania: Parametry skalowania powinny zostać obliczone na całym zbiorze treningowym czy tylko na danych treningowych po wydzieleniu zbioru trening/walidacja? (Jeżeli druga opcja to trzeba podzielić zbiór treningowy na trening/walidację) 
 * Problemy z overfitting na pojedynczym zestawie danych. Ile epok potrzebne jest by stwierdzić że model overfituje we właściwy sposób na pojedynczym batchu?
 
 ## 12.05.2022
 * Trening modelu **ResNet50** 1Param na wyskalowanych predykowanych parametrach
 
 ## 13-14.05.2022
 * Exploratory analysis: Analiza dystrybucji parametrów względem siebie i względem wielkości obrazka (liczby niezerowych pikseli). Próba predykcji jednego parametru na podstawie pozostałych trzech z wykorzystaniem XGBoost.
 * Predykcja na modelach wyuczonych na znormalizowanych (wyskalowanych) parametrach. Uzyskany wynik: *ResNet50_12052022_1ParamScaled*: **0.94276**.
 
 ## 15.05.2022
 * Stworzono notebook do treningu na danych z RandomCrop o rozmiarze 11x11: `HyperView_Torch_1Param_Clean_ParamsScaling_RandomCrop_ResNets_14_05_2022`. W celu minimalizacji liczby pikseli o zerowej informacji okno o rozmiarze 11x11 jest losowane do momentu otrzymania co najmniej 60 pikseli niezerowych.
 * Wykonano predykcję na danych testowych z wykorzystaniem RandomCrop o rozmiarze 11x11. Wynik nosi nazwę *ResNet50_150520221*: **1.00906**
 
 ## 16.05.2022 i 17.05.2022
 * Stworzono notebook do treningu na modelu pretrenowanym na ImageNet: `HyperView_Torch_1Param_Clean_ParamsScaling_Pretrained_ResNets_16_05_2022`. Wytrenowano modele dla poszczególnych parametrów. 
 * Trening wykonano w stosunku 1000:732. Można zwiększyść ilość danych treningowych.

 ## 18.05.2022
 * Wykonano predykcję na wytrenowanych modelach. Otrzymany wynik: *ResNet50_18052022*: **0.90309**.
 
 ## 19.05.2022 i 20.05.2022
 * Wytrenowano modele dla pojedynczych parametrów z wykorzystaniem modelu ResNet50 i AdaptiveLoss zgodnie z repo: https://github.com/jonbarron/robust_loss_pytorch Otrzymany wynik: *ResNet50_19_05_2022*: **0.94470**.
 
# Eksperymenty do wykonania

 * Wytrenować model 4-parametrowy (4Param) dla parametrów wyskalowanych
 * Wypróbowanie OneCycleLR https://pytorch.org/docs/stable/generated/torch.optim.lr_scheduler.OneCycleLR.html
 * Wykorzystanie RMSLELoss do przy wysokich wartościach (nieskalowanych) parametrów gleby: P, K, Mg
 * Wypróbowanie modeli z biblioteki timm (https://github.com/rwightman/pytorch-image-models) oraz modelu DeiT (https://github.com/facebookresearch/deit/blob/main/models.py). Biblioteka timm daje duże możliwości customizacji - warto przeczytać https://towardsdatascience.com/getting-started-with-pytorch-image-models-timm-a-practitioners-guide-4e77b4bf9055#9388 oraz  https://fastai.github.io/timmdocs/models#So-how-is-timm-able-to-load-these-weights? 
 * Wykonanie crop np. 5 metrów środkowych. Domyślamy się, że zazwyczaj w tym obszarze były wykonane pomiary terenowe. Dodatkowo, może to zmniejszyć ilośc danych do trenowania (?szum informacyjny?). Mozna to uzyskać z wykorzystaniem scipy.ndimage.center_of_mass na masce numpy.
 * 
 
 
 