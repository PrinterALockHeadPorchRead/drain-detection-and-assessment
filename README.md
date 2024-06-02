# Решение задачи распознавания дренажей и задачи оценки их состояния
***
Первая задача решается применением ансамбля моделей глубокого обучения, вторая - применением сверточной сети U-Net

## Структура файлов
***
### Задача распознавания дренажей
***
preprocessing.ipynb - обработка крупномасштабного изображения и его маски, получение предобработанных входных данных.

model_training_and_testing.ipynb - тренировка отдельных моделей и их тестирование (вычисление метрик оценки качества, построение графиков обучения, визуализация отдельных результатов).

ensembles.ipynb - применение методов ансамблевого обучения, сравнение их с отдельными моделями.

large_scale_vis.ipynb - применение моделей на крупномасштабном изображении.

### Задача оценки состояния дренажей
***
multispectrum.ipynb - предобработка входных мультиспектральных изображений, обучение модели U-Net, оценка результатов.

multispectrum_vis.ipynb - визуализация результатов на крупномасштабном мультиспектральном изображении.

### Архитектуры моделей

models - архитектуры моделей

## Данные
***
Тренировочные изображения - <https://drive.google.com/file/d/1bHnTqA-TPlU91d0UfVcNZcy7n6r9-u9-/view?usp=sharing>

Все изображения 256x256 - <https://drive.google.com/file/d/1VzlSMkshfGutyYHL3mDRklnfr6qUVD2_/view?usp=sharing>

Тренировочные маски - <https://drive.google.com/file/d/12HOwix1isKQTrSecsQEU0UdZxGM1cmmY/view?usp=sharing>

Маски всех изображений - <https://drive.google.com/file/d/1-CiXBuu9Q02XBYrtiO10SosOAiVpynwY/view?usp=sharing> и <https://drive.google.com/file/d/1Hnfw4BuRh12yjyNiWVTc4dtBA6o8Msyn/view?usp=sharing>

## Предобученные модели
***
U-Net - <https://drive.google.com/file/d/1mlkoCDE3xpTIotpCSiYzlNnZrixOHPqV/view?usp=sharing>

SegNet - <https://drive.google.com/file/d/1g4equJmC6h_uQiWBv0jlHLlMNhrI-gdj/view?usp=sharing>

DeepLabV3+ - <https://drive.google.com/file/d/1P8mvKKIyDq1YsElvvBNtTM8wwGwPMwQy/view?usp=sharing>

DDRNet - <https://drive.google.com/file/d/1C-VxlnoXceXjhsfkkeyG79bU_2-16CVx/view?usp=sharing>

FCN8s - <https://drive.google.com/file/d/1R88fbv6IrDcpsYNIPGUiJXz21jmXNL8R/view?usp=sharing>

FCN_ResNet - <https://drive.google.com/file/d/1owmbGP_SH_a0c5heES2pMnhfoFUfbM3H/view?usp=sharing>
