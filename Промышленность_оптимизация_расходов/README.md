# Промышленность оптимизация расходов
# Введение: 
Чтобы оптимизировать производственные расходы, необходимо построить модель, которая предскажет температуру стали и позвлит металлургическому комбинату ООО «Так закаляем сталь» р уменьшить потребление электроэнергии на этапе ее обработки. 
# Цель проекта: 
В срок до 1 июля 2023 года изучить предоставленные заказчиком данные, на которых построить модель со значением МАЕ неменее 6.8 для предсказания температуры металла, поступающего в машину непрерывной разливки. По результатам выполненой работы подготовить отчет.
```
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_absolute_error, r2_score, make_scorer
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestRegressor
from lightgbm import LGBMRegressor
from xgboost import XGBRegressor
from sklearn.dummy import DummyRegressor
```
### Решение трудностей в ходе проекта
Необходимые данные содержались в 7 разных таблицах, решено при помощи их объединения concat. Данные содержали пропуски и аномалии - удалены в ходе предоработки. Таблицы data_arc и data_temp содержали несколько объектов с одинаковым номером партии - решено при помощи агрегирования через сводную таблицу: в data_arc примена агрегирующая ф-я sum, а в data_temp выделен признак первого замера температуры как обучающий и последнего как целевой.
### Описание Итоговой модели 
В ходе обучения различных моделей с использованием кросс-валидации была выбрана модель, обученная при помощи LGBMRegressor с максимальной глубиной древа решений = 6 и числом итераций = 70. Данная модель показала лучшие результаты (5.9 показатель MAE на тренировочной выборке) и была проверена на тестовой выборке (MAE = 6.1).
