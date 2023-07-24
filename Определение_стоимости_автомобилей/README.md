# Определение стоимости автомобилей
 В вашем распоряжении исторические данные: технические характеристики, комплектации и цены автомобилей. Вам нужно построить модель для определения стоимости. 

Заказчику важны:

- качество предсказания;
- скорость предсказания;
- время обучения.

```
import numpy as np
import pandas as pd
import seaborn as sns

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler

from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.dummy import DummyRegressor
from lightgbm import LGBMRegressor
from sklearn.metrics import mean_squared_error, make_scorer
```

# Вывод: 
Мы построили модель с минимальным значением RMSE, для этого избавились от пустых значений в исходных данных, удалили неинформативные признаки, перевели категориальные признаки в численные, стандартизировали численые признаки и избавились от дисбаланса. Модель проверена на тестовой выборке.
