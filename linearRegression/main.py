import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

# Загружаем датасет ирисов: признаки и названия
iris = load_iris()
X = iris.data
y = iris.target
feature_names = iris.feature_names

# Перебор всех пар признаков для построения линейной регрессии
best_r2 = 0
best_i, best_j = 0, 0
best_reg = None

for j in range(4):  # Целевой признак
    for i in range(4):  # Признак-модель
        if i != j:
            reg = LinearRegression()
            reg.fit(X[:, i].reshape(-1, 1), X[:, j])  # Обучаем: i -> j
            r2 = reg.score(X[:, i].reshape(-1, 1), X[:, j])  # Оцениваем R^2
            if r2 > best_r2:
                best_r2 = r2
                best_i, best_j = i, j
                best_reg = reg

print(f"Наилучшая регрессия: признак {feature_names[best_i]} -> признак {feature_names[best_j]}")
print(f"Коэффициент детерминации R^2: {best_r2:.2f}")

# Визуализация
plt.figure(figsize=(8, 6))
plt.scatter(X[:, best_i], X[:, best_j], c=y, cmap='viridis')  # Точки: реальные данные
plt.plot(X[:, best_i], best_reg.predict(X[:, best_i].reshape(-1, 1)), color='r')  # Линия: предсказание модели
plt.xlabel(feature_names[best_i])
plt.ylabel(feature_names[best_j])
plt.title(f"Линейная регрессия: {feature_names[best_i]} -> {feature_names[best_j]}")
plt.show()

# Поиск лучшей регрессии по двум признакам
best_r2 = 0
best_i1, best_i2, best_j = 0, 0, 0
best_reg = None

for j in range(4):  # Целевой признак
    for i1 in range(4):
        for i2 in range(i1 + 1, 4):  # Уникальные пары
            if i1 != j and i2 != j:  # Признаки не должны совпадать с целевым
                reg = LinearRegression()
                reg.fit(X[:, [i1, i2]], X[:, j])  # Обучение на двух признаках
                r2 = reg.score(X[:, [i1, i2]], X[:, j])  # Оценка качества
                if r2 > best_r2:  # Сохраняем наилучший результат
                    best_r2 = r2
                    best_i1, best_i2, best_j = i1, i2, j
                    best_reg = reg

print(f"Наилучшая регрессия с несколькими признаками: {feature_names[best_i1]}, {feature_names[best_i2]} -> {feature_names[best_j]}")
print(f"Коэффициент детерминации R^2: {best_r2:.2f}")
