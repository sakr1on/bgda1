import numpy as np
from sklearn.datasets import load_iris
from sklearn.cluster import KMeans
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from collections import Counter

# Загружаем датасет
iris = load_iris()
X = iris.data # Массив признаков
y = iris.target # Массив меток классов 

# Делим выборку: обучение, тест
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Задаём диапазон значений k (число кластеров) для перебора
k_range = range(1, 21)

accuracy_scores = [] # Список для хранения точности на каждом k
best_accuracy = 0 # Максимальная достигнутая точность
best_k = 0 # Значение k, при котором была достигнута лучшая точность

# возможные значения k
for k in k_range:
    kmeans = KMeans(n_clusters=k, random_state=42) # KMeans с заданным числом кластеров
    kmeans.fit(X_train) # Обучаем модель на тренировочных данных
    y_pred = kmeans.predict(X_test) # Предсказываем, к какому кластеру отнести каждый объект из теста
    labels = np.zeros_like(y_pred) # Создаём массив финальных меток для сравнения с y_test

    # Для каждого кластера определяем наиболее частую метку
    for i in range(k):
        mask = (y_pred == i) # Находим все объекты, попавшие в кластер i
        if np.any(mask):                             
            most_common = Counter(y_test[mask]).most_common(1)
            if most_common:
                labels[mask] = most_common[0][0]

    accuracy = accuracy_score(y_test, labels)
    accuracy_scores.append(accuracy)

    if accuracy > best_accuracy:
        best_accuracy = accuracy
        best_k = k

average_error = 1 - best_accuracy

print(f"Оптимальное значение k: {best_k}")
print(f"Средняя ошибка на тестовой выборке: {average_error:.2f}")


plt.figure(figsize=(12, 6))

# Слева — реальные сорта из тестовой выборки
plt.subplot(1, 2, 1)
plt.scatter(X_test[:, 0], X_test[:, 1], c=y_test, cmap='viridis')  # Цвет — реальная метка
plt.title("Реальные сорта ирисов")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

# Справа — результат кластеризации при лучшем k
plt.subplot(1, 2, 2)
kmeans = KMeans(n_clusters=best_k, random_state=42)
kmeans.fit(X_train)
y_pred = kmeans.predict(X_test)

labels = np.zeros_like(y_pred)
for i in range(best_k):
    mask = (y_pred == i)
    if np.any(mask):
        most_common = Counter(y_test[mask]).most_common(1)
        if most_common:
            labels[mask] = most_common[0][0]

plt.scatter(X_test[:, 0], X_test[:, 1], c=labels, cmap='viridis')  # Цвет — переназначенные метки
plt.title("Результаты кластеризации KMeans")
plt.xlabel(iris.feature_names[0])
plt.ylabel(iris.feature_names[1])

plt.tight_layout()
plt.show()
