#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Програма для класифікації Iris dataset з використанням різних алгоритмів машинного навчання
"""

# Крок 1. Підготовка середовища
# Встановіть необхідні бібліотеки:
# pip install numpy pandas matplotlib seaborn scikit-learn

# Крок 2. Завантаження та огляд даних
import sys
import traceback
import io

# Налаштування кодування для Windows консолі
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

try:
    import matplotlib
    # Встановлюємо backend для Windows
    matplotlib.use('Agg')  # Використовуємо non-interactive backend
    import matplotlib.pyplot as plt
    from sklearn.datasets import load_iris
    import pandas as pd
    import seaborn as sns
    import numpy as np
    
    # Налаштування для seaborn
    sns.set_style("whitegrid")
    
    print("=" * 60, flush=True)
    print("Крок 2. Завантаження та огляд даних", flush=True)
    print("=" * 60, flush=True)

    data = load_iris()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target

    print("\nПерші 5 рядків даних:", flush=True)
    print(df.head(), flush=True)

    print("\nІнформація про дані:", flush=True)
    print(f"Розмір датафрейму: {df.shape}", flush=True)
    print(f"Типи даних:\n{df.dtypes}", flush=True)
    print(f"Пропущені значення:\n{df.isnull().sum()}", flush=True)
    sys.stdout.flush()

    print("\nСтатистика даних:", flush=True)
    print(df.describe(), flush=True)

# Завдання: Візуалізуйте розподіл ознак
    print("\nВізуалізація розподілу ознак...", flush=True)
    sys.stdout.flush()
    sns.pairplot(df, hue='target', diag_kind='kde')
    plt.suptitle('Розподіл ознак Iris dataset', y=1.02)
    plt.tight_layout()
    plt.savefig('pairplot.png', dpi=150, bbox_inches='tight')
    print("Графік збережено як 'pairplot.png'", flush=True)
    plt.close()  # Закриваємо figure для звільнення пам'яті
    sys.stdout.flush()

# Крок 3. Розбиття на тренувальну і тестову вибірку
    from sklearn.model_selection import train_test_split

    print("\n" + "=" * 60, flush=True)
    print("Крок 3. Розбиття на тренувальну і тестову вибірку", flush=True)
    print("=" * 60, flush=True)

    X = df[data.feature_names]
    y = df['target']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    print(f"Розмір тренувальної вибірки: {X_train.shape}", flush=True)
    print(f"Розмір тестової вибірки: {X_test.shape}", flush=True)
    sys.stdout.flush()

# Крок 4. Лінійна модель: логістична регресія
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

    print("\n" + "=" * 60, flush=True)
    print("Крок 4. Лінійна модель: логістична регресія", flush=True)
    print("=" * 60, flush=True)

    print("Навчання моделі...", flush=True)
    model_lr = LogisticRegression(max_iter=200, random_state=42)
    model_lr.fit(X_train, y_train)
    y_pred_lr = model_lr.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred_lr):.4f}", flush=True)
    print("\nМатриця плутанини:", flush=True)
    print(confusion_matrix(y_test, y_pred_lr), flush=True)
    print("\nЗвіт класифікації:", flush=True)
    print(classification_report(y_test, y_pred_lr, target_names=data.target_names), flush=True)
    sys.stdout.flush()

# Крок 5. Дерева рішень
    from sklearn.tree import DecisionTreeClassifier, plot_tree

    print("\n" + "=" * 60, flush=True)
    print("Крок 5. Дерева рішень", flush=True)
    print("=" * 60, flush=True)

    print("Навчання моделі...", flush=True)
    model_tree = DecisionTreeClassifier(max_depth=3, random_state=42)
    model_tree.fit(X_train, y_train)
    y_pred_tree = model_tree.predict(X_test)

    print(f"\nAccuracy: {accuracy_score(y_test, y_pred_tree):.4f}", flush=True)
    print("\nМатриця плутанини:", flush=True)
    print(confusion_matrix(y_test, y_pred_tree), flush=True)
    print("\nЗвіт класифікації:", flush=True)
    print(classification_report(y_test, y_pred_tree, target_names=data.target_names), flush=True)

    print("\nВізуалізація дерева рішень...", flush=True)
    sys.stdout.flush()
    plt.figure(figsize=(15, 10))
    plot_tree(model_tree, feature_names=data.feature_names, class_names=data.target_names, filled=True)
    plt.title('Дерево рішень для класифікації Iris', fontsize=16)
    plt.tight_layout()
    plt.savefig('decision_tree.png', dpi=150, bbox_inches='tight')
    print("Графік збережено як 'decision_tree.png'", flush=True)
    plt.close()  # Закриваємо figure для звільнення пам'яті
    sys.stdout.flush()

# Крок 6. Метод k-ближчих сусідів
    from sklearn.neighbors import KNeighborsClassifier

    print("\n" + "=" * 60, flush=True)
    print("Крок 6. Метод k-ближчих сусідів", flush=True)
    print("=" * 60, flush=True)

    print("Навчання моделі...", flush=True)
    model_knn = KNeighborsClassifier(n_neighbors=5)
    model_knn.fit(X_train, y_train)
    y_pred_knn = model_knn.predict(X_test)

    print(f"\nAccuracy (k=5): {accuracy_score(y_test, y_pred_knn):.4f}", flush=True)
    print("\nМатриця плутанини:", flush=True)
    print(confusion_matrix(y_test, y_pred_knn), flush=True)
    print("\nЗвіт класифікації:", flush=True)
    print(classification_report(y_test, y_pred_knn, target_names=data.target_names), flush=True)

# Візуалізація ефекту k
    print("\nВізуалізація ефекту k...", flush=True)
    sys.stdout.flush()
    acc = []
    for k in range(1, 15):
        model = KNeighborsClassifier(n_neighbors=k)
        model.fit(X_train, y_train)
        acc.append(accuracy_score(y_test, model.predict(X_test)))
        if k % 5 == 0:
            print(f"  Обчислено для k={k}...", flush=True)

    plt.figure(figsize=(10, 6))
    plt.plot(range(1, 15), acc, marker='o', linewidth=2, markersize=8)
    plt.xlabel('k (кількість сусідів)', fontsize=12)
    plt.ylabel('Accuracy', fontsize=12)
    plt.title('Вплив кількості сусідів k на точність моделі', fontsize=14)
    plt.grid(True, alpha=0.3)
    plt.xticks(range(1, 15))
    plt.tight_layout()
    plt.savefig('knn_accuracy.png', dpi=150, bbox_inches='tight')
    print("Графік збережено як 'knn_accuracy.png'", flush=True)
    plt.close()  # Закриваємо figure для звільнення пам'яті
    sys.stdout.flush()

# Порівняння моделей
    print("\n" + "=" * 60, flush=True)
    print("Порівняння моделей", flush=True)
    print("=" * 60, flush=True)
    print(f"Логістична регресія: {accuracy_score(y_test, y_pred_lr):.4f}", flush=True)
    print(f"Дерево рішень:       {accuracy_score(y_test, y_pred_tree):.4f}", flush=True)
    print(f"K-NN (k=5):          {accuracy_score(y_test, y_pred_knn):.4f}", flush=True)

    print("\nПрограма завершена!", flush=True)
    print("\nВсі графіки збережено у файли:", flush=True)
    print("  - pairplot.png", flush=True)
    print("  - decision_tree.png", flush=True)
    print("  - knn_accuracy.png", flush=True)
    sys.stdout.flush()  # Гарантуємо виведення всіх повідомлень

except Exception as e:
    print(f"\nПОМИЛКА: {e}", file=sys.stderr, flush=True)
    traceback.print_exc(file=sys.stderr)
    sys.stderr.flush()
    sys.exit(1)

