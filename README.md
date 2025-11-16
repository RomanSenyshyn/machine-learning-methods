# machine-learning-methods

The goal is to become familiar with the main methods of supervised learning, learn how to build and evaluate classification and regression models, and practically apply key algorithms such as linear regression, logistic regression, decision trees, and the k-nearest neighbors (k-NN) method.

## Installation / Встановлення

Install required dependencies:
```bash
pip install -r requirements.txt
```

Або встановіть окремо:
```bash
pip install numpy pandas matplotlib seaborn scikit-learn
```

## Usage / Використання

Run the Iris classification program:
```bash
python -u iris_classification.py
```

**Note:** The `-u` flag ensures unbuffered output. You can also run without it: `python iris_classification.py`

## Program Description / Опис програми

The program demonstrates three classification algorithms on the Iris dataset:

1. **Logistic Regression** - Linear classification model
2. **Decision Trees** - Tree-based classification with visualization
3. **K-Nearest Neighbors (K-NN)** - Instance-based learning with k parameter analysis

The program includes:
- Data loading and exploration
- Feature distribution visualization (pairplot)
- Train/test split
- Model training and evaluation
- Decision tree visualization
- K-NN accuracy analysis for different k values

Програма демонструє три алгоритми класифікації на наборі даних Iris:

1. **Логістична регресія** - Лінійна модель класифікації
2. **Дерева рішень** - Класифікація на основі дерев з візуалізацією
3. **K-найближчих сусідів (K-NN)** - Навчання на основі прикладів з аналізом параметра k

Програма включає:
- Завантаження та огляд даних
- Візуалізацію розподілу ознак (pairplot)
- Розбиття на тренувальну та тестову вибірки
- Навчання та оцінку моделей
- Візуалізацію дерева рішень
- Аналіз точності K-NN для різних значень k