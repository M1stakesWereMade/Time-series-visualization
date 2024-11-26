# Визуализация временных рядов с использованием Matplotlib, Seaborn и Plotly

Этот проект создает сложные временные ряды, комбинируя различные компоненты: тренд, сезонность, автокорреляцию и шум. Для визуализации использованы библиотеки Matplotlib, Seaborn и Plotly, что позволяет получить статичные и интерактивные графики. Проект демонстрирует, как эти компоненты могут взаимодействовать друг с другом и создавать реалистичные модели данных.

## Описание

В этом коде:
- **Matplotlib** используется для создания стандартных графиков.
- **Seaborn** применяется для улучшения визуализации графиков с более эстетичным стилем.
- **Plotly** используется для создания интерактивных графиков, которые можно масштабировать и исследовать в реальном времени.

## Требования

Перед запуском кода убедитесь, что у вас установлены следующие библиотеки:

- **`numpy`** — для работы с массивами данных.
- **`matplotlib`** — для создания статичных графиков.
- **`seaborn`** — для улучшения визуализации с помощью стильных графиков.
- **`plotly`** — для создания интерактивных графиков.

Чтобы установить эти библиотеки, выполните следующую команду:

```bash
pip install numpy matplotlib seaborn plotly
```

## Структура кода

Импорт библиотек

```python
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.graph_objects as go
```

- **`numpy`** - используется для работы с числовыми данными.
- **`matplotlib.pyplot`** - отвечает за создание стандартных графиков.
- **`seaborn`** - используется для улучшения визуализации графиков.
- **`plotly.graph_objects`** - предоставляет возможность строить интерактивные графики.

## Создание тренда

Для создания линейного тренда используется функция **`trend`**:

```python
def trend(time, slope=0):
    return slope * time
```
Тренд добавляется к основной временной серии с определенным наклоном (положительным или отрицательным).

## Сезонность

Для создания сезонности используются функции **`seasonal_pattern`** и **`seasonality`**:

```python
def seasonal_pattern(season_time):
    return np.where(season_time<0.45,
                   np.cos(season_time * 2 * np.pi),
                   1 / np.exp(3 * season_time))

def seasonality(time , period , amplitude = 1 , phase = 0 ):
    season_time = ((time + phase) %  period ) / period
    return amplitude * seasonal_pattern(season_time)

```
Эти функции генерируют сезонные колебания, которые можно использовать для моделирования циклических данных.

## Белый шум

Функция **`white_noise`** генерирует случайный белый шум:

```python
def white_noise(time, noise_level = 1 , seed = None):
    random = np.random.RandomState(seed)
    return random.random(len(time)) * noise_level
```

## Автокорреляция

Для создания автокорреляции с различными параметрами используются функции:

```python
def autocorrelation_1(time , amplitude , seed = None):
    rnd = np.random.RandomState(seed)
    a1 = 0.5
    a2 = -0.1
    rnd_ar = rnd.randn(len(time) + 50)
    rnd_ar[:50] = 100
    for step in range(50, len(time) + 50 ):
        rnd_ar[step] += a1 * rnd_ar[step - 50]
        rnd_ar[step] += a2 * rnd_ar[step - 33]
    return rnd_ar[50:] * amplitude

def autocorrelation_2(time, amplitude, seed=None):
    rnd = np.random.RandomState(seed)
    a1 = 0.8
    ar = rnd.randn(len(time) + 1)
    for step in range(1, len(time) + 1):
        ar[step] += a1 * ar[step - 1]
    return ar[1:] * amplitude
```
Эти функции создают серию данных с автокорреляцией.

## Визуализация

Для каждого типа данных (тренд, сезонность, шум и т.д.) предусмотрены три варианта визуализации:

1. **Matplotlib** (стандартная визуализация)
2. **Seaborn** (стилизованные графики)
3. **Plotly** (интерактивные графики)

Для каждой из этих библиотек определены функции, например, для визуализации с помощью **`seaborn`**:

```python
def plot_seaborn_series(time, series, start=0, end=None, color="blue"):
    sns.lineplot(x=time[start:end], y=series[start:end], color=color)
    plt.xlabel("Время")
    plt.ylabel("Значение")
    plt.tight_layout()
```

Аналогичная функция для **`plotly`**:

```python
def plot_plotly_series(time, series, start=0, end=None, title="Plot", color="blue"):
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=time[start:end], y=series[start:end], mode='lines', line=dict(color=color)))
    fig.update_layout(title=title, xaxis_title="Время", yaxis_title="Значение")
    fig.show()
```

## Пример использования

1. **Тренд (положительный и отрицательный наклон):**
    - Графики тренда можно построить с помощью **`matplotlib`**, **`seaborn`** и **`plotly`**.

2. **Сезонность:**
    - Графики сезонных колебаний можно визуализировать с помощью всех трех библиотек.

3. **Белый шум:**
    - Сгенерировать графики белого шума можно и с использованием всех библиотек.

4. **Сложные графики:** 
    - Сложные временные ряды, комбинирующие тренд, сезонность, автокорреляцию и шум, также можно визуализировать с использованием различных библиотек.

## Запуск кода

После того как все зависимости установлены, запустите скрипт:

```bash
python time_series_plot.py
```
Код создаст несколько графиков для различных типов данных и сохранит их в виде изображений.

## Примечания

- Для корректного отображения интерактивных графиков **`plotly`**, требуется установленный Jupyter Notebook или интеграция с браузером.
- Для лучшей визуализации в Jupyter Notebook можно использовать встроенную поддержку **`plotly`** с помощью команды **`fig.show()`**.