

# Mapas de Kohonen

Implementación de Mapas Autoorganizados (SOM), SOM Creciente (GSOM) y Neural Gas para clustering no supervisado y reducción de dimensionalidad.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Stars](https://img.shields.io/github/stars/abhinavralhan/kohonen-maps)
![Forks](https://img.shields.io/github/forks/abhinavralhan/kohonen-maps)

---

## ¿Qué hay en este repositorio?

| Algoritmo | Descripción | Notebook |
|---|---|---|
| SOM | Mapa Autoorganizado estándar en cuadrícula fija | `notebooks/01-som-basics.ipynb` |
| SOM | Caso de uso real de segmentación de clientes | `notebooks/02-som-customer-segmentation.ipynb` |
| Neural Gas | Colocación libre de neuronas, sin restricción de cuadrícula | `notebooks/03-neural-gas.ipynb` |
| GSOM | SOM Creciente — expansión dinámica de la cuadrícula | `notebooks/gsom-iris-python.ipynb` |

---

## Inicio rápido

### Instalación
```bash
git clone https://github.com/abhinavralhan/kohonen-maps.git
cd kohonen-maps
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Ejecutar un notebook
```bash
jupyter notebook
```
Navega a `notebooks/` y abre cualquier notebook.

### Usar el paquete directamente
```python
from kohonen.som import SOM
from kohonen.neural_gas import NeuralGas
from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data

# Train SOM
som = SOM(grid_x=10, grid_y=10, learning_rate=0.5)
som.fit(X, epochs=1000)
som.plot_umatrix()
som.plot_heatmap(X, labels=iris.target_names[iris.target])

# Train Neural Gas
ng = NeuralGas(n_neurons=30)
ng.fit(X, epochs=2000)
ng.plot_neurons(X, labels=iris.target)
ng.plot_convergence()
```

---

## ¿Cómo funcionan los SOM?

Los SOM, inventados por Teuvo Kohonen en la década de 1980, son redes neuronales que producen una representación de baja dimensión (típicamente 2D) de datos de alta dimensión mientras preservan la estructura topológica. Los puntos de datos similares permanecen cerca en el mapa.

### Algoritmo
1. Inicializar una cuadrícula de neuronas con vectores de pesos aleatorios
2. Para cada entrada, encontrar la Unidad de Mejor Coincidencia (BMU): la neurona con pesos más cercanos a la entrada
3. Actualizar la BMU y sus vecinas para que sean más similares a la entrada
4. Disminuir la tasa de aprendizaje y el tamaño de la vecindad con el tiempo
5. Repetir hasta la convergencia

### Parámetros clave

| Parámetro | Descripción |
|---|---|
| Tamaño de cuadrícula | Dimensiones de la cuadrícula de neuronas (ej. 10×10) |
| Tasa de aprendizaje | Cuánto se ajustan los pesos por paso — disminuye con el tiempo |
| Vecindad (sigma) | Radio de neuronas afectadas por cada actualización |
| Épocas | Número de iteraciones de entrenamiento |

### Visualizaciones

| Gráfico | Qué muestra |
|---|---|
| U-Matrix | Distancias entre neuronas — oscuro = límite de cluster |
| Hit Map | Cuántos puntos de datos caen en cada neurona |
| Planos de componentes | Cómo varía cada característica de entrada a través del mapa |

---

## ¿Cómo funciona Neural Gas?

Neural Gas (Martinetz & Schulten, 1991) coloca las neuronas libremente en el espacio de datos, sin restricciones de cuadrícula. En lugar de la distancia en cuadrícula, utiliza el **rango** de distancia para determinar la influencia de la vecindad.

### Diferencia clave con SOM

| | SOM | Neural Gas |
|---|---|---|
| Colocación de neuronas | Cuadrícula fija | Libre en el espacio de datos |
| Vecindad | Distancia en cuadrícula | Rango de distancia |
| Mejor para | Visualización, mapeo topológico | Clusters irregulares, estimación de densidad |
| Interpretabilidad | Alta (U-Matrix) | Media |

---

## Elegir el algoritmo correcto

| Situación | Usar |
|---|---|
| Se necesita un mapa visual 2D de los datos | SOM |
| Los clusters tienen formas irregulares | Neural Gas |
| El conjunto de datos crece con el tiempo | GSOM |
| Se desea topología + flexibilidad | GSOM |
| Benchmarking contra PCA/t-SNE | SOM |

---

## Comparación con otros métodos

| Método | Preserva topología | Interpretable | Escalable |
|---|---|---|---|
| SOM | Sí | Alta | Media |
| Neural Gas | Parcial | Media | Media |
| GSOM | Sí | Alta | Media |
| PCA | No | Media | Alta |
| t-SNE | Parcial | Baja | Baja |
| UMAP | Sí | Media | Alta |
| K-Means | No | Alta | Alta |

---

## Estructura del repositorio
```
kohonen-maps/
├── kohonen/                  # Reusable Python package
│   ├── __init__.py
│   ├── som.py                # SOM class
│   ├── neural_gas.py         # Neural Gas class
│   ├── gsom.py               # GSOM (coming soon)
│   ├── metrics.py            # Evaluation utilities (coming soon)
│   └── visualize.py          # Visualization helpers (coming soon)
├── notebooks/
│   ├── 01-som-basics.ipynb
│   ├── 02-som-customer-segmentation.ipynb
│   ├── 03-neural-gas.ipynb
│   └── gsom-iris-python.ipynb
├── requirements.txt
├── CONTRIBUTING.md
└── README.md
```

---

## Aplicaciones

- **Segmentación de clientes** — agrupar clientes por comportamiento de compra
- **Clustering de documentos** — organizar texto por tema
- **Detección de anomalías** — identificar valores atípicos como puntos lejanos de cualquier neurona
- **Compresión de imágenes** — reducir paletas de colores preservando la calidad visual
- **Bioinformática** — análisis de expresión génica

---

## Referencias

- [Mapas Autoorganizados — Vista conceptual](https://medium.com/@abhinavr8/self-organizing-maps-ff5853a118d4)
- [Artículo de GSOM](https://pdfs.semanticscholar.org/ea7b/88d583abdb8cd5976a636540ca7ec27261e3.pdf)
- [Neural Gas — Martinetz & Schulten (1991)](https://papers.cnl.salk.edu/PDFs/A%20%22Neural-Gas%22%20Network%20Learns%20Topologies%201991-3522.pdf)
- [Biblioteca MiniSom](https://github.com/JustGlowing/minisom)

---

## Contribuciones

¡No dudes en ponerte en contacto!
