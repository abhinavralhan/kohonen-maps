# Kohonen Maps

Implementation of Self-Organizing Maps (SOM), Growing SOM (GSOM), and Neural Gas for unsupervised clustering and dimensionality reduction.

![Python](https://img.shields.io/badge/python-3.9%2B-blue)
![License](https://img.shields.io/badge/license-MIT-green)
![Stars](https://img.shields.io/github/stars/abhinavralhan/kohonen-maps)
![Forks](https://img.shields.io/github/forks/abhinavralhan/kohonen-maps)

---

## What's in this repo

| Algorithm | Description | Notebook |
|---|---|---|
| SOM | Standard Self-Organizing Map on fixed grid | `notebooks/01-som-basics.ipynb` |
| SOM | Customer segmentation real-world use case | `notebooks/02-som-customer-segmentation.ipynb` |
| Neural Gas | Free neuron placement, no grid constraint | `notebooks/03-neural-gas.ipynb` |
| GSOM | Growing SOM — dynamic grid expansion | `notebooks/gsom-iris-python.ipynb` |

---

## Quickstart

### Install
```bash
git clone https://github.com/abhinavralhan/kohonen-maps.git
cd kohonen-maps
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### Run a notebook
```bash
jupyter notebook
```
Navigate to `notebooks/` and open any notebook.

### Use the package directly
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

## How SOMs Work

SOMs, invented by Teuvo Kohonen in the 1980s, are neural networks that produce a low-dimensional (typically 2D) representation of high-dimensional data while preserving topological structure. Similar data points stay close on the map.

### Algorithm
1. Initialize a grid of neurons with random weight vectors
2. For each input, find the Best Matching Unit (BMU) — neuron with weights closest to the input
3. Update the BMU and its neighbors to be more similar to the input
4. Decrease learning rate and neighborhood size over time
5. Repeat until convergence

### Key Parameters

| Parameter | Description |
|---|---|
| Grid Size | Dimensions of the neuron grid (e.g. 10×10) |
| Learning Rate | How much weights adjust per step — decreases over time |
| Neighborhood (sigma) | Radius of neurons affected by each update |
| Epochs | Number of training iterations |

### Visualizations

| Plot | What it shows |
|---|---|
| U-Matrix | Distances between neurons — dark = cluster boundary |
| Hit Map | How many data points land on each neuron |
| Component Planes | How each input feature varies across the map |

---

## How Neural Gas Works

Neural Gas (Martinetz & Schulten, 1991) places neurons freely in data space — no grid constraints. Instead of grid distance, it uses **rank** of distance to determine neighborhood influence.

### Key difference from SOM

| | SOM | Neural Gas |
|---|---|---|
| Neuron placement | Fixed grid | Freely in data space |
| Neighborhood | Grid distance | Rank of distance |
| Best for | Visualization, topology mapping | Irregular clusters, density estimation |
| Interpretability | High (U-Matrix) | Medium |

---

## Choosing the Right Algorithm

| Situation | Use |
|---|---|
| Need a 2D visual map of data | SOM |
| Clusters have irregular shapes | Neural Gas |
| Dataset grows over time | GSOM |
| Want topology + flexibility | GSOM |
| Benchmarking against PCA/t-SNE | SOM |

---

## Comparison with Other Methods

| Method | Preserves Topology | Interpretable | Scalable |
|---|---|---|---|
| SOM | Yes | High | Medium |
| Neural Gas | Partial | Medium | Medium |
| GSOM | Yes | High | Medium |
| PCA | No | Medium | High |
| t-SNE | Partial | Low | Low |
| UMAP | Yes | Medium | High |
| K-Means | No | High | High |

---

## Repo Structure
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

## Applications

- **Customer Segmentation** — group customers by purchasing behavior
- **Document Clustering** — organize text by topic
- **Anomaly Detection** — identify outliers as points far from any neuron
- **Image Compression** — reduce color palettes while preserving visual quality
- **Bioinformatics** — gene expression analysis

---

## References

- [Self-Organizing Maps — Conceptual Overview](https://medium.com/@abhinavr8/self-organizing-maps-ff5853a118d4)
- [GSOM Paper](https://pdfs.semanticscholar.org/ea7b/88d583abdb8cd5976a636540ca7ec27261e3.pdf)
- [Neural Gas — Martinetz & Schulten (1991)](https://papers.cnl.salk.edu/PDFs/A%20%22Neural-Gas%22%20Network%20Learns%20Topologies%201991-3522.pdf)
- [MiniSom Library](https://github.com/JustGlowing/minisom)

---

## Contributing

Feel free to get in touch!
