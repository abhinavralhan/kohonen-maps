# kohonen-maps

# Self-Organizing Maps (Kohonen Networks)

Implementation of Kohonen networks and Growing Self-Organizing Maps (GSOM) for unsupervised clustering and dimensionality reduction with topology preservation.

## What are Self-Organizing Maps?

SOMs, invented by Teuvo Kohonen in the 1980s, are neural networks that produce a low-dimensional (typically 2D) representation of high-dimensional data while preserving topological properties. Similar data points remain close on the map.

## How SOMs Work

### The Algorithm

1. **Initialize** a grid of neurons with random weight vectors
2. **For each input**, find the Best Matching Unit (BMU) - the neuron with weights closest to the input
3. **Update** the BMU and its neighbors to be more similar to the input
4. **Decrease** the learning rate and neighborhood size over time
5. **Repeat** until convergence

### Key Parameters

| Parameter | Description |
|-----------|-------------|
| Grid Size | Dimensions of the neuron grid (e.g., 10x10) |
| Learning Rate | How much weights adjust (decreases over time) |
| Neighborhood | Radius of neurons affected by each update |
| Iterations | Number of training passes |

## Implementations

### Standard SOM (Python)

```python
from minisom import MiniSom

# Initialize
som = MiniSom(10, 10, input_dim, sigma=1.0, learning_rate=0.5)

# Train
som.train_random(data, 1000)

# Get cluster assignments
winner = som.winner(sample)
```

### Growing SOM (GSOM)

Unlike standard SOMs with fixed grid size, GSOMs dynamically grow the network based on data complexity. New neurons are added when quantization error exceeds a threshold.

## Applications

- **Customer Segmentation**: Group customers by behavior patterns
- **Document Clustering**: Organize text documents by topic
- **Image Compression**: Reduce color palettes while preserving visual quality
- **Anomaly Detection**: Identify outliers as points far from any neuron

## Visualization

SOMs produce intuitive visualizations:
- **U-Matrix**: Shows distances between neurons (cluster boundaries)
- **Component Planes**: Visualize how each input feature varies across the map
- **Hit Maps**: Show where data points cluster

## Tech Stack

- Python (MiniSom, Somoclu)
- R (kohonen package)
- NumPy / Pandas
- Matplotlib / Seaborn

## Comparison with Other Methods

| Method | Preserves Topology | Interpretable | Scalable |
|--------|-------------------|---------------|----------|
| SOM | Yes | High | Medium |
| PCA | No | Medium | High |
| t-SNE | Partial | Low | Low |
| UMAP | Yes | Medium | High |
| K-Means | No | High | High |

## References

### Conceptual understanding of SOM and GSOM:

https://medium.com/@abhinavr8/self-organizing-maps-ff5853a118d4

https://pdfs.semanticscholar.org/ea7b/88d583abdb8cd5976a636540ca7ec27261e3.pdf

### Implementation of SOM (Python):

https://github.com/abhinavralhan/kohonen-maps/blob/master/som-random.ipynb

https://github.com/abhinavralhan/kohonen-maps/blob/master/somoclu-iris.ipynb


### Implementation of GSOM (Python):

https://github.com/abhinavralhan/kohonen-maps/blob/master/gsom-iris-python.ipynb

### Implementation of GSOM (R):

http://rpubs.com/abhinavralhan/gsom
