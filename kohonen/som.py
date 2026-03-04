import numpy as np
import matplotlib.pyplot as plt
from minisom import MiniSom
from sklearn.preprocessing import MinMaxScaler


class SOM:
    """
    Self-Organizing Map (Kohonen Network)

    A clean wrapper around MiniSom with built-in training,
    prediction, and visualization utilities.

    Parameters
    ----------
    grid_x : int
        Width of the neuron grid
    grid_y : int
        Height of the neuron grid
    learning_rate : float
        Initial learning rate (default 0.5)
    sigma : float
        Initial neighborhood radius (default 1.0)
    random_seed : int
        Seed for reproducibility (default 42)

    Example
    -------
    >>> from kohonen.som import SOM
    >>> som = SOM(grid_x=10, grid_y=10)
    >>> som.fit(X)
    >>> labels = som.predict(X)
    >>> som.plot_umatrix()
    """

    def __init__(self, grid_x=10, grid_y=10, learning_rate=0.5,
                 sigma=1.0, random_seed=42):
        self.grid_x = grid_x
        self.grid_y = grid_y
        self.learning_rate = learning_rate
        self.sigma = sigma
        self.random_seed = random_seed
        self.som = None
        self.scaler = MinMaxScaler()
        self._X_scaled = None

    def fit(self, X, epochs=1000, verbose=True):
        """
        Train the SOM on input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        epochs : int
            Number of training iterations (default 1000)
        verbose : bool
            Print training progress (default True)
        """
        self._X_scaled = self.scaler.fit_transform(X)
        input_dim = self._X_scaled.shape[1]

        self.som = MiniSom(
            x=self.grid_x,
            y=self.grid_y,
            input_len=input_dim,
            sigma=self.sigma,
            learning_rate=self.learning_rate,
            random_seed=self.random_seed
        )

        self.som.random_weights_init(self._X_scaled)

        if verbose:
            print(f"Training SOM ({self.grid_x}x{self.grid_y}) "
                  f"for {epochs} epochs...")

        self.som.train_random(self._X_scaled, epochs)

        if verbose:
            qe = self.quantization_error()
            te = self.topographic_error()
            print(f"Done. Quantization Error: {qe:.4f} | "
                  f"Topographic Error: {te:.4f}")

        return self

    def predict(self, X):
        """
        Assign each sample to its Best Matching Unit (BMU).

        Returns
        -------
        labels : array, shape (n_samples,)
            Flattened grid index of BMU for each sample
        """
        X_scaled = self.scaler.transform(X)
        labels = []
        for sample in X_scaled:
            bmu = self.som.winner(sample)
            labels.append(bmu[0] * self.grid_y + bmu[1])
        return np.array(labels)

    def quantization_error(self):
        """
        Average distance between each data point and its BMU.
        Lower is better — indicates how well neurons represent data.
        """
        return self.som.quantization_error(self._X_scaled)

    def topographic_error(self):
        """
        Fraction of samples where the two best matching neurons
        are not adjacent. Lower is better — indicates topology preservation.
        """
        return self.som.topographic_error(self._X_scaled)

    def plot_umatrix(self, figsize=(10, 8), title="U-Matrix"):
        """
        Plot the Unified Distance Matrix.

        Dark areas = cluster boundaries (neurons far apart)
        Light areas = cluster centers (neurons close together)
        """
        if self.som is None:
            raise RuntimeError("Train the SOM first using .fit()")

        fig, ax = plt.subplots(figsize=figsize)
        umatrix = self.som.distance_map()
        im = ax.pcolor(umatrix.T, cmap="bone_r")
        plt.colorbar(im, ax=ax, label="Neuron distance")
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        plt.tight_layout()
        plt.show()

    def plot_heatmap(self, X, labels=None, figsize=(10, 8),
                     title="Hit Map"):
        """
        Plot how many data points land on each neuron.
        Darker = more data points mapped there.

        Parameters
        ----------
        X : array-like, input data
        labels : array-like, optional class labels for coloring
        """
        if self.som is None:
            raise RuntimeError("Train the SOM first using .fit()")

        X_scaled = self.scaler.transform(X)
        fig, ax = plt.subplots(figsize=figsize)

        # Count hits per neuron
        hit_map = np.zeros((self.grid_x, self.grid_y))
        for sample in X_scaled:
            bmu = self.som.winner(sample)
            hit_map[bmu] += 1

        im = ax.pcolor(hit_map.T, cmap="Blues")
        plt.colorbar(im, ax=ax, label="Number of samples")

        # Overlay class labels if provided
        if labels is not None:
            unique_labels = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique_labels)))
            label_color = dict(zip(unique_labels, colors))

            for sample, label in zip(X_scaled, labels):
                bmu = self.som.winner(sample)
                ax.plot(bmu[0] + 0.5, bmu[1] + 0.5, 'o',
                        color=label_color[label],
                        markersize=4, alpha=0.6)

            # Legend
            handles = [plt.Line2D([0], [0], marker='o', color='w',
                       markerfacecolor=label_color[l], markersize=8,
                       label=str(l)) for l in unique_labels]
            ax.legend(handles=handles, loc='upper right',
                      title="Classes")

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.set_xlabel("Grid X")
        ax.set_ylabel("Grid Y")
        plt.tight_layout()
        plt.show()

    def plot_component_planes(self, feature_names=None, figsize=None):
        """
        Plot how each input feature varies across the map.
        Reveals which features drive cluster separation.

        Parameters
        ----------
        feature_names : list of str, optional
        """
        if self.som is None:
            raise RuntimeError("Train the SOM first using .fit()")

        weights = self.som.get_weights()
        n_features = weights.shape[2]

        if feature_names is None:
            feature_names = [f"Feature {i}" for i in range(n_features)]

        cols = min(4, n_features)
        rows = (n_features + cols - 1) // cols

        if figsize is None:
            figsize = (cols * 4, rows * 3.5)

        fig, axes = plt.subplots(rows, cols, figsize=figsize)
        axes = np.array(axes).flatten()

        for i in range(n_features):
            plane = weights[:, :, i]
            im = axes[i].pcolor(plane.T, cmap="coolwarm")
            plt.colorbar(im, ax=axes[i])
            axes[i].set_title(feature_names[i], fontsize=11)
            axes[i].set_xlabel("Grid X")
            axes[i].set_ylabel("Grid Y")

        # Hide unused subplots
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)

        fig.suptitle("Component Planes", fontsize=14,
                     fontweight="bold", y=1.02)
        plt.tight_layout()
        plt.show()

    def suggest_grid_size(self, n_samples):
        """
        Suggest a grid size based on dataset size.
        Rule of thumb: 5 * sqrt(n_samples) total neurons.

        Parameters
        ----------
        n_samples : int

        Returns
        -------
        (grid_x, grid_y) : tuple
        """
        total_neurons = int(5 * np.sqrt(n_samples))
        side = int(np.ceil(np.sqrt(total_neurons)))
        print(f"Suggested grid size for {n_samples} samples: "
              f"{side}x{side} ({side*side} neurons)")
        return side, side
