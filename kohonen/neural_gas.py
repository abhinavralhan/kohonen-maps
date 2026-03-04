import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class NeuralGas:
    """
    Neural Gas Algorithm (Martinetz & Schulten, 1991)

    Unlike SOM which uses a fixed grid, Neural Gas places neurons
    freely in data space — no topology constraints. Neurons
    arrange themselves purely by data density.

    Key difference from SOM:
    - SOM: neurons on fixed grid, neighborhood = grid distance
    - Neural Gas: neurons float freely, neighborhood = rank of distance

    Parameters
    ----------
    n_neurons : int
        Number of neurons (default 100)
    learning_rate_i : float
        Initial learning rate (default 0.5)
    learning_rate_f : float
        Final learning rate (default 0.005)
    neighborhood_i : float
        Initial neighborhood range (default 10.0)
    neighborhood_f : float
        Final neighborhood range (default 0.1)
    random_seed : int
        Seed for reproducibility (default 42)

    Example
    -------
    >>> from kohonen.neural_gas import NeuralGas
    >>> ng = NeuralGas(n_neurons=50)
    >>> ng.fit(X, epochs=1000)
    >>> labels = ng.predict(X)
    >>> ng.plot_neurons(X)
    """

    def __init__(self, n_neurons=100,
                 learning_rate_i=0.5, learning_rate_f=0.005,
                 neighborhood_i=10.0, neighborhood_f=0.1,
                 random_seed=42):
        self.n_neurons = n_neurons
        self.learning_rate_i = learning_rate_i
        self.learning_rate_f = learning_rate_f
        self.neighborhood_i = neighborhood_i
        self.neighborhood_f = neighborhood_f
        self.random_seed = random_seed
        self.neurons = None
        self.scaler = MinMaxScaler()
        self._X_scaled = None
        self._errors = []

    def fit(self, X, epochs=1000, verbose=True):
        """
        Train Neural Gas on input data.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
        epochs : int
            Number of training iterations (default 1000)
        verbose : bool
            Print progress (default True)
        """
        np.random.seed(self.random_seed)
        self._X_scaled = self.scaler.fit_transform(X)
        n_samples, n_features = self._X_scaled.shape

        # Initialize neurons randomly within data range
        self.neurons = np.random.rand(self.n_neurons, n_features)

        self._errors = []

        for epoch in range(epochs):
            # Decay learning rate and neighborhood over time
            t = epoch / epochs
            lr = self.learning_rate_i * (
                self.learning_rate_f / self.learning_rate_i) ** t
            nb = self.neighborhood_i * (
                self.neighborhood_f / self.neighborhood_i) ** t

            # Pick random sample
            idx = np.random.randint(0, n_samples)
            sample = self._X_scaled[idx]

            # Rank neurons by distance to sample
            dists = np.linalg.norm(self.neurons - sample, axis=1)
            ranking = np.argsort(dists)

            # Update all neurons — closer rank = larger update
            for rank, neuron_idx in enumerate(ranking):
                influence = np.exp(-rank / nb)
                self.neurons[neuron_idx] += (
                    lr * influence * (sample - self.neurons[neuron_idx])
                )

            # Track quantization error every 100 epochs
            if epoch % 100 == 0:
                qe = self.quantization_error()
                self._errors.append((epoch, qe))

        if verbose:
            print(f"Training complete. "
                  f"Quantization Error: {self.quantization_error():.4f}")

        return self

    def predict(self, X):
        """
        Assign each sample to its nearest neuron.

        Returns
        -------
        labels : array, shape (n_samples,)
            Index of nearest neuron for each sample
        """
        if self.neurons is None:
            raise RuntimeError("Train the model first using .fit()")

        X_scaled = self.scaler.transform(X)
        labels = []
        for sample in X_scaled:
            dists = np.linalg.norm(self.neurons - sample, axis=1)
            labels.append(np.argmin(dists))
        return np.array(labels)

    def quantization_error(self):
        """
        Average distance between each data point and its nearest neuron.
        Lower is better.
        """
        if self.neurons is None or self._X_scaled is None:
            raise RuntimeError("Train the model first using .fit()")

        errors = []
        for sample in self._X_scaled:
            dists = np.linalg.norm(self.neurons - sample, axis=1)
            errors.append(np.min(dists))
        return np.mean(errors)

    def plot_neurons(self, X, labels=None, feature_idx=(0, 1),
                     figsize=(10, 7), title="Neural Gas — Neuron Positions"):
        """
        Plot neuron positions overlaid on data.
        Only works well on 2D projection — use feature_idx to pick axes.

        Parameters
        ----------
        X : array-like, original input data
        labels : array-like, optional class labels for coloring data points
        feature_idx : tuple of 2 ints
            Which two features to plot (default first two)
        """
        if self.neurons is None:
            raise RuntimeError("Train the model first using .fit()")

        X_scaled = self.scaler.transform(X)
        f0, f1 = feature_idx

        fig, ax = plt.subplots(figsize=figsize)

        # Plot data points
        if labels is not None:
            unique = np.unique(labels)
            colors = plt.cm.Set1(np.linspace(0, 1, len(unique)))
            for label, color in zip(unique, colors):
                mask = np.array(labels) == label
                ax.scatter(X_scaled[mask, f0], X_scaled[mask, f1],
                           c=[color], alpha=0.4, s=20, label=str(label))
            ax.legend(title="Class", loc="upper right")
        else:
            ax.scatter(X_scaled[:, f0], X_scaled[:, f1],
                       alpha=0.3, s=20, color='steelblue', label='Data')

        # Plot neurons on top
        ax.scatter(self.neurons[:, f0], self.neurons[:, f1],
                   c='red', s=60, marker='x', linewidths=1.5,
                   zorder=5, label='Neurons')

        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel(f"Feature {f0}")
        ax.set_ylabel(f"Feature {f1}")
        ax.legend()
        plt.tight_layout()
        plt.show()

    def plot_convergence(self, figsize=(9, 4),
                         title="Neural Gas — Training Convergence"):
        """
        Plot quantization error over training epochs.
        Should decrease and flatten — confirms training converged.
        """
        if not self._errors:
            raise RuntimeError("Train the model first using .fit()")

        epochs, errors = zip(*self._errors)

        fig, ax = plt.subplots(figsize=figsize)
        ax.plot(epochs, errors, color='steelblue', linewidth=2)
        ax.fill_between(epochs, errors, alpha=0.1, color='steelblue')
        ax.set_title(title, fontsize=13, fontweight='bold')
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Quantization Error")
        ax.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def compare_with_som(self, X, som_model, labels=None,
                         feature_idx=(0, 1), figsize=(14, 6)):
        """
        Side-by-side comparison of Neural Gas vs SOM neuron placement.

        Parameters
        ----------
        X : array-like, input data
        som_model : kohonen.som.SOM, already trained SOM instance
        labels : array-like, optional class labels
        feature_idx : tuple of 2 ints
        """
        X_scaled = self.scaler.transform(X)
        f0, f1 = feature_idx

        fig, axes = plt.subplots(1, 2, figsize=figsize)

        for ax, (neurons, title) in zip(axes, [
            (self.neurons, "Neural Gas (free placement)"),
            (som_model.som.get_weights().reshape(-1, X_scaled.shape[1]),
             "SOM (grid constrained)")
        ]):
            if labels is not None:
                unique = np.unique(labels)
                colors = plt.cm.Set1(np.linspace(0, 1, len(unique)))
                for label, color in zip(unique, colors):
                    mask = np.array(labels) == label
                    ax.scatter(X_scaled[mask, f0], X_scaled[mask, f1],
                               c=[color], alpha=0.3, s=15)
            else:
                ax.scatter(X_scaled[:, f0], X_scaled[:, f1],
                           alpha=0.3, s=15, color='steelblue')

            ax.scatter(neurons[:, f0], neurons[:, f1],
                       c='red', s=50, marker='x', linewidths=1.5,
                       zorder=5, label='Neurons')
            ax.set_title(title, fontsize=12, fontweight='bold')
            ax.set_xlabel(f"Feature {f0}")
            ax.set_ylabel(f"Feature {f1}")

        plt.suptitle("Neural Gas vs SOM — Neuron Placement",
                     fontsize=13, fontweight='bold')
        plt.tight_layout()
        plt.show()
