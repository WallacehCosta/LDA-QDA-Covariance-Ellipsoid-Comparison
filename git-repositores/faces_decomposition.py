# Faces Dataset Decompositions: A Comparison of Dimensionality Reduction Techniques
# This script applies various unsupervised matrix decomposition (dimension reduction) methods
# from scikit-learn on the Olivetti faces dataset, visualizing the resulting components.

# Author: Wallace de Holanda Costa
# License: MIT License

import logging
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import RandomState
from sklearn import cluster, decomposition
from sklearn.datasets import fetch_olivetti_faces

# Global configuration for the gallery plot
N_ROW, N_COL = 2, 3
N_COMPONENTS = N_ROW * N_COL
IMAGE_SHAPE = (64, 64)

# Display progress logs on stdout
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")


class FacesDecompositionDemo:
    """
    A unified class to load the Olivetti faces dataset and compare various
    unsupervised matrix decomposition techniques (PCA, NMF, ICA, etc.)
    by plotting the extracted components.
    """

    def __init__(self, random_seed=0):
        self.rng = RandomState(random_seed)
        self.faces = None
        self.faces_centered = None
        self.n_samples = 0
        self.n_features = 0
        self._load_and_preprocess_data()

    def _load_and_preprocess_data(self):
        """Loads and preprocesses the Olivetti faces dataset."""
        print("Loading and preprocessing the Olivetti faces dataset...")
        faces, _ = fetch_olivetti_faces(return_X_y=True, shuffle=True, random_state=self.rng)
        self.faces = faces
        self.n_samples, self.n_features = faces.shape

        # Global centering (focus on one feature, centering all samples)
        faces_centered = self.faces - self.faces.mean(axis=0)
        # Local centering (focus on one sample, centering all features)
        faces_centered -= faces_centered.mean(axis=1).reshape(self.n_samples, -1)
        self.faces_centered = faces_centered

        print(f"Dataset consists of {self.n_samples} faces ({self.n_features} features each)")
        print("-" * 50)

    def plot_gallery(self, title, images, cmap=plt.cm.gray):
        """Displays a gallery of images (faces or components)."""
        fig, axs = plt.subplots(
            nrows=N_ROW,
            ncols=N_COL,
            figsize=(2.0 * N_COL, 2.3 * N_ROW),
            facecolor="white",
            constrained_layout=True,
        )
        # Custom layout padding for a tighter, cleaner look
        fig.set_constrained_layout_pads(w_pad=0.01, h_pad=0.02, hspace=0.05, wspace=0.05)
        fig.set_edgecolor("black")
        fig.suptitle(title, size=16, fontweight='bold')  # Added bold to title

        for ax, vec in zip(axs.flat, images):
            vmax = max(vec.max(), -vec.min())
            # Use fixed vmax for better comparison across different decompositions
            im = ax.imshow(
                vec.reshape(IMAGE_SHAPE),
                cmap=cmap,
                interpolation="nearest",
                vmin=-vmax,
                vmax=vmax,
            )
            ax.axis("off")

        # Colorbar for value representation
        fig.colorbar(im, ax=axs, orientation="horizontal", shrink=0.99, aspect=40, pad=0.01)
        plt.show()

    def run_decomposition_comparison(self):
        """Initial display of data and sequential execution of all decomposition methods."""

        # 1. Display original faces (centered)
        self.plot_gallery("Original Centered Faces Sample", self.faces_centered[:N_COMPONENTS], cmap=plt.cm.RdBu)

        # --- List of Decomposition Estimators to Test ---
        decomposition_estimators = [
            # PCA/Eigenfaces (Requires centered data, uses randomized SVD for efficiency)
            (
                "Eigenfaces - PCA (Randomized SVD)",
                decomposition.PCA(n_components=N_COMPONENTS, svd_solver="randomized", whiten=True),
                self.faces_centered,  # Input Data
                'components_',  # Component Attribute
                plt.cm.gray  # Colormap
            ),
            # NMF (Requires non-negative data)
            (
                "Non-negative components - NMF",
                decomposition.NMF(n_components=N_COMPONENTS, tol=5e-3, max_iter=200),
                self.faces,  # Input Data (Original non-negative)
                'components_',
                plt.cm.gray
            ),
            # ICA (Independent components, requires centered data)
            (
                "Independent components - FastICA",
                decomposition.FastICA(
                    n_components=N_COMPONENTS, max_iter=400, whiten="arbitrary-variance", tol=15e-5
                ),
                self.faces_centered,
                'components_',
                plt.cm.gray
            ),
            # Sparse PCA (Mini-Batch for speed)
            (
                "Sparse components - MiniBatchSparsePCA",
                decomposition.MiniBatchSparsePCA(
                    n_components=N_COMPONENTS, alpha=0.1, max_iter=100, batch_size=3, random_state=self.rng
                ),
                self.faces_centered,
                'components_',
                plt.cm.gray
            ),
            # Dictionary Learning (Mini-Batch for speed)
            (
                "Dictionary Learning",
                decomposition.MiniBatchDictionaryLearning(
                    n_components=N_COMPONENTS, alpha=0.1, max_iter=50, batch_size=3, random_state=self.rng
                ),
                self.faces_centered,
                'components_',
                plt.cm.gray
            ),
            # MiniBatchKMeans (Uses cluster centers as components)
            (
                "Cluster Centers - MiniBatchKMeans",
                cluster.MiniBatchKMeans(
                    n_clusters=N_COMPONENTS, tol=1e-3, batch_size=20, max_iter=50, random_state=self.rng, n_init="auto"
                ),
                self.faces_centered,
                'cluster_centers_',  # Different attribute name for components
                plt.cm.gray
            ),
            # Factor Analysis (FA)
            (
                "Factor Analysis (FA)",
                decomposition.FactorAnalysis(n_components=N_COMPONENTS, max_iter=20),
                self.faces_centered,
                'components_',
                plt.cm.gray
            )
        ]

        # --- Run all Decomposition Methods ---
        for title, estimator, data, component_attr, cmap in decomposition_estimators:
            print(f"Fitting {title}...")
            estimator.fit(data)

            # Retrieve components using the specified attribute name
            components = getattr(estimator, component_attr)

            # Plot results
            self.plot_gallery(title, components[:N_COMPONENTS], cmap=cmap)

            # Special Plot for FA Noise Variance
            if title == "Factor Analysis (FA)":
                self._plot_fa_noise_variance(estimator)

    def _plot_fa_noise_variance(self, fa_estimator):
        """Special plotting function for Factor Analysis's pixel-wise variance."""
        plt.figure(figsize=(3.5, 4.0), facecolor="white")
        vec = fa_estimator.noise_variance_
        vmax = max(vec.max(), -vec.min())

        plt.imshow(
            vec.reshape(IMAGE_SHAPE),
            cmap=plt.cm.gray,
            interpolation="nearest",
            vmin=0,  # Noise variance must be non-negative
            vmax=vmax,
        )
        plt.axis("off")
        plt.title("Pixelwise Variance from Factor Analysis (FA)", size=14, wrap=True, fontweight='bold')
        plt.colorbar(orientation="horizontal", shrink=0.8, pad=0.03)
        plt.tight_layout()
        plt.show()

    def run_dictionary_learning_variants(self):
        """Demonstrates the effect of positivity constraints on Dictionary Learning."""
        print("\n--- Running Dictionary Learning Constraint Variants ---")

        # 1. Display original faces with RdBu colormap (Red=Negative, Blue=Positive)
        self.plot_gallery("Faces from dataset (RdBu Colormap)", self.faces_centered[:N_COMPONENTS], cmap=plt.cm.RdBu)

        # Configurations for different positivity constraints
        dl_variants = [
            (
                "DL - Positive Dictionary",
                {'positive_dict': True}
            ),
            (
                "DL - Positive Code",
                {'positive_code': True, 'fit_algorithm': "cd"}  # 'cd' needed for positive code
            ),
            (
                "DL - Positive Dictionary & Code",
                {'positive_dict': True, 'positive_code': True, 'fit_algorithm': "cd"}
            ),
        ]

        base_params = {
            'n_components': N_COMPONENTS,
            'alpha': 0.1,
            'max_iter': 50,
            'batch_size': 3,
            'random_state': self.rng,
        }

        for title, params in dl_variants:
            print(f"Fitting {title}...")
            # Merge base parameters with variant-specific constraints
            current_params = base_params.copy()
            current_params.update(params)

            dl_estimator = decomposition.MiniBatchDictionaryLearning(**current_params)
            dl_estimator.fit(self.faces_centered)
            self.plot_gallery(title, dl_estimator.components_[:N_COMPONENTS], cmap=plt.cm.RdBu)


# --- Execution Block ---
if __name__ == "__main__":
    demo = FacesDecompositionDemo()

    # Run the main comparison of all decomposition methods
    demo.run_decomposition_comparison()

    # Run the focused comparison on Dictionary Learning constraints
    demo.run_dictionary_learning_variants()

    print("\nScript execution finished.")
