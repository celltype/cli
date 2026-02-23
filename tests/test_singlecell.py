"""Tests for single-cell analysis tools: cluster, trajectory, cell_type_annotate."""

import pytest
import pandas as pd
import numpy as np
from unittest.mock import patch, MagicMock


# ─── singlecell.cluster ──────────────────────────────────────


class TestSinglecellCluster:
    """Tests for singlecell.cluster."""

    def test_missing_scanpy_returns_error(self):
        """If scanpy is not installed, return helpful error."""
        with patch("ct.tools.singlecell._check_scanpy", return_value=None):
            from ct.tools.singlecell import cluster
            result = cluster(data_path="test.h5ad")
            assert "error" in result
            assert "scanpy" in result["error"].lower()
            assert "summary" in result

    def test_unsupported_format_returns_error(self):
        """Unsupported file formats should return an error."""
        mock_sc = MagicMock()
        with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc):
            from ct.tools.singlecell import cluster
            result = cluster(data_path="data.txt")
            assert "error" in result
            assert "Unsupported" in result["error"]

    def test_cluster_h5ad_success(self):
        """Full clustering pipeline on mock h5ad data."""
        mock_sc = MagicMock()

        n_cells, n_genes = 500, 3000

        adata = MagicMock()
        adata.shape = (n_cells, n_genes)
        adata.X = MagicMock()
        adata.X.copy = MagicMock(return_value=np.random.randn(n_cells, n_genes))
        adata.layers = {}
        adata.var = MagicMock()
        adata.var.__getitem__ = MagicMock(return_value=np.ones(n_genes, dtype=bool))

        # Mock slicing adata[:, mask]
        adata_hvg = MagicMock()
        adata_hvg.shape = (n_cells, 2000)
        adata_hvg.obsm = {}
        adata_hvg.copy = MagicMock(return_value=adata_hvg)
        adata.__getitem__ = MagicMock(return_value=adata_hvg)

        # After PCA
        adata_hvg.obsm = {"X_pca": np.random.randn(n_cells, 50)}
        adata.obsm = {}

        # Cluster assignments — use pd.Series so .astype(str) returns a Series
        cluster_labels = pd.Series([str(i % 5) for i in range(n_cells)], dtype="category")
        adata.obs = {"cluster": cluster_labels}

        # UMAP coords
        adata.obsm["X_umap"] = np.random.randn(n_cells, 2)

        # Marker genes
        adata.uns = {
            "rank_genes_groups": {
                "names": {str(i): np.array(["GENE_A", "GENE_B", "GENE_C", "GENE_D", "GENE_E"]) for i in range(5)},
                "scores": {str(i): np.array([10.0, 8.0, 6.0, 4.0, 2.0]) for i in range(5)},
            }
        }

        mock_sc.read_h5ad = MagicMock(return_value=adata)

        with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc):
            from ct.tools.singlecell import cluster
            result = cluster(data_path="test.h5ad", resolution=0.8, method="leiden")

        assert "summary" in result
        assert result["n_cells"] == n_cells
        assert result["n_genes"] == n_genes
        assert result["resolution"] == 0.8
        assert "cluster" in result["summary"].lower()

    def test_cluster_csv_success(self):
        """Clustering from CSV format."""
        mock_sc = MagicMock()

        n_cells, n_genes = 50, 100

        adata = MagicMock()
        adata.shape = (n_cells, n_genes)
        raw_X = np.random.randn(n_cells, n_genes).astype(np.float32)
        mock_X = MagicMock()
        mock_X.copy = MagicMock(return_value=raw_X.copy())
        mock_X.mean = MagicMock(return_value=raw_X.mean(axis=0))
        adata.X = mock_X
        adata.layers = {}
        adata.var = MagicMock()

        # No HVG filtering needed (< 2000 genes)
        adata_copy = MagicMock()
        adata_copy.shape = (n_cells, n_genes)
        adata_copy.obsm = {"X_pca": np.random.randn(n_cells, 50)}
        adata.copy = MagicMock(return_value=adata_copy)

        adata.obsm = {}

        cluster_labels = pd.Series([str(i % 3) for i in range(n_cells)], dtype="category")
        adata.obs = {"cluster": cluster_labels}
        adata.obsm["X_umap"] = np.random.randn(n_cells, 2)
        adata.uns = {
            "rank_genes_groups": {
                "names": {str(i): np.array(["G1", "G2", "G3", "G4", "G5"]) for i in range(3)},
                "scores": {str(i): np.array([5.0, 4.0, 3.0, 2.0, 1.0]) for i in range(3)},
            }
        }

        mock_df = pd.DataFrame(
            np.random.randn(n_cells, n_genes),
            index=[f"cell_{i}" for i in range(n_cells)],
            columns=[f"gene_{i}" for i in range(n_genes)],
        )

        # Create a mock anndata module if it's not installed
        import sys
        mock_anndata = MagicMock()
        mock_anndata.AnnData = MagicMock(return_value=adata)
        with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc), \
             patch("pandas.read_csv", return_value=mock_df), \
             patch.dict(sys.modules, {"anndata": mock_anndata}):
            from ct.tools.singlecell import cluster
            result = cluster(data_path="data.csv", resolution=1.0)

        assert "summary" in result
        assert result["n_cells"] == n_cells


# ─── singlecell.trajectory ───────────────────────────────────


class TestSinglecellTrajectory:
    """Tests for singlecell.trajectory."""

    def test_missing_scanpy_returns_error(self):
        with patch("ct.tools.singlecell._check_scanpy", return_value=None):
            from ct.tools.singlecell import trajectory
            result = trajectory(data_path="test.h5ad")
            assert "error" in result
            assert "scanpy" in result["error"].lower()

    def test_non_h5ad_returns_error(self):
        mock_sc = MagicMock()
        with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc):
            from ct.tools.singlecell import trajectory
            result = trajectory(data_path="data.csv")
            assert "error" in result

    def test_trajectory_success(self):
        """Full trajectory inference on mock data."""
        mock_sc = MagicMock()

        n_cells = 100
        n_clusters = 4

        adata = MagicMock()
        adata.shape = (n_cells, 500)

        # Neighbors computed
        adata.uns = {"neighbors": {}}

        # Cluster assignments — use pd.Series
        cluster_labels = pd.Series([str(i % n_clusters) for i in range(n_cells)], dtype="category")

        # Diffmap output
        diffmap = np.random.randn(n_cells, 15)
        adata.obsm = {"X_diffmap": diffmap}

        # PAGA connectivities
        from scipy.sparse import csr_matrix
        paga_conn = csr_matrix(np.array([
            [0.0, 0.5, 0.2, 0.0],
            [0.5, 0.0, 0.3, 0.4],
            [0.2, 0.3, 0.0, 0.6],
            [0.0, 0.4, 0.6, 0.0],
        ]))

        # Pseudotime
        pseudotime = np.linspace(0, 1, n_cells)

        # Use a real dict-like obs that supports __contains__ and __getitem__
        class MockObs:
            def __init__(self):
                self.columns = ["cluster"]
                self._data = {"cluster": cluster_labels}

            def __contains__(self, key):
                return key in self._data or key in self.columns

            def __getitem__(self, key):
                return self._data[key]

            def __setitem__(self, key, value):
                self._data[key] = value

        adata.obs = MockObs()

        mock_sc.read_h5ad = MagicMock(return_value=adata)

        # Mock scanpy functions
        def mock_diffmap(ad, **kw):
            ad.obsm["X_diffmap"] = diffmap

        def mock_paga(ad, **kw):
            ad.uns["paga"] = {"connectivities": paga_conn}

        def mock_dpt(ad, **kw):
            ad.obs["dpt_pseudotime"] = pd.Series(pseudotime)

        mock_sc.tl.diffmap = MagicMock(side_effect=mock_diffmap)
        mock_sc.tl.paga = MagicMock(side_effect=mock_paga)
        mock_sc.tl.dpt = MagicMock(side_effect=mock_dpt)

        with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc):
            from ct.tools.singlecell import trajectory
            result = trajectory(data_path="test.h5ad", root_cluster="0")

        assert "summary" in result
        assert result["n_cells"] == n_cells
        assert result["root_cluster"] == "0"
        assert "pseudotime_range" in result
        assert "branch_points" in result

    def test_missing_clusters_returns_error(self):
        """If no cluster column exists, return error."""
        mock_sc = MagicMock()

        adata = MagicMock()
        adata.shape = (50, 100)
        adata.uns = {"neighbors": {}}

        # Use a real dict-like obs without cluster columns
        class MockObs:
            def __init__(self):
                self.columns = ["other_col"]

            def __contains__(self, key):
                return False

        adata.obs = MockObs()

        mock_sc.read_h5ad = MagicMock(return_value=adata)

        with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc):
            from ct.tools.singlecell import trajectory
            result = trajectory(data_path="test.h5ad")
            assert "error" in result
            assert "cluster" in result["error"].lower()


# ─── singlecell.cell_type_annotate ───────────────────────────


class TestSinglecellCellTypeAnnotate:
    """Tests for singlecell.cell_type_annotate."""

    def test_missing_scanpy_returns_error(self):
        with patch("ct.tools.singlecell._check_scanpy", return_value=None):
            from ct.tools.singlecell import cell_type_annotate
            result = cell_type_annotate(data_path="test.h5ad")
            assert "error" in result
            assert "scanpy" in result["error"].lower()

    def test_celltypist_not_installed(self):
        """When celltypist method requested but not installed."""
        # Patch the import inside the function to raise ImportError
        from ct.tools.singlecell import cell_type_annotate

        original_import = __builtins__.__import__ if hasattr(__builtins__, '__import__') else __import__

        def mock_import(name, *args, **kwargs):
            if name == "celltypist":
                raise ImportError("No module named 'celltypist'")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=mock_import):
            result = cell_type_annotate(data_path="test.h5ad", method="celltypist")
            assert "error" in result
            assert "celltypist" in result["error"].lower()

    def test_marker_based_annotation(self):
        """Marker-based annotation identifies cell types from marker genes."""
        mock_sc = MagicMock()

        n_cells = 80
        n_genes = 50
        n_clusters = 4

        # Build gene names with markers
        gene_names = [f"GENE_{i}" for i in range(n_genes)]
        gene_names[0] = "CD3D"
        gene_names[1] = "CD3E"
        gene_names[2] = "CD79A"
        gene_names[3] = "MS4A1"
        gene_names[4] = "NKG7"
        gene_names[5] = "GNLY"
        gene_names[6] = "LYZ"
        gene_names[7] = "S100A8"

        # Create expression matrix where cluster 0 has high T cell markers
        X = np.random.randn(n_cells, n_genes).astype(np.float32) * 0.1
        X[:20, 0] = 5.0   # CD3D
        X[:20, 1] = 4.0   # CD3E
        X[20:40, 2] = 5.0  # CD79A
        X[20:40, 3] = 4.0  # MS4A1
        X[40:60, 4] = 5.0  # NKG7
        X[40:60, 5] = 4.0  # GNLY
        X[60:80, 6] = 5.0  # LYZ
        X[60:80, 7] = 4.0  # S100A8

        adata = MagicMock()
        adata.shape = (n_cells, n_genes)
        adata.X = X
        adata.var_names = np.array(gene_names)

        cluster_labels = pd.Series([str(i // 20) for i in range(n_cells)], dtype="category")

        class MockObs:
            def __init__(self):
                self.columns = ["cluster"]
                self._data = {"cluster": cluster_labels}

            def __contains__(self, key):
                return key in self._data

            def __getitem__(self, key):
                return self._data[key]

        adata.obs = MockObs()

        mock_sc.read_h5ad = MagicMock(return_value=adata)

        with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc):
            from ct.tools.singlecell import cell_type_annotate
            result = cell_type_annotate(data_path="test.h5ad", reference="immune", method="marker_based")

        assert "summary" in result
        assert result["n_clusters"] == n_clusters
        assert result["method"] == "marker_based"
        assert "annotations" in result
        assert len(result["annotations"]) == n_clusters
        for cl, ann in result["annotations"].items():
            assert "cell_type" in ann
            assert "confidence" in ann
            assert "n_cells" in ann

    def test_no_cluster_column_returns_error(self):
        """If data has no clusters, return error."""
        mock_sc = MagicMock()

        adata = MagicMock()
        adata.shape = (50, 100)
        adata.var_names = np.array([f"G{i}" for i in range(100)])

        class MockObs:
            def __init__(self):
                self.columns = ["other"]

            def __contains__(self, key):
                return False

        adata.obs = MockObs()

        mock_sc.read_h5ad = MagicMock(return_value=adata)

        with patch("ct.tools.singlecell._check_scanpy", return_value=mock_sc):
            from ct.tools.singlecell import cell_type_annotate
            result = cell_type_annotate(data_path="test.h5ad")
            assert "error" in result
