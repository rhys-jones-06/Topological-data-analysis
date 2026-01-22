"""
Graph Diffusion Module using NetworkX

Implements Laplacian diffusion on the correlation graph to identify
sectors that are "leaking" or "absorbing" capital (mispricing).
"""

import numpy as np
import pandas as pd
import networkx as nx
from typing import Dict, Optional, Tuple
from scipy.linalg import expm


class GraphDiffusion:
    """
    Graph-based diffusion analysis using Laplacian.
    
    Identifies capital flow patterns and mispricing opportunities
    through heat kernel diffusion on correlation graphs.
    """
    
    def __init__(
        self,
        correlation_threshold: float = 0.3,
        diffusion_time: float = 1.0
    ):
        """
        Initialize the Graph Diffusion model.
        
        Parameters
        ----------
        correlation_threshold : float, optional
            Minimum correlation to create edge. Default is 0.3.
        diffusion_time : float, optional
            Time parameter for heat kernel. Default is 1.0.
        """
        self.correlation_threshold = correlation_threshold
        self.diffusion_time = diffusion_time
        self.graph = None
        self.laplacian = None
    
    def build_correlation_graph(
        self,
        correlation_matrix: np.ndarray,
        asset_names: Optional[list] = None
    ) -> nx.Graph:
        """
        Build a NetworkX graph from correlation matrix.
        
        Parameters
        ----------
        correlation_matrix : np.ndarray
            Correlation matrix between assets
        asset_names : list, optional
            Names of assets. If None, uses indices.
        
        Returns
        -------
        nx.Graph
            Correlation graph
        """
        n_assets = correlation_matrix.shape[0]
        
        if asset_names is None:
            asset_names = [f"Asset_{i}" for i in range(n_assets)]
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(asset_names)
        
        # Add edges based on correlation
        for i in range(n_assets):
            for j in range(i + 1, n_assets):
                corr = abs(correlation_matrix[i, j])
                if corr >= self.correlation_threshold:
                    G.add_edge(asset_names[i], asset_names[j], weight=corr)
        
        self.graph = G
        return G
    
    def compute_laplacian(self, normalized: bool = True) -> np.ndarray:
        """
        Compute the graph Laplacian.
        
        Parameters
        ----------
        normalized : bool, optional
            If True, compute normalized Laplacian. Default is True.
        
        Returns
        -------
        np.ndarray
            Laplacian matrix
        """
        if self.graph is None:
            raise ValueError("Graph must be built first")
        
        if normalized:
            L = nx.normalized_laplacian_matrix(self.graph).toarray()
        else:
            L = nx.laplacian_matrix(self.graph).toarray()
        
        self.laplacian = L
        return L
    
    def heat_kernel_diffusion(
        self,
        initial_state: Optional[np.ndarray] = None,
        time: Optional[float] = None
    ) -> np.ndarray:
        """
        Simulate heat kernel diffusion on the graph.
        
        Parameters
        ----------
        initial_state : np.ndarray, optional
            Initial heat distribution. If None, uses uniform.
        time : float, optional
            Diffusion time. If None, uses self.diffusion_time.
        
        Returns
        -------
        np.ndarray
            Heat distribution after diffusion
        """
        if self.laplacian is None:
            self.compute_laplacian(normalized=True)
        
        n = self.laplacian.shape[0]
        
        if initial_state is None:
            initial_state = np.ones(n) / n
        
        if time is None:
            time = self.diffusion_time
        
        # Heat kernel: exp(-t * L)
        heat_kernel = expm(-time * self.laplacian)
        diffused = heat_kernel @ initial_state
        
        return diffused
    
    def identify_sinks_and_sources(
        self,
        initial_state: Optional[np.ndarray] = None,
        threshold: float = 0.1
    ) -> Dict[str, list]:
        """
        Identify assets that absorb (sinks) or leak (sources) capital.
        
        Parameters
        ----------
        initial_state : np.ndarray, optional
            Initial distribution
        threshold : float, optional
            Threshold for classification. Default is 0.1.
        
        Returns
        -------
        dict
            Dictionary with 'sinks' and 'sources' lists
        """
        if initial_state is None:
            n = self.laplacian.shape[0]
            initial_state = np.ones(n) / n
        
        diffused = self.heat_kernel_diffusion(initial_state)
        
        # Compare final vs initial
        change = diffused - initial_state
        
        asset_names = list(self.graph.nodes())
        
        sinks = [asset_names[i] for i, c in enumerate(change) if c > threshold]
        sources = [asset_names[i] for i, c in enumerate(change) if c < -threshold]
        
        return {
            'sinks': sinks,
            'sources': sources,
            'change': change,
            'diffused': diffused
        }
    
    def compute_graph_metrics(self) -> Dict[str, float]:
        """
        Compute graph-based metrics for regime detection.
        
        Returns
        -------
        dict
            Graph metrics including clustering, density, etc.
        """
        if self.graph is None:
            raise ValueError("Graph must be built first")
        
        metrics = {}
        
        # Basic properties
        metrics['n_nodes'] = self.graph.number_of_nodes()
        metrics['n_edges'] = self.graph.number_of_edges()
        metrics['density'] = nx.density(self.graph)
        
        # Clustering
        metrics['avg_clustering'] = nx.average_clustering(self.graph)
        
        # Connected components
        metrics['n_components'] = nx.number_connected_components(self.graph)
        
        # Centrality (average)
        if metrics['n_edges'] > 0:
            degree_centrality = nx.degree_centrality(self.graph)
            metrics['avg_degree_centrality'] = np.mean(list(degree_centrality.values()))
        else:
            metrics['avg_degree_centrality'] = 0.0
        
        # Laplacian spectrum (indicator of graph structure)
        if self.laplacian is not None:
            eigenvalues = np.linalg.eigvalsh(self.laplacian)
            metrics['spectral_gap'] = eigenvalues[1] - eigenvalues[0]  # Algebraic connectivity
            metrics['max_eigenvalue'] = eigenvalues[-1]
        
        return metrics
    
    def compute_leakage_score(
        self,
        returns: Optional[np.ndarray] = None
    ) -> float:
        """
        Compute a leakage score indicating market fragmentation.
        
        Higher scores indicate more fragmentation/leakage.
        
        Parameters
        ----------
        returns : np.ndarray, optional
            Recent returns to weight initial state
        
        Returns
        -------
        float
            Leakage score (0 to 1, higher = more fragmentation)
        """
        metrics = self.compute_graph_metrics()
        
        # Leakage is indicated by:
        # - Low density (disconnected assets)
        # - High number of components
        # - Low clustering
        
        leakage_score = 0.0
        
        # Low density contributes to leakage
        leakage_score += (1.0 - metrics['density']) * 0.4
        
        # Multiple components indicate fragmentation
        if metrics['n_components'] > 1:
            component_penalty = min(metrics['n_components'] / 5.0, 1.0)
            leakage_score += component_penalty * 0.3
        
        # Low clustering indicates lack of coherence
        leakage_score += (1.0 - metrics['avg_clustering']) * 0.3
        
        return min(leakage_score, 1.0)


def create_sample_graph_diffusion(
    correlation_threshold: float = 0.3,
    diffusion_time: float = 1.0
) -> GraphDiffusion:
    """
    Factory function to create a sample GraphDiffusion with default parameters.
    
    Parameters
    ----------
    correlation_threshold : float, optional
        Minimum correlation to create edge. Default is 0.3.
    diffusion_time : float, optional
        Time parameter for heat kernel. Default is 1.0.
    
    Returns
    -------
    GraphDiffusion
        Configured graph diffusion model
    """
    return GraphDiffusion(
        correlation_threshold=correlation_threshold,
        diffusion_time=diffusion_time
    )
