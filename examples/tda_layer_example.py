"""
Example demonstrating the TDA Layer for Financial Data Analysis.

This script shows how to use topological data analysis to extract
global market shape from financial correlation data, including:

1. Vietoris-Rips filtration construction
2. Persistent homology computation
3. H_0 (clusters) and H_1 (loops/cycles) feature extraction
4. Feature vectorization for machine learning
5. Market regime identification
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd
from tda import (
    compute_rolling_correlation,
    build_vietoris_rips_filtration,
    compute_persistent_homology,
    extract_h0_features,
    extract_h1_features,
    compute_persistence_landscape,
    compute_persistence_images,
    vectorize_persistence_diagrams,
    identify_market_regimes
)


def generate_market_scenarios(n_assets=10, n_periods=100):
    """
    Generate different market scenarios for demonstration.
    
    Returns three different market correlation structures:
    1. Unified market: All assets highly correlated
    2. Fragmented market: Multiple distinct clusters
    3. Crisis market: Complex feedback loops
    """
    np.random.seed(42)
    
    scenarios = {}
    
    # Scenario 1: Unified Market - High correlation across all assets
    print("Generating Scenario 1: Unified Market")
    base_correlation = 0.7
    unified_corr = np.eye(n_assets)
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            corr_val = base_correlation + np.random.uniform(-0.1, 0.1)
            unified_corr[i, j] = unified_corr[j, i] = corr_val
    scenarios['unified'] = unified_corr
    
    # Scenario 2: Fragmented Market - Three distinct clusters
    print("Generating Scenario 2: Fragmented Market")
    fragmented_corr = np.eye(n_assets)
    cluster_size = n_assets // 3
    # High correlation within clusters, low between
    for i in range(n_assets):
        for j in range(i+1, n_assets):
            if i // cluster_size == j // cluster_size:
                # Same cluster - high correlation
                corr_val = 0.8 + np.random.uniform(-0.05, 0.05)
            else:
                # Different cluster - low correlation
                corr_val = 0.2 + np.random.uniform(-0.1, 0.1)
            fragmented_corr[i, j] = fragmented_corr[j, i] = corr_val
    scenarios['fragmented'] = fragmented_corr
    
    # Scenario 3: Crisis Market - Complex structure with cycles
    print("Generating Scenario 3: Crisis Market with Feedback Loops")
    crisis_corr = np.eye(n_assets)
    # Create circular dependencies (feedback loops)
    for i in range(n_assets):
        # Strong correlation with next asset (circular)
        next_asset = (i + 1) % n_assets
        crisis_corr[i, next_asset] = crisis_corr[next_asset, i] = 0.85
        # Moderate correlation with asset two steps away
        two_away = (i + 2) % n_assets
        crisis_corr[i, two_away] = crisis_corr[two_away, i] = 0.5
        # Weak random correlations with others
        for j in range(i+3, n_assets):
            corr_val = np.random.uniform(0.1, 0.3)
            crisis_corr[i, j] = crisis_corr[j, i] = corr_val
    scenarios['crisis'] = crisis_corr
    
    return scenarios


def analyze_market_scenario(name, correlation_matrix):
    """Perform complete TDA analysis on a market scenario."""
    print(f"\n{'='*80}")
    print(f"Analyzing: {name.upper()} MARKET")
    print(f"{'='*80}\n")
    
    # 1. Build Vietoris-Rips filtration
    print("1. Building Vietoris-Rips Filtration...")
    vr_result = build_vietoris_rips_filtration(
        correlation_matrix,
        max_dimension=2
    )
    print(f"   ✓ Filtration constructed with {len(vr_result['dgms'])} homology dimensions")
    print(f"   ✓ Distance matrix shape: {vr_result['distance_matrix'].shape}")
    
    # 2. Examine persistence diagrams
    print("\n2. Examining Persistence Diagrams...")
    for i, dgm in enumerate(vr_result['dgms']):
        finite_dgm = dgm[dgm[:, 1] < np.inf]
        print(f"   H_{i}: {len(finite_dgm)} finite features")
        if len(finite_dgm) > 0:
            persistences = finite_dgm[:, 1] - finite_dgm[:, 0]
            print(f"        Max persistence: {np.max(persistences):.4f}")
            print(f"        Mean persistence: {np.mean(persistences):.4f}")
    
    # 3. Extract H_0 features (clusters)
    print("\n3. Extracting H_0 Features (Market Clusters)...")
    h0_dgm = vr_result['dgms'][0]
    h0_features = extract_h0_features(h0_dgm)
    print(f"   Number of clusters: {h0_features['num_components']}")
    print(f"   Max persistence: {h0_features['max_persistence']:.4f}")
    print(f"   Mean persistence: {h0_features['mean_persistence']:.4f}")
    print(f"   Persistence entropy: {h0_features['persistence_entropy']:.4f}")
    
    # 4. Extract H_1 features (loops/cycles)
    print("\n4. Extracting H_1 Features (Market Feedback Loops)...")
    h1_dgm = vr_result['dgms'][1]
    h1_features = extract_h1_features(h1_dgm)
    print(f"   Number of loops: {h1_features['num_loops']}")
    print(f"   Max persistence: {h1_features['max_persistence']:.4f}")
    print(f"   Total persistence: {h1_features['total_persistence']:.4f}")
    if h1_features['num_loops'] > 0:
        print(f"   Mean persistence: {h1_features['mean_persistence']:.4f}")
    
    # 5. Identify market regime
    print("\n5. Identifying Market Regime...")
    regime = identify_market_regimes(h0_features, h1_features)
    print(f"   Regime: {regime['regime']}")
    print(f"   Is fragmented: {regime['is_fragmented']}")
    print(f"   Has feedback cycles: {regime['has_cycles']}")
    
    # 6. Compute feature vectorizations
    print("\n6. Computing Feature Vectorizations for ML...")
    
    # Statistical features
    stat_features = vectorize_persistence_diagrams(vr_result, method='statistics')
    print(f"   ✓ Statistical features extracted")
    print(f"     H_0 keys: {list(stat_features['H_0'].keys())}")
    print(f"     H_1 keys: {list(stat_features['H_1'].keys())}")
    
    # Persistence landscapes
    landscape_features = vectorize_persistence_diagrams(
        vr_result,
        method='landscape',
        k=3,
        num_samples=50
    )
    print(f"   ✓ Persistence landscapes computed")
    print(f"     H_0 landscape shape: {landscape_features['H_0'].shape}")
    print(f"     H_1 landscape shape: {landscape_features['H_1'].shape}")
    
    # Persistence images
    image_features = vectorize_persistence_diagrams(
        vr_result,
        method='image',
        resolution=(15, 15)
    )
    print(f"   ✓ Persistence images computed")
    print(f"     H_0 image shape: {image_features['H_0'].shape}")
    print(f"     H_1 image shape: {image_features['H_1'].shape}")
    
    return {
        'vr_result': vr_result,
        'h0_features': h0_features,
        'h1_features': h1_features,
        'regime': regime,
        'stat_features': stat_features,
        'landscape_features': landscape_features,
        'image_features': image_features
    }


def main():
    print("=" * 80)
    print("TDA LAYER DEMONSTRATION - Extracting Global Market Shape")
    print("=" * 80)
    print()
    print("This example demonstrates topological data analysis on financial markets.")
    print("We analyze three different market scenarios:")
    print("  1. Unified Market - All assets move together")
    print("  2. Fragmented Market - Distinct sector clusters")
    print("  3. Crisis Market - Complex feedback loops")
    print()
    
    # Generate market scenarios
    scenarios = generate_market_scenarios(n_assets=10, n_periods=100)
    
    # Analyze each scenario
    results = {}
    for scenario_name, corr_matrix in scenarios.items():
        results[scenario_name] = analyze_market_scenario(scenario_name, corr_matrix)
    
    # Summary comparison
    print(f"\n{'='*80}")
    print("SUMMARY: Comparing Market Regimes")
    print(f"{'='*80}\n")
    
    comparison_data = []
    for scenario_name, result in results.items():
        comparison_data.append({
            'Scenario': scenario_name.capitalize(),
            'Regime': result['regime']['regime'],
            'Clusters (H_0)': result['h0_features']['num_components'],
            'Loops (H_1)': result['h1_features']['num_loops'],
            'Is Fragmented': '✓' if result['regime']['is_fragmented'] else '✗',
            'Has Cycles': '✓' if result['regime']['has_cycles'] else '✗'
        })
    
    df = pd.DataFrame(comparison_data)
    print(df.to_string(index=False))
    print()
    
    # Practical application notes
    print(f"{'='*80}")
    print("PRACTICAL APPLICATIONS")
    print(f"{'='*80}\n")
    print("These topological features can be used for:")
    print()
    print("1. REGIME DETECTION:")
    print("   - Use H_0 features to detect market fragmentation")
    print("   - Monitor H_1 features to identify feedback loops")
    print("   - Track changes in persistence over time")
    print()
    print("2. RISK MANAGEMENT:")
    print("   - Fragmented markets → Higher diversification benefit")
    print("   - Feedback loops → Potential for contagion/crisis")
    print("   - Unified markets → Systematic risk dominates")
    print()
    print("3. MACHINE LEARNING:")
    print("   - Use persistence landscapes as features for prediction")
    print("   - Persistence images work well with CNNs")
    print("   - Statistical features for traditional ML models")
    print()
    print("4. PORTFOLIO CONSTRUCTION:")
    print("   - Identify natural market clusters from H_0")
    print("   - Avoid over-concentration in single clusters")
    print("   - Monitor for emerging feedback loops")
    print()
    
    print(f"{'='*80}")
    print("Analysis Complete!")
    print(f"{'='*80}")


if __name__ == '__main__':
    main()
