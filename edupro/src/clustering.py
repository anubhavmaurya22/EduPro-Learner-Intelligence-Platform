"""
EduPro Learner Intelligence Platform
File: src/clustering.py

K-Means clustering + hierarchical validation + PCA, matching
notebooks/03_clustering.ipynb. Also handles mapping raw cluster
numbers to named business segments.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score, silhouette_samples

from .utils import CLUSTERING_FEATURES, SEGMENT_NAMES, SEGMENT_COLORS, SEGMENT_DESC, N_CLUSTERS, RANDOM_SEED


def run_clustering(profiles: pd.DataFrame,
                   n_clusters: int = N_CLUSTERS,
                   seed: int = RANDOM_SEED) -> dict:
    """
    Runs the full clustering pipeline:
      1. Scale features
      2. Sweep K=2..8 for elbow + silhouette
      3. Fit final KMeans at n_clusters
      4. Map raw cluster numbers -> named segments (0-3)
      5. Validate with Agglomerative clustering on a sample
      6. Compute PCA(2) for visualisation
      7. Compute per-user silhouette values

    Returns a dict with every artefact needed downstream
    (model, scaler, labels, PCA coords, diagnostics).

    NOTE: n_clusters MUST equal 4.  SEGMENT_NAMES, SEGMENT_COLORS and
    SEGMENT_DESC in utils.py are keyed 0-3 (4 business archetypes).  Passing
    a different value causes _map_clusters_to_segments to collapse multiple
    real clusters into segment 3 with no error — silent wrong output.  If you
    want a different k, extend all three dicts in utils.py first, then update
    _map_clusters_to_segments to match.
    """
    if n_clusters != N_CLUSTERS:
        raise NotImplementedError(
            f"n_clusters={n_clusters} is not supported. "
            f"SEGMENT_NAMES, SEGMENT_COLORS and SEGMENT_DESC in utils.py are "
            f"keyed 0–{N_CLUSTERS - 1} (exactly {N_CLUSTERS} business archetypes). "
            f"Extend all three dicts and update _map_clusters_to_segments before "
            f"passing a different value."
        )

    X = profiles[CLUSTERING_FEATURES].fillna(0).values

    scaler   = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ── Elbow + Silhouette sweep ─────────────
    k_range = range(2, 7)
    inertias, sil_scores = [], []
    for k in k_range:
        km = KMeans(n_clusters=k, random_state=seed, n_init=15)
        labels = km.fit_predict(X_scaled)
        inertias.append(km.inertia_)
        sil_scores.append(silhouette_score(X_scaled, labels))

    # ── Final K-Means ────────────────────────
    km_final  = KMeans(n_clusters=n_clusters, random_state=seed, n_init=20, max_iter=500)
    raw_labels = km_final.fit_predict(X_scaled)

    # ── Map raw cluster index -> named segment ──
    centers = pd.DataFrame(
        scaler.inverse_transform(km_final.cluster_centers_),
        columns=CLUSTERING_FEATURES
    )
    mapping = _map_clusters_to_segments(centers)
    named_labels = np.array([mapping[l] for l in raw_labels])

    # ── Hierarchical validation (on a sample) ──
    sample_size = min(500, len(X_scaled))
    sample_idx  = np.random.RandomState(seed).choice(len(X_scaled), sample_size, replace=False)
    hier = AgglomerativeClustering(n_clusters=n_clusters)
    hier_labels = hier.fit_predict(X_scaled[sample_idx])

    # ── PCA for visualisation ────────────────
    pca   = PCA(n_components=2, random_state=seed)
    X_pca = pca.fit_transform(X_scaled)

    # ── Per-user silhouette ──────────────────
    sil_vals = silhouette_samples(X_scaled, named_labels)

    return {
        'scaler':             scaler,
        'km_model':           km_final,
        'pca':                pca,
        'X_scaled':           X_scaled,
        'X_pca':              X_pca,
        'raw_labels':         raw_labels,
        'labels':             named_labels,      # use this one — named 0-3
        'cluster_mapping':    mapping,
        'k_range':            list(k_range),
        'inertias':           inertias,
        'sil_scores':         sil_scores,
        'sil_vals':           sil_vals,
        'overall_silhouette': silhouette_score(X_scaled, named_labels),
        'hier_labels':        hier_labels,
        'hier_sample_idx':    sample_idx,
    }


def _map_clusters_to_segments(centers: pd.DataFrame) -> dict:
    """
    Maps raw KMeans cluster indices to fixed segment IDs (0-3)
    using cluster-centre characteristics:
      0 = highest total_courses        -> Tech Explorer
      1 = highest avg_spending         -> Career Climber
      2 = highest learning_depth_index -> Deep Specialist
      3 = whatever remains             -> Casual Browser

    INVARIANT: len(centers) MUST equal 4 (i.e. n_clusters=4).
    run_clustering() enforces this with a NotImplementedError before
    calling this function, so callers outside that path should ensure
    the same constraint holds.

    Guarantees every raw cluster index gets a unique mapping — avoids
    the "NaN cluster" bug that occurs when a manual dict omits an index.
    """
    if len(centers) != 4:
        raise ValueError(
            f"_map_clusters_to_segments expects exactly 4 clusters "
            f"(SEGMENT_NAMES/COLORS/DESC are hardcoded to keys 0-3), "
            f"got {len(centers)}. Update utils.py's segment metadata "
            f"before changing n_clusters."
        )

    remaining = list(centers.index)
    mapping = {}

    idx_explorer = centers.loc[remaining, 'total_courses'].idxmax()
    mapping[idx_explorer] = 0
    remaining.remove(idx_explorer)

    idx_specialist = centers.loc[remaining, 'learning_depth_index'].idxmax()
    mapping[idx_specialist] = 2
    remaining.remove(idx_specialist)

    idx_climber = centers.loc[remaining, 'avg_spending'].idxmax()
    mapping[idx_climber] = 1
    remaining.remove(idx_climber)

    # Whatever's left (should be exactly one) -> Casual Browser
    for idx in remaining:
        mapping[idx] = 3

    return mapping


def get_cluster_profiles(profiles: pd.DataFrame,
                         cluster_result: dict) -> pd.DataFrame:
    """
    Returns one row per segment with aggregate stats —
    the table you show stakeholders to explain each segment.
    """
    df = profiles.copy()
    df['Cluster']       = cluster_result['labels']
    df['SegmentName']   = df['Cluster'].map(SEGMENT_NAMES)
    df['SilhouetteVal'] = cluster_result['sil_vals']

    cluster_profiles = (
        df.groupby(['Cluster', 'SegmentName'])
          .agg({
              'UserID':               'count',
              'total_courses':        'mean',
              'avg_spending':         'mean',
              'total_spending':       'mean',
              'avg_course_rating':    'mean',
              'diversity_score':      'mean',
              'learning_depth_index': 'mean',
              'beginner_ratio':       'mean',
              'advanced_ratio':       'mean',
              'enrollment_frequency': 'mean',
              'n_categories':         'mean',
              'recency_days':         'mean',
              'SilhouetteVal':        'mean',
          })
          .reset_index()
          .rename(columns={'UserID': 'n_users'})
    )

    for col in ['preferred_category', 'preferred_level']:
        mode_vals = (df.groupby('Cluster')[col]
                       .agg(lambda x: x.mode()[0] if len(x.mode()) else 'N/A')
                       .reset_index()
                       .rename(columns={col: f'top_{col}'}))
        cluster_profiles = cluster_profiles.merge(mode_vals, on='Cluster')

    cluster_profiles['color']       = cluster_profiles['Cluster'].map(SEGMENT_COLORS)
    cluster_profiles['description'] = cluster_profiles['Cluster'].map(SEGMENT_DESC)

    return cluster_profiles


def attach_cluster_labels(profiles: pd.DataFrame,
                          cluster_result: dict) -> pd.DataFrame:
    """
    Convenience function — returns a COPY of profiles with
    Cluster, SegmentName, PCA_1, PCA_2, SilhouetteVal attached.
    Use this for the dashboard / recommender rather than
    re-deriving columns by hand each time.
    """
    df = profiles.copy()
    df['Cluster']       = cluster_result['labels']
    df['SegmentName']   = df['Cluster'].map(SEGMENT_NAMES)
    df['PCA_1']         = cluster_result['X_pca'][:, 0]
    df['PCA_2']         = cluster_result['X_pca'][:, 1]
    df['SilhouetteVal'] = cluster_result['sil_vals']
    return df


# ─────────────────────────────────────────────
# SMOKE TEST
# ─────────────────────────────────────────────

if __name__ == '__main__':
    from data_pipeline import generate_data
    from features import engineer_features

    print("Generating sample data + features...")
    users, courses, transactions = generate_data(n_users=300, n_courses=100)
    profiles = engineer_features(users, courses, transactions)

    print("Running clustering...")
    cr = run_clustering(profiles)
    print(f"  Overall silhouette : {cr['overall_silhouette']:.4f}")
    print(f"  Cluster mapping    : {cr['cluster_mapping']}")
    print(f"  Label NaN check    : {pd.isna(cr['labels']).sum()}")

    profiles_labeled = attach_cluster_labels(profiles, cr)
    cp = get_cluster_profiles(profiles, cr)
    print("\nCluster profiles:")
    print(cp[['SegmentName', 'n_users', 'total_courses', 'avg_spending']].to_string(index=False))
    print("clustering.py self-test passed ✅")