"""
Module: Regime Clustering & Shift Detection
Description: Uses PCA and t-SNE to detect operational regimes and visualize clusters in the last 96-minute window.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from darts import TimeSeries
from darts.dataprocessing.transformers import Scaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

def run_regime_clustering(csv_path, output_dir, window_minutes=96, target_col="Sensor 1 [Hydrocarbon_Dew_Point_C]"):
    print("📈 Running regime shift detection and clustering...")

    # Load and preprocess
    df = pd.read_csv(csv_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'], dayfirst=True)
    df.set_index('timestamp', inplace=True)
    df.sort_index(inplace=True)

    recent_df = df.tail(window_minutes)
    features = recent_df.drop(columns=[target_col], errors='ignore')  # drop target if needed
    scaler = Scaler()
    scaled = scaler.fit_transform(TimeSeries.from_dataframe(features)).pd_dataframe()

    # PCA
    pca = PCA(n_components=2)
    pca_result = pca.fit_transform(scaled)
    pca_path = os.path.join(output_dir, "pca_projection.png")
    plt.figure(figsize=(8, 6))
    plt.scatter(pca_result[:, 0], pca_result[:, 1], c='blue')
    plt.title("PCA - Operational Regimes (Last 96 minutes)")
    plt.savefig(pca_path)
    plt.close()

    # t-SNE
    tsne = TSNE(n_components=2, perplexity=10, n_iter=1000, random_state=42)
    tsne_result = tsne.fit_transform(scaled)
    tsne_path = os.path.join(output_dir, "tsne_projection.png")
    plt.figure(figsize=(8, 6))
    plt.scatter(tsne_result[:, 0], tsne_result[:, 1], c='green')
    plt.title("t-SNE - Operational Regimes (Last 96 minutes)")
    plt.savefig(tsne_path)
    plt.close()

    print(f"✅ Regime clustering plots saved to:\n- {pca_path}\n- {tsne_path}")
    return pca_path, tsne_path
