# =============================================================================
# WHERE DOES AFROBEATS SIT IN THE GLOBAL MUSIC LANDSCAPE?
# An Unsupervised Learning Deep Dive
# =============================================================================
# Tools: PCA, K-Means, Hierarchical Clustering, t-SNE, NMF
# Dataset: Spotify Tracks Dataset (Maharshi Pandya, Kaggle)
# Author: [Your Name]
# =============================================================================


# =============================================================================
# SECTION 1: IMPORTS & SETUP
# =============================================================================

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns

from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.decomposition import PCA, NMF
from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
from sklearn.metrics import silhouette_score

from scipy.cluster.hierarchy import linkage, dendrogram, fcluster

import warnings
warnings.filterwarnings('ignore')

# ── Consistent plot style throughout ──────────────────────────────────────────
plt.rcParams['figure.figsize'] = (13, 7)
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['axes.spines.top'] = False
plt.rcParams['axes.spines.right'] = False
sns.set_style('whitegrid')

print("✅ All libraries imported successfully")


# =============================================================================
# SECTION 2: LOAD & FIRST LOOK
# =============================================================================

df = pd.read_csv('dataset.csv')

# Drop the unnamed index column Kaggle sometimes adds
if 'Unnamed: 0' in df.columns:
    df = df.drop(columns=['Unnamed: 0'])

print(f"\n📦 Dataset shape: {df.shape}")
print(f"   {df.shape[0]:,} tracks  ×  {df.shape[1]} columns\n")

print("── Column names ──")
print(df.columns.tolist())

print("\n── First 3 rows ──")
print(df.head(3))

print("\n── Data types & missing values ──")
print(df.info())

print("\n── Missing values per column ──")
print(df.isnull().sum())

print(f"\n── Total unique genres: {df['track_genre'].nunique()} ──")
print("\nAll genres in the dataset:")
print(sorted(df['track_genre'].unique()))


# =============================================================================
# SECTION 3: IDENTIFY AFRICAN & COMPARISON GENRES
# =============================================================================

# ── Step 3a: Search for African/African-origin genres ─────────────────────────
all_genres = df['track_genre'].unique()

african_keywords = [
    'afro', 'african', 'highlife', 'amapiano',
    'bongo', 'gengetone', 'afrobeat', 'afrobeats',
    'kizomba', 'azonto', 'afropop'
]

detected_african = sorted([
    g for g in all_genres
    if any(k in g.lower() for k in african_keywords)
])

print("\n🌍 African/African-origin genres detected in dataset:")
for g in detected_african:
    count = df[df['track_genre'] == g].shape[0]
    print(f"   {g:<30} {count:>5} tracks")

# ── Step 3b: African diaspora / rhythmically related genres ───────────────────
diaspora_genres = ['dancehall', 'reggae', 'reggaeton', 'latin', 'samba']
detected_diaspora = [g for g in diaspora_genres if g in all_genres]

print("\n🌐 Diaspora/rhythmically related genres found:")
print(detected_diaspora)

# ── Step 3c: Global comparison genres (anchors for the map) ───────────────────
comparison_genres = ['pop', 'hip-hop', 'r-n-b', 'electronic', 'classical', 'rock', 'jazz']
detected_comparison = [g for g in comparison_genres if g in all_genres]

print("\n🎵 Global comparison genres (anchors):")
print(detected_comparison)

# ── Step 3d: Build the working dataset ────────────────────────────────────────
# NOTE: If detected_african is empty or very small, check the genre list printed
# above and manually add genre names from the list that look African.
# For example: AFRICAN_GENRES = ['afrobeat', 'afropop', 'highlife']

AFRICAN_GENRES  = detected_african          # Update this list if needed
DIASPORA_GENRES = detected_diaspora
COMPARISON_GENRES = detected_comparison

ALL_SELECTED = AFRICAN_GENRES + DIASPORA_GENRES + COMPARISON_GENRES

df_work = df[df['track_genre'].isin(ALL_SELECTED)].copy()
df_work = df_work.drop_duplicates(subset=['track_id'])   # remove duplicate tracks
df_work = df_work.dropna()                               # drop any rows with NaN

print(f"\n✅ Working dataset: {df_work.shape[0]:,} tracks across {df_work['track_genre'].nunique()} genres")
print(df_work['track_genre'].value_counts())


# =============================================================================
# SECTION 4: AUDIO FEATURES — WHAT WE'RE MEASURING
# =============================================================================

# These are the 10 Spotify audio features we'll analyse
AUDIO_FEATURES = [
    'danceability',    # 0–1  : how suitable for dancing
    'energy',          # 0–1  : intensity & activity
    'loudness',        # dB   : overall loudness (negative values — will scale)
    'speechiness',     # 0–1  : presence of spoken words
    'acousticness',    # 0–1  : confidence that track is acoustic
    'instrumentalness',# 0–1  : predicts whether track has no vocals
    'liveness',        # 0–1  : presence of live audience
    'valence',         # 0–1  : musical positivity / happiness
    'tempo',           # BPM  : estimated tempo
    'popularity',      # 0–100: Spotify popularity score
]

# ── Distribution plots — understand the raw features ──────────────────────────
fig, axes = plt.subplots(2, 5, figsize=(18, 8))
axes = axes.flatten()

for i, feat in enumerate(AUDIO_FEATURES):
    axes[i].hist(df_work[feat], bins=40, color='#9B6EBF', edgecolor='white', linewidth=0.3)
    axes[i].set_title(feat, fontsize=11, fontweight='bold')
    axes[i].set_xlabel('')

plt.suptitle('Distribution of Audio Features in Selected Dataset',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('01_feature_distributions.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: 01_feature_distributions.png")

# ── Correlation heatmap — find redundant features (PCA will handle these) ─────
corr = df_work[AUDIO_FEATURES].corr()

plt.figure(figsize=(11, 9))
mask = np.triu(np.ones_like(corr, dtype=bool))
sns.heatmap(corr, mask=mask, annot=True, fmt='.2f', cmap='RdYlGn',
            center=0, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('Feature Correlation Matrix\n(High correlations = PCA will compress these)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('02_correlation_heatmap.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: 02_correlation_heatmap.png")

# ── Average audio profile per genre — first look at differences ───────────────
genre_profile = df_work.groupby('track_genre')[AUDIO_FEATURES].mean()

# Focus on African genres vs a few comparison genres for the radar feel
focus_genres = AFRICAN_GENRES + ['pop', 'hip-hop', 'classical'][:min(3, len(COMPARISON_GENRES))]
focus_genres = [g for g in focus_genres if g in genre_profile.index]

genre_profile_focus = genre_profile.loc[focus_genres, [
    'danceability', 'energy', 'valence', 'acousticness', 'speechiness', 'tempo'
]]

# Normalise tempo to 0-1 for fair visual comparison
genre_profile_focus['tempo'] = (
    (genre_profile_focus['tempo'] - genre_profile_focus['tempo'].min()) /
    (genre_profile_focus['tempo'].max() - genre_profile_focus['tempo'].min())
)

plt.figure(figsize=(13, 6))
genre_profile_focus.T.plot(kind='bar', ax=plt.gca(), width=0.8)
plt.title('Average Audio Profile: African vs Global Genres\n(tempo normalised to 0–1)',
          fontsize=13, fontweight='bold')
plt.ylabel('Feature Value (0 – 1)')
plt.xlabel('Audio Feature')
plt.xticks(rotation=30, ha='right')
plt.legend(bbox_to_anchor=(1.01, 1), loc='upper left', fontsize=9)
plt.tight_layout()
plt.savefig('03_genre_profiles.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: 03_genre_profiles.png")


# =============================================================================
# SECTION 5: DATA PREPROCESSING
# =============================================================================

X_raw = df_work[AUDIO_FEATURES].values

# StandardScaler: mean=0, variance=1 → required for PCA and K-Means
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_raw)

# MinMaxScaler: shifts everything to 0–1 → required for NMF (no negatives allowed)
minmax = MinMaxScaler()
X_minmax = minmax.fit_transform(X_raw)

print(f"✅ X_scaled shape  : {X_scaled.shape}  (for PCA, K-Means, Hierarchical, t-SNE)")
print(f"✅ X_minmax shape  : {X_minmax.shape}  (for NMF)")


# =============================================================================
# SECTION 6: PCA — "What Actually Drives Musical Variation?"
# =============================================================================
# Question: Of all 10 audio features, which underlying dimensions
# actually explain the differences between songs? How many truly
# independent axes of musical variation exist in African music?

print("\n" + "="*60)
print("SECTION 6: PCA ANALYSIS")
print("="*60)

# ── 6a: Fit PCA on all components first (for scree plot) ──────────────────────
pca_full = PCA()
pca_full.fit(X_scaled)

explained = pca_full.explained_variance_ratio_
cumulative = np.cumsum(explained)

# ── 6b: Scree plot + Cumulative variance ──────────────────────────────────────
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

# Scree plot
ax1.bar(range(1, len(explained)+1), explained * 100,
        color='#1D9E75', edgecolor='white', linewidth=0.5)
ax1.set_xlabel('Principal Component', fontsize=11)
ax1.set_ylabel('Variance Explained (%)', fontsize=11)
ax1.set_title('Scree Plot\n(How much each PC explains)', fontsize=12, fontweight='bold')
ax1.set_xticks(range(1, len(explained)+1))

# Cumulative variance
ax2.plot(range(1, len(cumulative)+1), cumulative * 100,
         'o-', color='#9B6EBF', linewidth=2, markersize=7)
ax2.axhline(y=80, color='#E05A2B', linestyle='--', alpha=0.7, label='80% threshold')
ax2.axhline(y=90, color='#E0A02B', linestyle='--', alpha=0.7, label='90% threshold')
ax2.set_xlabel('Number of Principal Components', fontsize=11)
ax2.set_ylabel('Cumulative Variance Explained (%)', fontsize=11)
ax2.set_title('Cumulative Explained Variance\n(How many PCs do we need?)',
              fontsize=12, fontweight='bold')
ax2.set_xticks(range(1, len(cumulative)+1))
ax2.legend()

plt.suptitle('PCA: Finding the True Dimensions of Musical Variation',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('04_pca_scree.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: 04_pca_scree.png")

# Print explained variance numbers
print("\n── Variance explained per component ──")
for i, (e, c) in enumerate(zip(explained, cumulative), 1):
    print(f"  PC{i}: {e*100:5.1f}%   cumulative: {c*100:5.1f}%")

# ── 6c: Choose number of components ───────────────────────────────────────────
# Rule: pick components that together explain ~80% variance
N_COMPONENTS = int(np.argmax(cumulative >= 0.80)) + 1
print(f"\n✅ Chosen N_COMPONENTS = {N_COMPONENTS}  (explains {cumulative[N_COMPONENTS-1]*100:.1f}% of variance)")

# ── 6d: Fit final PCA with chosen components ───────────────────────────────────
pca = PCA(n_components=N_COMPONENTS, random_state=42)
X_pca = pca.fit_transform(X_scaled)

print(f"   X_pca shape: {X_pca.shape}")

# ── 6e: Component loadings heatmap ────────────────────────────────────────────
# This shows WHICH features drive each principal component
# This is where you name the components (e.g. "Intensity axis", "Mood axis")

loadings = pd.DataFrame(
    pca.components_.T,
    index=AUDIO_FEATURES,
    columns=[f'PC{i+1}' for i in range(N_COMPONENTS)]
)

plt.figure(figsize=(max(8, N_COMPONENTS * 1.2), 6))
sns.heatmap(loadings, annot=True, fmt='.2f', cmap='RdBu_r',
            center=0, linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('PCA Component Loadings\n(Which features drive each component?)',
          fontsize=13, fontweight='bold')
plt.xlabel('Principal Component')
plt.ylabel('Audio Feature')
plt.tight_layout()
plt.savefig('05_pca_loadings.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: 05_pca_loadings.png")

print("\n── Interpret each PC by its dominant features ──")
for i, col in enumerate(loadings.columns):
    dominant = loadings[col].abs().nlargest(3)
    signs = loadings[col][dominant.index]
    print(f"\n  {col} (explains {explained[i]*100:.1f}%):")
    for feat, sign in signs.items():
        direction = "↑ high" if sign > 0 else "↓ low"
        print(f"    {direction} {feat}  ({sign:.2f})")

# ── 6f: PC1 vs PC2 scatter coloured by genre ──────────────────────────────────
plt.figure(figsize=(13, 8))

genre_list = df_work['track_genre'].values
unique_genres = sorted(df_work['track_genre'].unique())
palette = sns.color_palette('tab20', len(unique_genres))
color_map = {g: palette[i] for i, g in enumerate(unique_genres)}
colors = [color_map[g] for g in genre_list]

plt.scatter(X_pca[:, 0], X_pca[:, 1],
            c=colors, alpha=0.4, s=15, linewidths=0)

# Draw genre centroids (easier to read than 10k overlapping dots)
for genre in unique_genres:
    mask = genre_list == genre
    cx = X_pca[mask, 0].mean()
    cy = X_pca[mask, 1].mean()
    plt.scatter(cx, cy, s=180, color=color_map[genre],
                edgecolors='white', linewidths=1.5, zorder=5)
    plt.annotate(genre, (cx, cy), fontsize=7.5, fontweight='bold',
                 xytext=(5, 5), textcoords='offset points')

plt.xlabel(f'PC1 ({explained[0]*100:.1f}% variance)', fontsize=11)
plt.ylabel(f'PC2 ({explained[1]*100:.1f}% variance)', fontsize=11)
plt.title('PCA Map: Where Each Genre Sits in Musical Space\n(centroids labelled)',
          fontsize=13, fontweight='bold')
plt.tight_layout()
plt.savefig('06_pca_scatter.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: 06_pca_scatter.png")


# =============================================================================
# SECTION 7: K-MEANS CLUSTERING — "How Many Distinct Sound Groups Exist?"
# =============================================================================
# Question: If you forget genre labels entirely and just look at audio features,
# how many genuinely distinct types of music emerge?
# We cluster on PCA-reduced data (cleaner, less noise than raw features)

print("\n" + "="*60)
print("SECTION 7: K-MEANS CLUSTERING")
print("="*60)

# ── 7a: Elbow plot — find optimal k ───────────────────────────────────────────
inertias     = []
silhouettes  = []
K_RANGE      = range(2, 12)

print("Testing k values...")
for k in K_RANGE:
    km = KMeans(n_clusters=k, random_state=42, n_init=10)
    km.fit(X_pca)
    inertias.append(km.inertia_)
    sil = silhouette_score(X_pca, km.labels_, sample_size=5000, random_state=42)
    silhouettes.append(sil)
    print(f"  k={k:2d}  inertia={km.inertia_:,.0f}  silhouette={sil:.3f}")

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

ax1.plot(K_RANGE, inertias, 'o-', color='#1D9E75', linewidth=2, markersize=8)
ax1.set_xlabel('Number of Clusters (k)', fontsize=11)
ax1.set_ylabel('Inertia', fontsize=11)
ax1.set_title('Elbow Plot\n(Look for the "elbow" — where curve flattens)',
              fontsize=12, fontweight='bold')
ax1.set_xticks(list(K_RANGE))

ax2.plot(K_RANGE, silhouettes, 'o-', color='#9B6EBF', linewidth=2, markersize=8)
ax2.set_xlabel('Number of Clusters (k)', fontsize=11)
ax2.set_ylabel('Silhouette Score (higher = better)', fontsize=11)
ax2.set_title('Silhouette Scores\n(Pick the k with the highest score)',
              fontsize=12, fontweight='bold')
ax2.set_xticks(list(K_RANGE))

plt.suptitle('Finding the Optimal Number of Music Clusters',
             fontsize=14, fontweight='bold', y=1.02)
plt.tight_layout()
plt.savefig('07_kmeans_elbow.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: 07_kmeans_elbow.png")

# ── 7b: Pick k and fit final model ────────────────────────────────────────────
# Pick the k where silhouette is highest (or use the elbow visually)
BEST_K = K_RANGE.start + int(np.argmax(silhouettes))
print(f"\n✅ Best k by silhouette = {BEST_K}")
print("   (You can manually override this by changing BEST_K if the elbow suggests different)")
# BEST_K = 6  ← uncomment and change if you want to override

km_final = KMeans(n_clusters=BEST_K, random_state=42, n_init=10)
km_final.fit(X_pca)
df_work['cluster'] = km_final.labels_

# ── 7c: Audio profile of each cluster — NAME your clusters ────────────────────
cluster_profiles = df_work.groupby('cluster')[AUDIO_FEATURES].mean()
print("\n── Average audio profile per cluster ──")
print(cluster_profiles.round(3).to_string())

print("\n── 🎯 NAME YOUR CLUSTERS by reading the table above ──")
print("   High danceability + high energy            → something like 'Dance/Club'")
print("   High acousticness + low energy             → something like 'Acoustic/Calm'")
print("   High speechiness                            → something like 'Rap/Spoken'")
print("   High instrumentalness + low speechiness    → something like 'Instrumental'")
print("   High valence + high danceability            → something like 'Feel-Good'")
print("   Update CLUSTER_NAMES below after reading the table!\n")

# ← UPDATE THIS DICTIONARY after reading cluster_profiles above
CLUSTER_NAMES = {i: f'Sound Group {i+1}' for i in range(BEST_K)}
# Example after reading profiles:
# CLUSTER_NAMES = {
#     0: 'High-Energy Dance',
#     1: 'Acoustic & Calm',
#     2: 'Rap & Spoken',
#     3: 'Instrumental',
#     4: 'Feel-Good Rhythmic',
#     5: 'Melancholic',
# }

df_work['cluster_name'] = df_work['cluster'].map(CLUSTER_NAMES)

# ── 7d: How do African genres distribute across clusters? ─────────────────────
african_cluster = df_work[df_work['track_genre'].isin(AFRICAN_GENRES)]
ct = pd.crosstab(african_cluster['track_genre'],
                 african_cluster['cluster_name'],
                 normalize='index') * 100

print("\n── African genre distribution across clusters (%) ──")
print(ct.round(1).to_string())

plt.figure(figsize=(12, 5))
ct.plot(kind='bar', ax=plt.gca(), width=0.75, colormap='Set2')
plt.title('How African Genres Are Distributed Across Sound Clusters',
          fontsize=13, fontweight='bold')
plt.xlabel('African Genre')
plt.ylabel('% of Tracks in Each Cluster')
plt.xticks(rotation=30, ha='right')
plt.legend(title='Sound Group', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('08_african_cluster_distribution.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: 08_african_cluster_distribution.png")


# =============================================================================
# SECTION 8: HIERARCHICAL CLUSTERING — "How Do African Genres Relate?"
# =============================================================================
# Question: Is Afrobeats closer to Dancehall or to Hip-Hop?
# Which African genres are most sonically similar to each other?
# The dendrogram is a family tree of musical DNA.

print("\n" + "="*60)
print("SECTION 8: HIERARCHICAL CLUSTERING")
print("="*60)

# We cluster on GENRE CENTROIDS (average audio profile per genre)
# This is fast, readable, and tells us genre-level relationships
genre_centroids = df_work.groupby('track_genre')[AUDIO_FEATURES].mean()
genre_centroids_scaled = scaler.transform(genre_centroids)

print(f"Clustering {len(genre_centroids)} genre centroids...")
print("Genres:", genre_centroids.index.tolist())

# ── 8a: Run linkage (complete = most conservative merging strategy) ────────────
mergings = linkage(genre_centroids_scaled, method='complete')

# ── 8b: Dendrogram ────────────────────────────────────────────────────────────
# Colour African genres differently from comparison genres
def colour_label(label):
    if label in AFRICAN_GENRES:
        return '#1D9E75'       # green for African
    elif label in DIASPORA_GENRES:
        return '#E05A2B'       # orange for diaspora
    else:
        return '#555555'       # grey for global comparison

fig, ax = plt.subplots(figsize=(15, 8))
dend = dendrogram(
    mergings,
    labels=genre_centroids.index.tolist(),
    leaf_rotation=45,
    leaf_font_size=11,
    color_threshold=0.7 * max(mergings[:, 2]),
    ax=ax
)
ax.set_title(
    'Musical Family Tree: How African Genres Relate to the World\n'
    'Height = sonic distance between genres (lower merge = more similar)',
    fontsize=13, fontweight='bold'
)
ax.set_ylabel('Distance (sonic dissimilarity)', fontsize=11)
ax.set_xlabel('Genre', fontsize=11)

# Colour the x-axis tick labels by category
for lbl in ax.get_xticklabels():
    genre_name = lbl.get_text()
    lbl.set_color(colour_label(genre_name))
    if genre_name in AFRICAN_GENRES:
        lbl.set_fontweight('bold')

# Add legend
patches = [
    mpatches.Patch(color='#1D9E75', label='African genres'),
    mpatches.Patch(color='#E05A2B', label='African diaspora genres'),
    mpatches.Patch(color='#555555', label='Global comparison genres'),
]
ax.legend(handles=patches, loc='upper right', fontsize=10)

plt.tight_layout()
plt.savefig('09_dendrogram.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: 09_dendrogram.png")

# ── 8c: Extract flat cluster labels from the hierarchy ────────────────────────
# Cut the tree to get a readable number of groups
n_hier_clusters = min(5, len(genre_centroids) // 2)
hier_labels = fcluster(mergings, n_hier_clusters, criterion='maxclust')

hier_df = pd.DataFrame({
    'genre': genre_centroids.index,
    'hierarchical_group': hier_labels
}).sort_values('hierarchical_group')

print("\n── Hierarchical groups ──")
for grp in sorted(hier_df['hierarchical_group'].unique()):
    members = hier_df[hier_df['hierarchical_group'] == grp]['genre'].tolist()
    print(f"  Group {grp}: {members}")

print("\n🔍 Key finding to look for:")
print("   Which African genre merges earliest (lowest height) with a NON-African genre?")
print("   That's your strongest cross-continental sonic connection — the story of your article.")


# =============================================================================
# SECTION 9: t-SNE — "The Global Sonic Map"
# =============================================================================
# Question: If you placed every song on a 2D map where similar-sounding
# songs sit close together, where would African music end up?
# This is the HERO VISUAL of your Medium article.

print("\n" + "="*60)
print("SECTION 9: t-SNE VISUALISATION")
print("="*60)

# Sample if dataset is very large (t-SNE is slow above ~30k rows)
MAX_TSNE = 20000
if len(X_pca) > MAX_TSNE:
    idx = np.random.choice(len(X_pca), MAX_TSNE, replace=False)
    X_tsne_input = X_pca[idx]
    genres_tsne  = df_work['track_genre'].values[idx]
    clusters_tsne = df_work['cluster'].values[idx]
    print(f"Sampled {MAX_TSNE:,} tracks for t-SNE (full dataset too large)")
else:
    X_tsne_input  = X_pca
    genres_tsne   = df_work['track_genre'].values
    clusters_tsne = df_work['cluster'].values
    print(f"Running t-SNE on all {len(X_pca):,} tracks")

# ── 9a: Run t-SNE ─────────────────────────────────────────────────────────────
print("Running t-SNE... (this takes 1–3 minutes)")
tsne = TSNE(n_components=2, learning_rate=150, perplexity=40,
            n_iter=1000, random_state=42)
X_embedded = tsne.fit_transform(X_tsne_input)
print("✅ t-SNE complete")

tsne_df = pd.DataFrame({
    'x': X_embedded[:, 0],
    'y': X_embedded[:, 1],
    'genre': genres_tsne,
    'cluster': clusters_tsne
})

# ── 9b: Plot 1 — coloured by K-Means cluster (validation: do clusters separate?) ──
fig, ax = plt.subplots(figsize=(13, 9))
cluster_palette = sns.color_palette('tab10', BEST_K)

for c in range(BEST_K):
    mask = tsne_df['cluster'] == c
    ax.scatter(tsne_df.loc[mask, 'x'], tsne_df.loc[mask, 'y'],
               s=8, alpha=0.35, color=cluster_palette[c],
               label=CLUSTER_NAMES.get(c, f'Cluster {c}'))

ax.set_title('Global Sonic Map: All Songs Coloured by Sound Cluster\n'
             '(Points close together = sonically similar)',
             fontsize=13, fontweight='bold')
ax.set_xlabel('t-SNE Dimension 1')
ax.set_ylabel('t-SNE Dimension 2')
ax.legend(title='Sound Group', bbox_to_anchor=(1.01, 1),
          loc='upper left', markerscale=2.5)
ax.axis('off')  # t-SNE axes are not meaningful — hide them
plt.tight_layout()
plt.savefig('10_tsne_clusters.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: 10_tsne_clusters.png")

# ── 9c: Plot 2 — HERO VISUAL: highlight African genres on the global map ───────
fig, ax = plt.subplots(figsize=(14, 10))

# All non-African genres as grey background
non_african_mask = ~tsne_df['genre'].isin(AFRICAN_GENRES)
ax.scatter(tsne_df.loc[non_african_mask, 'x'],
           tsne_df.loc[non_african_mask, 'y'],
           s=6, alpha=0.12, color='#CCCCCC', zorder=1)

# African genres in vivid colours on top
african_palette = sns.color_palette('Set1', len(AFRICAN_GENRES))
for i, genre in enumerate(AFRICAN_GENRES):
    mask = tsne_df['genre'] == genre
    if mask.sum() == 0:
        continue
    ax.scatter(tsne_df.loc[mask, 'x'], tsne_df.loc[mask, 'y'],
               s=25, alpha=0.75, color=african_palette[i],
               label=genre, zorder=3, linewidths=0)
    # Mark the centroid
    cx = tsne_df.loc[mask, 'x'].mean()
    cy = tsne_df.loc[mask, 'y'].mean()
    ax.scatter(cx, cy, s=250, color=african_palette[i],
               edgecolors='white', linewidths=2, zorder=5)
    ax.annotate(genre.upper(), (cx, cy), fontsize=9, fontweight='bold',
                color=african_palette[i],
                xytext=(8, 8), textcoords='offset points',
                bbox=dict(boxstyle='round,pad=0.2', facecolor='white',
                          edgecolor=african_palette[i], alpha=0.85))

# Also label global comparison genre centroids for context
for genre in COMPARISON_GENRES:
    mask = tsne_df['genre'] == genre
    if mask.sum() == 0:
        continue
    cx = tsne_df.loc[mask, 'x'].mean()
    cy = tsne_df.loc[mask, 'y'].mean()
    ax.annotate(genre, (cx, cy), fontsize=7.5, color='#777777',
                xytext=(5, 5), textcoords='offset points', style='italic')

ax.set_title(
    '🌍  Where Does African Music Sit in the Global Sonic Landscape?\n'
    'Grey = all global genres  |  Coloured = African genres  |  Each point = one song',
    fontsize=13, fontweight='bold'
)
ax.legend(title='African Genre', bbox_to_anchor=(1.01, 1),
          loc='upper left', markerscale=2)
ax.axis('off')
plt.tight_layout()
plt.savefig('11_tsne_african_hero.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: 11_tsne_african_hero.png  ← This is your hero visual for Medium")


# =============================================================================
# SECTION 10: NMF — "What Sound Archetypes Is African Music Built From?"
# =============================================================================
# Question: Every song is a blend of underlying sonic "templates."
# What are those templates, and how much of each does African music contain
# compared to the rest of the world?
# NMF uncovers these latent archetypes — interpretable building blocks.

print("\n" + "="*60)
print("SECTION 10: NMF ANALYSIS")
print("="*60)

N_ARCHETYPES = 5   # number of sonic archetypes to discover
               # Try 4–7, pick the one that gives most interpretable components

nmf = NMF(n_components=N_ARCHETYPES, random_state=42, max_iter=500)
W = nmf.fit_transform(X_minmax)    # W: how much of each archetype each song has
H = nmf.components_                # H: how each archetype is built from features

print(f"✅ NMF fitted with {N_ARCHETYPES} archetypes")
print(f"   W shape (song × archetype): {W.shape}")
print(f"   H shape (archetype × feature): {H.shape}")
print(f"   Reconstruction error: {nmf.reconstruction_err_:.2f}")

# ── 10a: Archetype heatmap — what is each archetype? ─────────────────────────
archetype_df = pd.DataFrame(
    H, columns=AUDIO_FEATURES,
    index=[f'Archetype {i+1}' for i in range(N_ARCHETYPES)]
)

# Normalise each archetype row to 0-1 for fair visual comparison
archetype_norm = archetype_df.div(archetype_df.max(axis=1), axis=0)

plt.figure(figsize=(13, 5))
sns.heatmap(archetype_norm, annot=True, fmt='.2f', cmap='YlOrRd',
            linewidths=0.5, cbar_kws={'shrink': 0.8})
plt.title('NMF Sound Archetypes\n(Brighter = stronger feature in that archetype)',
          fontsize=13, fontweight='bold')
plt.ylabel('Archetype')
plt.xlabel('Audio Feature')
plt.tight_layout()
plt.savefig('12_nmf_archetypes.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: 12_nmf_archetypes.png")

# ── 10b: Name the archetypes (do this after reading the heatmap) ──────────────
print("\n── 🎯 NAME YOUR ARCHETYPES by reading the heatmap ──")
print("   High danceability + high valence + high energy → e.g. 'Euphoric Dance'")
print("   High acousticness + low energy                 → e.g. 'Acoustic Soul'")
print("   High speechiness + high energy                 → e.g. 'Rap Energy'")
print("   High instrumentalness                          → e.g. 'Pure Instrumental'")
print("   High liveness + moderate energy                → e.g. 'Live Performance'")

# ← UPDATE these after reading the heatmap
ARCHETYPE_NAMES = {i: f'Archetype {i+1}' for i in range(N_ARCHETYPES)}
# Example:
# ARCHETYPE_NAMES = {
#     0: 'Euphoric Dance',
#     1: 'Acoustic Soul',
#     2: 'Rap Energy',
#     3: 'Pure Instrumental',
#     4: 'Live Performance',
# }

# ── 10c: Archetype blend per genre (the key finding) ─────────────────────────
df_work['dominant_archetype'] = W.argmax(axis=1)
W_df = pd.DataFrame(W, columns=[ARCHETYPE_NAMES[i] for i in range(N_ARCHETYPES)],
                    index=df_work.index)
df_work = pd.concat([df_work, W_df], axis=1)

archetype_cols = [ARCHETYPE_NAMES[i] for i in range(N_ARCHETYPES)]
genre_archetype = df_work.groupby('track_genre')[archetype_cols].mean()

# Normalise rows to show BLEND proportions
genre_archetype_norm = genre_archetype.div(genre_archetype.sum(axis=1), axis=0) * 100

print("\n── Archetype blend by genre (%) ──")
print(genre_archetype_norm.round(1).to_string())

# ── 10d: Stacked bar — African genres side by side with global ────────────────
focus = AFRICAN_GENRES + [g for g in COMPARISON_GENRES if g in genre_archetype_norm.index]
ga_focus = genre_archetype_norm.loc[[g for g in focus if g in genre_archetype_norm.index]]

ax = ga_focus.plot(kind='bar', stacked=True, figsize=(14, 6),
                   colormap='Set2', width=0.75, edgecolor='white', linewidth=0.5)
plt.title('Sound Archetype Blend: African Genres vs Global Genres\n'
          '(Each bar = 100%; colours show how much of each archetype)',
          fontsize=13, fontweight='bold')
plt.xlabel('Genre')
plt.ylabel('Archetype Blend (%)')
plt.xticks(rotation=35, ha='right')
plt.legend(title='Sound Archetype', bbox_to_anchor=(1.01, 1), loc='upper left')
plt.tight_layout()
plt.savefig('13_nmf_archetype_blend.png', dpi=150, bbox_inches='tight')
plt.show()
print("📊 Saved: 13_nmf_archetype_blend.png")

# ── 10e: Cross-continental connection — which global genre shares most ─────────
#         archetype DNA with African genres?
print("\n── Cross-continental archetype similarity ──")
print("   (Cosine similarity between African genre archetype vectors and global genres)\n")

from numpy.linalg import norm

def cosine_sim(a, b):
    return np.dot(a, b) / (norm(a) * norm(b))

for african_g in AFRICAN_GENRES:
    if african_g not in genre_archetype.index:
        continue
    vec_a = genre_archetype.loc[african_g].values
    sims  = {}
    for comp_g in COMPARISON_GENRES + DIASPORA_GENRES:
        if comp_g not in genre_archetype.index:
            continue
        vec_c = genre_archetype.loc[comp_g].values
        sims[comp_g] = cosine_sim(vec_a, vec_c)
    top3 = sorted(sims.items(), key=lambda x: x[1], reverse=True)[:3]
    print(f"  {african_g:<25} most similar to: {top3}")


# =============================================================================
# SECTION 11: SYNTHESIS — THE FINDINGS
# =============================================================================

print("\n" + "="*60)
print("SECTION 11: KEY FINDINGS SUMMARY")
print("="*60)

print("""
Your Medium article should answer these questions with the evidence you found:

1. PCA FINDING:
   → How many true dimensions of musical variation exist?
   → What are those dimensions? (Name them from the loadings heatmap)
   → Does African music occupy a unique region in PCA space?

2. K-MEANS FINDING:
   → How many distinct sound clusters did the algorithm discover?
   → Which clusters do African genres primarily fall into?
   → Do different African genres fall into DIFFERENT clusters
     (showing internal diversity) or the SAME cluster (showing cohesion)?

3. HIERARCHICAL FINDING:
   → Which African genres are most sonically similar to each other?
     (merged earliest / lowest height on dendrogram)
   → Which non-African genre does Afrobeats merge with first?
     That's your cross-continental sonic connection story.
   → Are African genres clustered together or spread across the tree?

4. t-SNE FINDING:
   → On the global sonic map, are African genres in one corner
     or spread across different regions?
   → Which global genre does the African music cloud sit nearest to?
   → Do different African genres (e.g. afrobeat vs highlife) sit far
     apart on the map, showing internal diversity?

5. NMF FINDING:
   → What are the 5 underlying sound archetypes in the dataset?
   → Which archetype dominates African music?
   → Is the archetype blend of Afrobeats more similar to Latin music
     or to Hip-Hop? That's the cross-cultural DNA story.
""")

print("="*60)
print("FILES SAVED:")
saved = [
    '01_feature_distributions.png',
    '02_correlation_heatmap.png',
    '03_genre_profiles.png',
    '04_pca_scree.png',
    '05_pca_loadings.png',
    '06_pca_scatter.png',
    '07_kmeans_elbow.png',
    '08_african_cluster_distribution.png',
    '09_dendrogram.png',
    '10_tsne_clusters.png',
    '11_tsne_african_hero.png  ← HERO VISUAL',
    '12_nmf_archetypes.png',
    '13_nmf_archetype_blend.png',
]
for f in saved:
    print(f"   📊 {f}")
print("="*60)
print("\n✅ Project complete. Now read every output and write your findings.")
print("   The code does the maths. YOU provide the interpretation.")
