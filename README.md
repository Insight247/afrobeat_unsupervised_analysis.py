# 🌍 Where Does Afrobeats Sit in the Global Music Landscape?
### An Unsupervised Machine Learning Deep Dive into African Music

<img width="1290" height="889" alt="image" src="https://github.com/user-attachments/assets/4afef083-7445-4d1a-8e22-d0bd4e7d6cd7" />

*(Above: A t-SNE projection mapping 114,000 global tracks. The dense cluster of colored points highlights the distinct sonic island of African music against the global gray backdrop.)*

---

## 📌 Project Overview
Afrobeats is the biggest music export story of the 21st century. But where does it actually sit in the global sonic landscape — and how does it relate to other African genres, to Latin music, to Hip-Hop?

This project answers those questions using unsupervised machine learning — no genre labels, no human-assigned categories. Five complementary ML techniques work together to build a data-driven map of global music, with African genres at the center of the story.

> *"The algorithm never saw a single genre label. It found the structure on its own."*

## ❓ Problem Statement
Where does Afrobeats actually sit in the global sonic landscape — and how does it relate to other African genres, to Latin music, to Hip-Hop when measured strictly by mathematical audio features?

## 🎯 Problem Specification
To fully answer the statement above, this project executes the following objectives:
1. Isolate African genres (e.g., Afrobeat) and contrast them against Global Anchors (Pop, Hip-Hop, Classical) and Diaspora Cousins (Dancehall, Reggaeton, Latin).
2. Determine if African genres are internally diverse or if they share a single monolithic sonic identity.
3. Identify which global genre Afrobeats is most similar to using purely mathematical distances.
4. Extract the underlying, latent "sound archetypes" that make up the African music recipe.

---

## 🗂️ Dataset

| Property | Detail | Source |
| :--- | :--- | :--- |
| **Name** | Spotify Tracks Dataset | Kaggle |
| **Author** | Maharshi Pandya | |
| **Size** | 114,000 tracks × 20 columns | |
| **Genres** | 114 unique genres | |
| **License** | Public / Kaggle Community | |

### Audio Features Used

| Feature | Range | What It Measures |
| :--- | :--- | :--- |
| **danceability** | 0–1 | How suitable for dancing |
| **energy** | 0–1 | Intensity and activity |
| **loudness** | dB (-) | Overall loudness |
| **speechiness** | 0–1 | Presence of spoken words |
| **acousticness** | 0–1 | Confidence the track is acoustic |
| **instrumentalness** | 0–1 | Likelihood of no vocals |
| **liveness** | 0–1 | Presence of live audience |
| **valence** | 0–1 | Musical positivity / happiness |
| **tempo** | BPM | Estimated beats per minute |
| **popularity** | 0–100 | Spotify popularity score |

*Note: `key`, `mode`, and `time_signature` were deliberately excluded — they are categorical musical properties, not continuous measurements, and would produce misleading results in distance-based algorithms.*

---

## 🧠 Methods & What Each One Answers

| Method | Question Answered | Output |
| :--- | :--- | :--- |
| **PCA** | What truly drives musical variation? How many independent dimensions exist? | Variance plot, component loadings, PCA scatter |
| **K-Means Clustering** | How many distinct sound groups emerge? Which cluster does African music fall into? | Elbow plots, cluster profiles, distribution bars |
| **Hierarchical Clustering** | Which genres are most sonically similar? Which non-African genre does Afrobeats merge with first? | Dendrogram (musical family tree) |
| **t-SNE** | Where does African music sit on a 2D global sonic map? | Hero visualization — tracks mapped |
| **NMF** | What underlying sound archetypes is African music built from? | Archetype heatmap, cross-continental similarity |

---

## 📁 Project Structure

```text
spotify-african-music-unsupervised/
│
├── afrobeat_unsupervised_analysis.py   # Main analysis script (all 5 techniques)
├── README.md                           # This file
│
├── outputs/                            # Generated plots 
│   ├── 01_feature_distributions.png
│   ├── 02_correlation_heatmap.png
│   ├── 03_genre_profiles.png
│   ├── 04_pca_scree.png
│   ├── 05_pca_loadings.png
│   ├── 06_pca_scatter.png
│   ├── 07_kmeans_elbow.png
│   ├── 08_african_cluster_distribution.png
│   ├── 09_dendrogram.png
│   ├── 10_tsne_clusters.png
│   ├── 11_tsne_african_hero.png        ← Hero visual
│   ├── 12_nmf_archetypes.png
│   └── 13_nmf_archetype_blend.png
│
└── dataset.csv                         # Not included — download from Kaggle
