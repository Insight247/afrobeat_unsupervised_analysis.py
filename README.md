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

**Key Findings**

**PCA (Principal Component Analysis)**
The 10 audio features successfully reduced to 6 principal components that explain 83.6% of the musical variance.
PC1 (34.0% variance) acts as the Energy Axis, driven heavily by high loudness and energy versus acousticness.
PC2 (11.4% variance) acts as the Live/Raw Axis, separating fast, live-recorded tracks from highly danceable studio tracks.
African genres occupy the high-energy, high-rhythm space on the scatter plot, completely isolating themselves from acoustic outliers like Classical and Jazz.

**K-Means Clustering**
The algorithm identified 6 distinct sound groups across the global dataset.
Afrobeats is not monolithic; it displays internal diversity. However, it predominantly falls into the "Upbeat & Positive" cluster (59.7% of tracks), characterized by immense valence and danceability.
A notable 17.6% of Afrobeats tracks fall into the "Mellow / Soft Acoustic" cluster, highlighting the genre's contemporary shift toward mid-tempo, relaxed rhythms.

**Hierarchical Clustering**
The dendrogram reveals the "transatlantic sonic highway." When grouped by average audio profiles, Afrobeats merges earliest with Latin music. * This mathematically proves that despite geographic distance, Afrobeat and Latin music share a nearly identical foundational DNA of high danceability, bright energy, and rhythmic syncopation.
These genres, alongside Dancehall, Reggaeton, and Hip-Hop, form a distinct "Diaspora Super-Cluster."

**t-SNE**
Projecting individual tracks onto a 2D global sonic map reveals that Afrobeats does not scatter randomly.
It forms a densely packed island, proving it has a highly unified, recognizable sonic signature, situated right on the border of its Caribbean and Latin rhythmic cousins.

**NMF (Non-Negative Matrix Factorization)**
NMF decomposed the dataset into 5 underlying sound archetypes: Acoustic & Unplugged, Euphoric Dance & Energy, Rap & Spoken Rhythm, Pure Instrumental, and Live Performance.
African music is predominantly built from the Euphoric Dance & Energy template.

**Tools & Libraries**
|Library | Purpose ||
| :--- | :--- |
|pandas | Data loading, manipulation, crosstabs ||
|numpy | Numerical operations ||
|matplotlib | Base plotting ||
|seaborn | Statistical visualization ||
|sscikit-learn | PCA, KMeans, NMF, t-SNE, StandardScaler ||
|scipy | Hierarchical clustering (linkage, dendrogram) ||

**Skills Demonstrated**
Unsupervised machine learning (no labels required)
Dimensionality reduction (PCA, NMF)
Clustering (K-Means, Hierarchical / Agglomerative)
High-dimensional data visualization (t-SNE)
Feature engineering and preprocessing (StandardScaler, MinMaxScaler)
Silhouette scoring and elbow method for cluster evaluation
Cross-technique synthesis and result interpretation
Data storytelling and technical writing

Full Write-Up
Read the complete article detailing the methodology and cultural implications on Medium:
Where Does Afrobeats Sit in the Global Music Landscape? — Abdulmalik on Medium (Update link after publishing)

**Author**
Akinwumi Abdulmalik

GitHub: @YOUR_USERNAME
Medium: @YOUR_MEDIUM
LinkedIn: Your LinkedIn

**License**
This project is licensed under the MIT License. The dataset is publicly available on Kaggle under its own terms.

**Acknowledgements**

Dataset: Maharshi Pandya via Kaggle

Audio feature definitions: Spotify Web API Documentation






