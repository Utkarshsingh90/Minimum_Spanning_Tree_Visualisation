# 🌳 Spanning Trees Explorer

**Interactive Streamlit App for Visualizing Spanning Tree Algorithms**

This project is a **Streamlit web application** designed for exploring
and visualizing graph spanning tree algorithms.\
It includes interactive demos of **Wilson's Algorithm**, **Prüfer
sequences**, **Matrix-Tree Theorem**, and a **brute-force spanning tree
enumerator**.

## 🚀 Features

### 1. Interactive Graph Input

-   Choose from preset graphs: Complete (K_n), Path (P_n), Cycle (C_n),
    Grid graphs, and random G(n,p).

-   Enter your own graph using an edge list such as:

        1 2
        a-b
        1,3

### 2. Matrix-Tree Theorem

Compute exact spanning tree count τ(G) using: - Laplacian L = D - A\
- Laplacian minor L'\
- Bareiss determinant for exact integer output

### 3. Wilson's Algorithm Visualizer

-   Step-by-step random walk\
-   Loop-erased paths\
-   Building the uniform spanning tree\
-   Heatmap of edge frequency from multiple samples

### 4. Prüfer Sequence Visualizer

Interactive: - Tree → Prüfer encode\
- Prüfer → Tree decode

### 5. Brute-Force Enumeration

For small graphs, enumerates and visualizes *all* spanning trees.

## 🛠 Installation

``` bash
pip install streamlit networkx numpy matplotlib
```

or

``` bash
pip install -r requirements.txt
```

## ▶ Run

``` bash
streamlit run app.py
```

## 🧭 App Usage

### Configure Graph

Choose preset or custom graph.

### Exact Count

Compute τ(G) using Matrix-Tree Theorem.

### Visualizers

Step-by-step Wilson / Prüfer sliders.

### Enumeration

List all spanning trees (for small graphs).

