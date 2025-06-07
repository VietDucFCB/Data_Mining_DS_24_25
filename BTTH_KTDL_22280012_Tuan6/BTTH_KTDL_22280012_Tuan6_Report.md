# Week 6 Report: K-means and K-medians Clustering

**Student:** NGUYEN XUAN VIET DUC - 22280012
**Lesson:** 6 - GROUP ANALYSIS

## Introduction

This report summarizes the implementation and comparison of two clustering algorithms: K-means and K-medians. Both algorithms were implemented in Python using a Jupyter Notebook environment, leveraging libraries such as NumPy, Pandas, and Matplotlib for data manipulation, computation, and visualization. The goal was to partition a given dataset into a predefined number of clusters (k=3 for detailed analysis) and evaluate the results.

## Algorithms Overview

### 1. K-means Algorithm

**Concept:**
K-means is an iterative clustering algorithm that aims to partition n observations into k clusters in which each observation belongs to the cluster with the nearest mean (cluster centroid).

**Steps:**
1.  **Initialization:** Randomly select k data points from the dataset as initial centroids.
2.  **Assignment:** Assign each data point to the closest centroid. The distance metric typically used is the Euclidean distance, and the error is calculated as the sum of squared errors (SSE) from each point to its assigned centroid.
3.  **Update:** Recalculate the centroids as the mean of all data points assigned to that centroid's cluster.
4.  **Iteration:** Repeat the assignment and update steps until the centroids no longer change significantly (i.e., the change in total error is below a defined tolerance) or a maximum number of iterations is reached.

**Error Metric:** Root Sum of Squared Errors (RSS).
   `rsserr(a, b) = np.sum(np.square(a - b))`

### 2. K-medians Algorithm

**Concept:**
K-medians is similar to K-means but differs in how centroids are updated and how distances (and thus errors) are calculated. It aims to minimize the sum of distances from points to the medians of their clusters. It is generally more robust to outliers than K-means.

**Steps:**
1.  **Initialization:** Randomly select k data points from the dataset as initial centroids (same method as K-means was used in the notebook).
2.  **Assignment:** Assign each data point to the closest centroid. The distance metric used is the Manhattan distance (Sum of Absolute Differences - SAD).
3.  **Update:** Recalculate the centroids as the median of all data points assigned to that centroid's cluster.
4.  **Iteration:** Repeat the assignment and update steps until the centroids no longer change significantly or a maximum number of iterations is reached.

**Error Metric:** Sum of Absolute Differences (SAD).
   `sad_err(a, b) = np.sum(np.abs(a - b))`

## Code Summary

The Jupyter Notebook (`BTTH_KTDL_22280012_Tuan6_code.ipynb`) contains the following key components:

**Setup and Data Loading (Cells 1-4):**
*   Imports necessary libraries: `numpy` (aliased as `npss` in K-means section, corrected to `np` implicitly for K-medians), `pandas`, `matplotlib.pyplot`, and `ListedColormap`.
*   Loads the dataset from `data.csv`.
*   Defines `colnames` for feature selection.

**K-means Implementation (Cells 3-18):**
*   **`initiate_centroids(k, dset)` (Cell 6):** Selects k random samples from the dataset as initial centroids.
*   **`rsserr(a, b)` (Cell 7):** Calculates the Root Sum of Squared Errors between two points.
*   **`centroid_assignation(dset, centroids)` (Cell 8):** Assigns each data point to the nearest centroid based on RSS and calculates the error for each assignment.
*   **Manual Iteration Example (Cells 9-13):** Demonstrates one iteration of centroid assignment and update.
*   **`kmeans(dset, k, tol)` (Cell 14):** The main K-means function that iteratively assigns points and updates centroids until convergence.
*   **Execution and Visualization (Cells 15-17):** Runs the K-means algorithm with k=3, prints the head of the resulting DataFrame with assigned centroids and errors, and plots the clustered data points along with the final centroids.
*   **Elbow Method (Cell 18):** Calculates and plots the total error for different numbers of clusters (1 to 10) to help determine an optimal k value for K-means.

**K-medians Implementation (Cells 19-24):**
*   **`sad_err(a, b)` (Cell 20):** Calculates the Sum of Absolute Differences (Manhattan distance) between two points.
*   **`centroid_assignation_median(dset, centroids)` (Cell 21):** Assigns each data point to the nearest centroid based on SAD and calculates the error.
*   **`kmedians(dset, k, tol)` (Cell 22):** The main K-medians function. It follows the same iterative structure as K-means but uses `sad_err` for error calculation and updates centroids by taking the median of the points in each cluster.
*   **Execution and Visualization (Cells 23-24):** Runs the K-medians algorithm with k=3, prints the head of the resulting DataFrame, final centroids, and plots the clustered data.
*   **Elbow Method (Cell 25):** Calculates and plots the total error (SAD) for different numbers of clusters (1 to 10) for K-medians.

## Results and Comparison (Conceptual - based on typical algorithm behavior)

*   **Centroid Calculation:** K-means uses the mean, which is sensitive to outliers. K-medians uses the median, making it more robust to outliers.
*   **Error Metric:** K-means minimizes the sum of squared Euclidean distances. K-medians minimizes the sum of Manhattan distances.
*   **Cluster Shapes:** K-means tends to find spherical clusters. K-medians can sometimes find clusters of different shapes, though this is more dependent on the data distribution.
*   **Performance:** The computational complexity per iteration is similar, but convergence behavior might differ.

The notebook visualizes the clusters formed by both algorithms and uses the elbow method to suggest an optimal number of clusters for each. The plots would show how data points are grouped and where the final centroids/medians are located.

## Conclusion

The notebook successfully implements both K-means and K-medians clustering algorithms. It demonstrates the key steps of initialization, assignment, and centroid/median updates. The use of helper functions for error calculation and centroid assignment makes the code modular. Visualizations and the elbow method provide tools for analyzing the clustering results. The K-medians algorithm offers an alternative that is less sensitive to outliers compared to K-means due to its use of medians and the Manhattan distance.
