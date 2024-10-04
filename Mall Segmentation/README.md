# Customer Segmentation using K-Means Clustering

This project focuses on segmenting customers based on their spending behavior using the **K-Means clustering algorithm**. Customer segmentation helps  businesses to identify and target specific groups of customers with tailored marketing strategies, increasing the effectiveness of their marketing.

---

## Table of Contents

1. [Introduction](#introduction)
2. [Dataset](#dataset)
3. [Project Steps](#projects-steps)
    - Data Processing
    - K-Means Clustering
    - Elbow Method
    - Cluster Visualization
    - Cluster Evaluation
4. [Results](#results)
5. [Conclusion](#conclusion)
6. [References](#references)

---

## Introduction

Customer segmentation is a key strategy for business to divide customers into distinct groups based on their purchasing behavior, income, and other important factors. In this project, **K-Means clustering** algorithm is used to create customer segments.

---

## Dataset

The dataset used for this project contains information such as:

- Customer ID
- Gender
- Age
- Annual Income (in thousands of dollars)
- Spending Score (on a scale of 1-100)

These features help in categorizing customers into meaningful groups

---

## Project Steps
### 1. Data Processing

The dataset was loaded and the following processing steps were performed:
- **Feature selection**:  Only the relevant features were selected for clustering i.e. `Age, Annual Income and Spending Score`.
- **Data Scaling**: Applied **StandardScaler** to normalize the features before clustering. K-Means is sensitive to the scale of data.

### 2. K-Means Clustering

**K-Means**, a popular clustering algorithm, was used to assign customers to distinct clusters. The key parameters were:

- `n_clusters`: Number of clusters to create.
- `init`: Method for initializing the centroids (`k-means++`).
- `max_iter`: Maximum number of iterations.

### 3. Elbow Method

To  determine the optimal number of clusters, the **Elbow Method** was used. This method plots the **Within-Cluster Sum of Squares (WCSS)** for different values of `k`. The "elbow" in the graph indicates the ideal number of clusters.

### 4. Cluster Visualization

Two types of plots were used:

- **2D Scatter plot**: To display customer groups using two features (income and spending score)
- **3D Scatter plot**: For a more detailed view of customer segmentation across three features.

###  5. Cluster Evaluation

The quality of the clusters was evaluated using the **Silhouette Score**, which means how well-defined the clusters are. Scores range between -1 and 1, where a score closer to 1 indicates that points are well-matched to their clusters and far from other clusters.

---

## Results

The **K-Means** algorithm successfully segmented customers into distinct groups:

- **Cluster 1**: High-income, high-spending customers.
- **Cluster 2**: Low-income, high-spending customers (potential impulse buyers).
- **Cluster 3**: Low-income, low-spending customers.
- **Cluster 4**: High-income, low-spending customers.

The clusters were visualized in both 2D and 3D, and evaluated the segmentation using the silhouette score.

---

## Conclusion

Customer segmentation using K-Means can help businesses to:
- Identify key customer segments.
- Develop targeted marketing strategies.
- Improve customer satisfaction and retention.

This project demonstrates the power of machine learning in **unsupervised learning** and provides valuable insights for real-world customer behavior analysis.

---

## References
- [K-Means Clustering on Scikit-Learn Documentation](https://scikit-learn.org/stable/modules/clustering.html#k-means)
- [Mall Customers Dataset](https://www.kaggle.com/vjchoudhary7/customer-segmentation-tutorial-in-python)
- [Matplotlib Visualization Library](https://matplotlib.org/stable/contents.html)