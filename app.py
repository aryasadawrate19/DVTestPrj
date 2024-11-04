import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, PolynomialFeatures
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import mean_squared_error, r2_score, accuracy_score, classification_report
from sklearn.cluster import KMeans
from sklearn.pipeline import make_pipeline

# Set up Streamlit configuration
st.title("Interactive Machine Learning Model Selector")
st.write("Choose an ML algorithm and see the performance and visualizations.")

# Sidebar for user selection
st.sidebar.header("Select Algorithm")
algorithm = st.sidebar.selectbox("Algorithm", ("K-Means Clustering", "Logistic Regression", "Linear Regression"))

# Add a slider for number of clusters only if K-Means is selected
n_clusters = None
if algorithm == "K-Means Clustering":
    n_clusters = st.sidebar.slider("Select number of clusters", min_value=2, max_value=5, value=2)

# Generate Synthetic Dataset
def generate_classification_data():
    np.random.seed(42)
    X1 = np.random.normal(loc=5.0, scale=1.5, size=(100, 2))
    X2 = np.random.normal(loc=2.0, scale=1.5, size=(100, 2))
    X = np.vstack((X1, X2))
    y = np.hstack((np.zeros(100), np.ones(100)))
    return X, y

def generate_regression_data():
    np.random.seed(42)
    X = np.random.rand(200, 1) * 10  # Random values between 0 and 10
    y = 3.5 * X.flatten() + np.random.normal(0, 3, 200)  # Linear relationship with noise
    return X, y

# Data Preparation
if algorithm == "Linear Regression":
    X, y = generate_regression_data()
else:
    X, y = generate_classification_data()

# Data Standardization
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the data (only used for regression and logistic regression)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.3, random_state=42)

# Initialize and train the model based on the selected algorithm
def train_model(algorithm, n_clusters=None):
    if algorithm == "K-Means Clustering":
        model = KMeans(n_clusters=n_clusters, random_state=42)
        model.fit(X_scaled)
        y_pred = model.predict(X_scaled)
        st.write("### K-Means Clustering Results")
        st.write(f"**Number of Clusters:** {n_clusters}")
        st.write(f"**Cluster Centers:**")
        st.dataframe(pd.DataFrame(model.cluster_centers_, columns=["Feature 1", "Feature 2"]))
        return model, y_pred

    elif algorithm == "Logistic Regression":
        model = LogisticRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        st.write("### Logistic Regression Results")
        st.metric(label="Accuracy", value=f"{accuracy:.2f}")
        st.write("**Classification Report:**")
        st.text(classification_report(y_test, y_pred, target_names=["Class 0", "Class 1"]))
        return model, y_pred

    elif algorithm == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.write("### Linear Regression Results")
        st.metric(label="Mean Squared Error", value=f"{mse:.2f}")
        st.metric(label="R² Score", value=f"{r2:.2f}")
        return model, y_pred

# Train the model
model, y_pred = train_model(algorithm, n_clusters if algorithm == "K-Means Clustering" else None)

# Visualization functions
def plot_clustering_results(X, y_pred, model):
    from sklearn.decomposition import PCA
    pca = PCA(n_components=2)
    X_pca = pca.fit_transform(X)
    cluster_centers = pca.transform(model.cluster_centers_)

    plt.figure(figsize=(8, 6))
    sns.scatterplot(x=X_pca[:, 0], y=X_pca[:, 1], hue=y_pred, palette='viridis', s=60, alpha=0.7)

    # Draw circles around the clusters
    for center in cluster_centers:
        circle = plt.Circle(center, 1.0, color='black', fill=False, linewidth=2, linestyle='--')
        plt.gca().add_patch(circle)

    # Mark cluster centers
    plt.scatter(cluster_centers[:, 0], cluster_centers[:, 1], s=200, c='red', marker='X', label="Centroids")
    plt.legend()
    plt.title("K-Means Clustering with Cluster Boundaries")
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    st.pyplot(plt)

def plot_decision_boundary(X, y, model):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01), np.arange(y_min, y_max, 0.01))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    plt.figure(figsize=(8, 5))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='viridis')
    sns.scatterplot(x=X[:, 0], y=X[:, 1], hue=y, palette='viridis', s=60)
    plt.title("Logistic Regression Decision Boundary")
    plt.xlabel("Feature 1")
    plt.ylabel("Feature 2")
    st.pyplot(plt)

def plot_regression_results(X, y, y_pred):
    plt.figure(figsize=(8, 5))
    plt.scatter(X, y, color="blue", label="Actual")
    plt.plot(X, y_pred, color="red", label="Prediction")
    plt.title("Linear Regression Fit")
    plt.xlabel("Feature")
    plt.ylabel("Target")
    plt.legend()
    st.pyplot(plt)

# Display the appropriate visualization
if algorithm == "K-Means Clustering":
    plot_clustering_results(X_scaled, y_pred, model)
elif algorithm == "Logistic Regression":
    plot_decision_boundary(X_scaled, y, model)
elif algorithm == "Linear Regression":
    plot_regression_results(X_test, y_test, y_pred)
