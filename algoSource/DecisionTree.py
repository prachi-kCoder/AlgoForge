import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_classification
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split
import streamlit as st

def generate_data(n_samples, n_features, n_classes):
    n_informative = min(n_features, int(np.ceil(np.log2(n_classes * 2))))
    X, y = make_classification(
        n_samples=n_samples,
        n_features=n_features,
        n_informative=n_informative,
        n_redundant=0,
        n_classes=n_classes,
        random_state=42
    )
    return X, y

def plot_decision_boundary(ax, model, X, y, resolution=0.02):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution),
                         np.arange(y_min, y_max, resolution))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='coolwarm')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='coolwarm', edgecolor='k')

def decision_tree_app():
    st.title("Decision Tree Visualization")

    n_samples = st.sidebar.slider('Number of Samples', 100, 1000, 200)
    n_classes = st.sidebar.slider("Number of Classes", 2, 4, 2)
    max_depth = st.sidebar.slider("Max Depth", 1, 10, 3)

    X, y = generate_data(n_samples=n_samples, n_features=2, n_classes=n_classes)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

    clf = DecisionTreeClassifier(max_depth=max_depth)
    clf.fit(X_train, y_train)

    # Plot decision boundary
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_decision_boundary(ax, clf, X, y)
    ax.set_xlabel('Feature 1')
    ax.set_ylabel('Feature 2')
    ax.set_title(f'Decision Tree Decision Boundary (max_depth={max_depth})')
    st.pyplot(fig)

    # Plot decision tree
    fig, ax = plt.subplots(figsize=(15, 10))
    plot_tree(clf, filled=True, feature_names=['Feature 1', 'Feature 2'],
              class_names=[f'Class {i}' for i in range(n_classes)], ax=ax)
    ax.set_title('Decision Tree Visualization')
    st.pyplot(fig)
