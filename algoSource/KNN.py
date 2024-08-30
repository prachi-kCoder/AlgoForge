import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score,  classification_report

def load_initial_graph(dataset, clusters, ax):
    if dataset == "Binary":
        clusters = 2
    X, y = make_blobs(n_features=2, centers=clusters, random_state=12 if dataset == 'Binary' else 2)
    ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
    return X, y

def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array

def knn_app():
    plt.style.use('Solarize_Light2')
    st.sidebar.markdown('# K-Nearest Neighbors Classifier')

    dataset = st.sidebar.selectbox('Select Dataset', ('Binary', 'Multiclass'))
    if dataset == 'Binary':
        num_clusters = 2
    else:
        num_clusters = st.sidebar.slider('Number of Clusters:', min_value=2, max_value=10, value=3)
    
    n_neighbors = st.sidebar.slider("Number of Neighbors",min_value=1, max_value=20, value=5)
    weights = st.sidebar.selectbox('Weights', ('uniform', 'distance'))
    algorithm = st.sidebar.selectbox('Algorithm', ('auto', 'ball_tree', 'kd_tree', 'brute'))

    fig, ax = plt.subplots()
    X, y = load_initial_graph(dataset, num_clusters, ax)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    orig = st.pyplot(fig)

    if st.sidebar.button("Run KNN"):
        orig.empty()
        clf = KNeighborsClassifier(
            n_neighbors=n_neighbors,
            weights=weights,
            algorithm=algorithm
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        XX,YY,input_array = draw_meshgrid(X) 
        labels = clf.predict(input_array)
        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', edgecolor='k')
        ax.set_xlabel('Feature 1')
        ax.set_ylabel('Feature 2')
        ax.set_title('KNN Decision Boundary')
        orig = st.pyplot(fig)

        accuracy = accuracy_score(y_test,y_pred)
        report = classification_report(y_test, y_pred, output_dict=True)
        st.code(f"Accuracy for KNN: {round(accuracy, 2)}", language='python')
        st.write("Classification Report:")
        st.json(report)

