import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, r2_score

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

def logistic_regression_app():
    plt.style.use('Solarize_Light2')
    st.sidebar.markdown('# Logistic Regression Classifier')

    dataset = st.sidebar.selectbox('Select Dataset', ('Binary', 'Multiclass'))
    if dataset == 'Binary':
        num_clusters = 2
    else:
        num_clusters = st.sidebar.slider('Number of Clusters:', min_value=2, max_value=10, value=3)

    penalty = st.sidebar.selectbox('Regularization', ('l2', 'l1', 'elasticnet', 'none'))
    c_input = float(st.sidebar.number_input('C', value=1.0))
    solver = st.sidebar.selectbox('Solver', ('newton-cg', 'lbfgs', 'liblinear', 'sag', 'saga'))
    max_iter = int(st.sidebar.number_input('Max Iterations', value=100))
    multi_class = st.sidebar.selectbox('Multi class', ('auto', 'ovr', 'multinomial'))

    if penalty == 'elasticnet':
        l1_ratio = float(st.sidebar.number_input('l1 Ratio', min_value=0.0, max_value=1.0, value=0.5))
    else:
        l1_ratio = None

    fig, ax = plt.subplots()
    X, y = load_initial_graph(dataset, num_clusters, ax)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)
    orig = st.pyplot(fig)

    if st.sidebar.button('Run Logistic Regression'):
        orig.empty()
        clf = LogisticRegression(
            penalty=penalty,
            C=c_input,
            solver=solver,
            max_iter=max_iter,
            multi_class=multi_class,
            l1_ratio=l1_ratio
        )
        clf.fit(X_train, y_train)
        y_pred = clf.predict(X_test)

        XX, YY, input_array = draw_meshgrid(X)
        labels = clf.predict(input_array)
        ax.contourf(XX, YY, labels.reshape(XX.shape), alpha=0.5, cmap='rainbow')
        plt.xlabel('Col1')
        plt.ylabel('Col2')
        plt.legend(['Decision Boundary', 'Data Points'])
        orig = st.pyplot(fig)

        accuracy = accuracy_score(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.code(f"Accuracy for Logistic Regression: {round(accuracy, 2)}", language='python')
        st.code(f"R2 Score for Logistic Regression: {round(r2, 2)}", language='python')
