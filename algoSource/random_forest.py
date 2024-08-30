import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_circles
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

def plot_decision_boundary(ax, model, X, y, resolution=0.02):
    x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, resolution), 
                         np.arange(y_min, y_max, resolution))
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)
    ax.contourf(xx, yy, Z, alpha=0.4, cmap='rainbow')
    ax.scatter(X[:, 0], X[:, 1], c=y, cmap='rainbow', edgecolor='k')

def random_forest_app():
    plt.style.use('Solarize_Light2')
    st.sidebar.markdown('# Random Forest Classifier')

    X, y = make_circles(n_samples=400, noise=0.3, factor=0.5, random_state=42)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    fig, ax = plt.subplots(figsize=(10, 8))
    # Initial plot with no decision boundary
    ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow', edgecolor='k')
    ax.set_xlabel('Col1')
    ax.set_ylabel('Col2')
    ax.legend(['Data Points'])
    st.pyplot(fig)

    n_estimators = int(st.sidebar.number_input('Num Estimators', min_value=5, max_value=25, value=10))
    bootstrap = st.sidebar.selectbox('Bootstrap', ['True', 'False'])
    criterion = st.sidebar.selectbox('Criterion', ['gini', 'entropy'])
    max_features = st.sidebar.slider('Max Features', min_value=2, max_value=8, value=4)
    max_samples = st.sidebar.slider('Max Samples', min_value=100, max_value=400, value=200)

    if st.sidebar.button('Run Random Forest'):
        rf = RandomForestClassifier(
            n_estimators=n_estimators,
            max_features=max_features,
            bootstrap=(bootstrap == 'True'),
            criterion=criterion,
            max_samples=max_samples
        )
        rf.fit(X_train, y_train)
        y_pred = rf.predict(X_test)

        ax.clear()
        plot_decision_boundary(ax, rf, X, y)
        st.pyplot(fig)

        accuracy = accuracy_score(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        st.code(f"Accuracy = {round(accuracy, 2)}", language='python')
        st.code(f"R2 Score = {round(r2, 2)}", language='python')

if __name__ == "__main__":
    random_forest_app()
