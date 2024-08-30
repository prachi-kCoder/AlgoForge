import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_regression, make_moons, make_circles
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.svm import SVR
from sklearn.neighbors import KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB

def generate_dataset(dataset_type, num_samples=300):
    if dataset_type == "Linear Regression":
        X, y = make_regression(n_samples=num_samples, n_features=1, noise=0.1, random_state=12)
    elif dataset_type == "U-shaped":
        X, y = make_moons(n_samples=num_samples, noise=0.1, random_state=12)
        X = X[:, 0].reshape(-1, 1)  # Make it single-feature for regression
    elif dataset_type == "Outlier":
        X, y = make_regression(n_samples=num_samples, n_features=1, noise=0.1, random_state=12)
        y[:10] += 50  # Adding outliers
    elif dataset_type == "Two Spirals":
        n = np.sqrt(np.random.rand(num_samples // 2, 1)) * 720 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(num_samples // 2, 1)
        d1y = np.sin(n) * n + np.random.rand(num_samples // 2, 1)
        X = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
        y = np.hstack((np.zeros(num_samples // 2), np.ones(num_samples // 2)))
        X = X[:, 0].reshape(-1, 1)  # Make it single-feature for regression
    elif dataset_type == "Concentric Circles":
        X, y = make_circles(n_samples=num_samples, factor=0.5, noise=0.05, random_state=12)
        X = X[:, 0].reshape(-1, 1)  # Make it single-feature for regression
    else:
        X, y = make_regression(n_samples=num_samples, n_features=1, noise=0.1, random_state=12)  # Default
    return X, y

def draw_meshgrid(X):
    a = np.linspace(X.min() - 1, X.max() + 1, 100)
    XX = a.reshape(-1, 1)
    return XX

def plot_decision_boundary(ax, regressor, X, y, label):
    XX = draw_meshgrid(X)
    y_pred = regressor.predict(XX)
    ax.plot(XX, y_pred, color='red')
    ax.scatter(X, y, color='blue')
    ax.set_title(label)

def boosting_regressor_app():
    st.sidebar.markdown('# Boosting Regressor')

    # Allow user to select dataset type
    dataset_type = st.sidebar.selectbox(
        'Select Dataset Type',
        ('Linear Regression', 'U-shaped', 'Outlier', 'Two Spirals', 'Concentric Circles')
    )

    # Allow user to select base estimator
    base_estimator_name = st.sidebar.selectbox(
        'Select Base Estimator',
        ('Gradient Boosting', 'Histogram-based Gradient Boosting', 'SVM', 'K-Nearest Neighbors')
    )

    # Boosting Regressor Parameters
    n_estimators = st.sidebar.slider('Number of Estimators', min_value=1, max_value=100, value=100)
    learning_rate = st.sidebar.slider('Learning Rate', min_value=0.01, max_value=1.0, value=0.1, step=0.01)

    # Generate the selected dataset
    X, y = generate_dataset(dataset_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

 # Select the base estimator
    if base_estimator_name == 'Gradient Boosting':
        regressor = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    elif base_estimator_name == 'Histogram-based Gradient Boosting':
        regressor = HistGradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)
    elif base_estimator_name == 'SVM':
        regressor = SVR(C=st.sidebar.slider('C', min_value=0.1, max_value=10.0, value=1.0), epsilon=st.sidebar.slider('Epsilon', min_value=0.01, max_value=1.0, value=0.1))
    elif base_estimator_name == 'K-Nearest Neighbors':
        regressor = KNeighborsRegressor(n_neighbors=st.sidebar.slider('Number of Neighbors', min_value=1, max_value=20, value=5))
    else:
        regressor = GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate, random_state=42)  # Default

    # Display the initial dataset
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X, y, color='blue')
    ax.set_title(f'{dataset_type} Dataset')
    st.pyplot(fig)

    if st.sidebar.button('Run Boosting Regressor'):
            regressor.fit(X_train, y_train)

            # Display regression results
            fig, main_ax = plt.subplots(figsize=(10, 6))
            plot_decision_boundary(main_ax, regressor, X, y, f'Boosting Regressor with {base_estimator_name}')
            
            # Display metrics for the regressor
            y_pred = regressor.predict(X_test)
            mse = mean_squared_error(y_test, y_pred)
            mae = mean_absolute_error(y_test, y_pred)
            r2 = r2_score(y_test, y_pred)

            main_ax.text(0.5, -0.1, f'MSE: {round(mse, 2)}, MAE: {round(mae, 2)}, R²: {round(r2, 2)}', size=12, ha='center', transform=main_ax.transAxes)
            st.pyplot(fig)  # Display final regression plot

            # Display regression metrics
            st.markdown(f"### Boosting Regressor Model Metrics")
            st.markdown(f"- **Mean Squared Error (MSE):** {mse:.2f}")
            st.markdown(f"- **Mean Absolute Error (MAE):** {mae:.2f}")
            st.markdown(f"- **R² Score:** {r2:.2f}")



