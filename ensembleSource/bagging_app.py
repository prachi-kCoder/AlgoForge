import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

def generate_dataset(dataset_type, num_samples=300):
    if dataset_type == "Linearly Separable":
        X, y = make_blobs(n_samples=num_samples, centers=2, random_state=12)
    elif dataset_type == "U-shaped":
        X, y = make_moons(n_samples=num_samples, noise=0.1, random_state=12)
    elif dataset_type == "Outlier":
        X, y = make_blobs(n_samples=num_samples, centers=2, random_state=12)
        X[:10] += 10  # Adding outliers
    elif dataset_type == "Two Spirals":
        n = np.sqrt(np.random.rand(num_samples // 2, 1)) * 720 * (2 * np.pi) / 360
        d1x = -np.cos(n) * n + np.random.rand(num_samples // 2, 1)
        d1y = np.sin(n) * n + np.random.rand(num_samples // 2, 1)
        X = np.vstack((np.hstack((d1x, d1y)), np.hstack((-d1x, -d1y))))
        y = np.hstack((np.zeros(num_samples // 2), np.ones(num_samples // 2)))
    elif dataset_type == "Concentric Circles":
        X, y = make_circles(n_samples=num_samples, factor=0.5, noise=0.05, random_state=12)
    elif dataset_type == "XOR":
        X, y = make_blobs(n_samples=num_samples, centers=2, random_state=42)
        y = np.logical_xor(X[:, 0] > 0, X[:, 1] > 0)
    else:
        X, y = make_blobs(n_samples=num_samples, centers=2, random_state=12)  # Default
    return X, y

def draw_meshgrid(X):
    a = np.arange(start=X[:, 0].min() - 1, stop=X[:, 0].max() + 1, step=0.01)
    b = np.arange(start=X[:, 1].min() - 1, stop=X[:, 1].max() + 1, step=0.01)
    XX, YY = np.meshgrid(a, b)
    input_array = np.array([XX.ravel(), YY.ravel()]).T
    return XX, YY, input_array

def plot_decision_boundary(ax, clf, XX, YY, X, y, label, marker='o'):
    labels = clf.predict(np.c_[XX.ravel(), YY.ravel()])
    labels = labels.reshape(XX.shape)
    ax.contourf(XX, YY, labels, alpha=0.3, cmap='rainbow')
    ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow', marker=marker)
    ax.set_title(label)

def bagging_app():
    st.sidebar.markdown('# Bagging Classifier')

    # Allow user to select dataset type
    dataset_type = st.sidebar.selectbox(
        'Select Dataset Type',
        ('Linearly Separable', 'U-shaped', 'Outlier', 'Two Spirals', 'Concentric Circles', 'XOR')
    )

    # Allow user to select base estimator
    base_estimator_name = st.sidebar.selectbox(
        'Select Base Estimator',
        ('Decision Tree', 'SVM', 'K-Nearest Neighbors', 'Naive Bayes')
    )

    # Bagging Classifier Parameters
    n_estimators = st.sidebar.slider('Number of Estimators', min_value=1, max_value=50, value=10)

    # Generate the selected dataset
    X, y = generate_dataset(dataset_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Select the base estimator
    if base_estimator_name == 'Decision Tree':
        base_estimator = DecisionTreeClassifier(max_depth=st.sidebar.slider('Max Depth of Trees', min_value=1, max_value=10, value=3))
    elif base_estimator_name == 'SVM':
        base_estimator = SVC(probability=True)
    elif base_estimator_name == 'K-Nearest Neighbors':
        base_estimator = KNeighborsClassifier(n_neighbors=st.sidebar.slider('Number of Neighbors', min_value=1, max_value=20, value=5))
    elif base_estimator_name == 'Naive Bayes':
        base_estimator = GaussianNB()
    else:
        base_estimator = DecisionTreeClassifier()  # Default

    # Display the initial dataset
    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
    ax.set_title(f'{dataset_type} Dataset')
    st.pyplot(fig)

    if st.sidebar.button('Run Bagging Ensemble'):
        fig, main_ax = plt.subplots(1, 1, figsize=(10, 8))
        XX, YY, input_array = draw_meshgrid(X)

        # Bagging Classifier with selected base estimator
        bagging_clf = BaggingClassifier(
            estimator=base_estimator,
            n_estimators=n_estimators,
            random_state=42
        )
        bagging_clf.fit(X_train, y_train)

        # Plot decision boundary for Bagging Ensemble
        plot_decision_boundary(main_ax, bagging_clf, XX, YY, X, y, f'Bagging with {base_estimator_name}', marker='x')
        
        # Display metrics for the ensemble
        y_pred = bagging_clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='macro')
        recall = recall_score(y_test, y_pred, average='macro')
        f1 = f1_score(y_test, y_pred, average='macro')

        main_ax.text(0.5, -0.1, f'Accuracy: {round(accuracy, 2)}, Precision: {round(precision, 2)}, Recall: {round(recall, 2)}, F1 Score: {round(f1, 2)}', size=12, ha='center', transform=main_ax.transAxes)
        st.pyplot(fig)  # Display final ensemble decision boundary

        # Display accuracy metrics
        st.markdown(f"### Bagging Ensemble Model Metrics")
        st.markdown(f"- **Accuracy:** {accuracy:.2f}")
        st.markdown(f"- **Precision:** {precision:.2f}")
        st.markdown(f"- **Recall:** {recall:.2f}")
        st.markdown(f"- **F1 Score:** {f1:.2f}")

        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred)
        st.markdown("### Confusion Matrix")
        fig, ax = plt.subplots()
        cax = ax.matshow(cm, cmap='Blues')
        plt.colorbar(cax)
        for (i, j), val in np.ndenumerate(cm):
            ax.text(j, i, f'{val}', ha='center', va='center')
        st.pyplot(fig)


