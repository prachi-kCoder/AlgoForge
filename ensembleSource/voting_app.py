
import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs, make_moons, make_circles
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import VotingClassifier
from sklearn.metrics import accuracy_score, r2_score

# Function to generate different types of datasets
def generate_dataset(dataset_type, num_samples=300):
    if dataset_type == "Linearly Separable":
        X, y = make_blobs(n_samples=num_samples, centers=2, random_state=12)
    elif dataset_type == "U-shaped":
        X, y = make_moons(n_samples=num_samples, noise=0.1, random_state=12)
    elif dataset_type == "Outlier":
        X, y = make_blobs(n_samples=num_samples, centers=2, random_state=12)
        # Adding outliers
        X[:10] += 10
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
    ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow', edgecolors='k', marker=marker)
    ax.set_title(label)

def voting_app():
    st.sidebar.markdown('# Voting Classifier')

    # Allow user to select dataset type
    dataset_type = st.sidebar.selectbox(
        'Select Dataset Type',
        ('Linearly Separable', 'U-shaped', 'Outlier', 'Two Spirals', 'Concentric Circles', 'XOR')
    )

    # Allow user to select which classifiers to use
    classifiers = st.sidebar.multiselect(
        'Select Classifiers to Include',
        ['Logistic Regression', 'KNN', 'Decision Tree', 'SVM', 'Gaussian Naive Bayes']
    )

    # Voting Classifier Parameters
    voting_type = st.sidebar.selectbox('Voting Type', ('hard', 'soft'))
    num_neighbors = st.sidebar.slider('Number of Neighbors (KNN)', min_value=1, max_value=10, value=5)
    max_depth_tree = st.sidebar.slider('Max Depth (Decision Tree)', min_value=1, max_value=10, value=3)
    svm_kernel = st.sidebar.selectbox('SVM Kernel', ('linear', 'rbf', 'poly'))
    nb_var_smoothing = st.sidebar.slider('Naive Bayes Variance Smoothing', min_value=1e-9, max_value=1e-6, value=1e-8, format="%.8f")

    # Generate the selected dataset
    X, y = generate_dataset(dataset_type)
    X_train, X_test, y_train, y_test = train_test_split(X, y, random_state=42)

    # Display the initial dataset and decision boundaries of individual classifiers
    fig, axes = plt.subplots(
        nrows=1, ncols=1 + len(classifiers), figsize=(15, 8),
        gridspec_kw={'width_ratios': [1] + [0.5] * len(classifiers)}
    )
    
    # Handle case where only one axis is created
    if len(classifiers) > 0:
        main_ax, small_axes = axes[0], axes[1:]
    else:
        main_ax = axes
        small_axes = []

    # Main plot: Initial dataset
    main_ax.scatter(X.T[0], X.T[1], c=y, cmap='rainbow')
    main_ax.set_title(f'{dataset_type} Dataset')

    # Show decision boundaries for individual classifiers
    for ax, clf_name in zip(small_axes, classifiers):
        if clf_name == 'Logistic Regression':
            clf = LogisticRegression()
        elif clf_name == 'KNN':
            clf = KNeighborsClassifier(n_neighbors=num_neighbors)
        elif clf_name == 'Decision Tree':
            clf = DecisionTreeClassifier(max_depth=max_depth_tree)
        elif clf_name == 'SVM':
            clf = SVC(kernel=svm_kernel, probability=True)
        elif clf_name == 'Gaussian Naive Bayes':
            clf = GaussianNB(var_smoothing=nb_var_smoothing)
        else:
            continue

        clf.fit(X_train, y_train)
        XX, YY, input_array = draw_meshgrid(X)
        plot_decision_boundary(ax, clf, XX, YY, X, y, f'{clf_name} Decision Boundary', marker='o')
        
        # Calculate and display accuracy
        y_pred = clf.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        ax.text(0.5, -0.1, f'Accuracy: {round(accuracy, 2)}', size=12, ha='center', transform=ax.transAxes)

    st.pyplot(fig)  # Show initial dataset and individual classifiers' boundaries

    if st.sidebar.button('Run Voting Ensemble'):
        fig, main_ax = plt.subplots(1, 1, figsize=(10, 8))
        XX, YY, input_array = draw_meshgrid(X)

        # Plot decision boundaries for individual classifiers
        for clf_name in classifiers:
            if clf_name == 'Logistic Regression':
                clf = LogisticRegression()
            elif clf_name == 'KNN':
                clf = KNeighborsClassifier(n_neighbors=num_neighbors)
            elif clf_name == 'Decision Tree':
                clf = DecisionTreeClassifier(max_depth=max_depth_tree)
            elif clf_name == 'SVM':
                clf = SVC(kernel=svm_kernel, probability=True)
            elif clf_name == 'Gaussian Naive Bayes':
                clf = GaussianNB(var_smoothing=nb_var_smoothing)
            else:
                continue

            clf.fit(X_train, y_train)
            plot_decision_boundary(main_ax, clf, XX, YY, X, y, f'{clf_name} Decision Boundary', marker='x')

        if classifiers:
            # Voting Classifier
            selected_classifiers = [(name, clf) for name, clf in {
                'Logistic Regression': LogisticRegression(),
                'KNN': KNeighborsClassifier(n_neighbors=num_neighbors),
                'Decision Tree': DecisionTreeClassifier(max_depth=max_depth_tree),
                'SVM': SVC(kernel=svm_kernel, probability=True),
                'Gaussian Naive Bayes': GaussianNB(var_smoothing=nb_var_smoothing)
            }.items() if name in classifiers]
            
            if selected_classifiers:
                voting_clf = VotingClassifier(
                    estimators=selected_classifiers,
                    voting=voting_type
                )
                voting_clf.fit(X_train, y_train)
                y_pred = voting_clf.predict(X_test)

                # Plot ensemble decision boundary
                plot_decision_boundary(main_ax, voting_clf, XX, YY, X, y, 'Ensemble Decision Boundary', marker='.')
                
                # Display metrics for the ensemble
                accuracy = accuracy_score(y_test, y_pred)
                main_ax.text(0.5, -0.1, f'Ensemble Accuracy: {round(accuracy, 2)}', size=12, ha='center', transform=main_ax.transAxes)
                
                st.pyplot(fig)  # Display final ensemble decision boundary

                # Display accuracy metrics
                st.markdown(f"### Ensemble Model Accuracy: {accuracy:.2f}")
            else:
                st.warning("No classifiers selected for voting.")
        else:
            st.warning("No classifiers selected for voting.")
