from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier, RandomForestClassifier

def train_models(X, y, test_size=0.2, random_state=42):
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )

    results = {}

    # --- SVM (tuned) ---
    svm_params = {'C': [0.1, 1, 10], 'kernel': ['rbf', 'linear']}
    grid_svm = GridSearchCV(SVC(probability=True), svm_params, cv=3)
    grid_svm.fit(X_train, y_train)
    svm_pred = grid_svm.predict(X_test)
    results['SVM'] = {
        'model': grid_svm.best_estimator_,
        'accuracy': accuracy_score(y_test, svm_pred),
        'report': classification_report(y_test, svm_pred, output_dict=True),
        'confusion': confusion_matrix(y_test, svm_pred)
    }

    # --- Decision Tree (tuned) ---
    dt_params = {'max_depth': [3, 5, 10], 'min_samples_split': [2, 5]}
    grid_dt = GridSearchCV(DecisionTreeClassifier(), dt_params, cv=3)
    grid_dt.fit(X_train, y_train)
    dt_pred = grid_dt.predict(X_test)
    results['DecisionTree'] = {
        'model': grid_dt.best_estimator_,
        'accuracy': accuracy_score(y_test, dt_pred),
        'report': classification_report(y_test, dt_pred, output_dict=True),
        'confusion': confusion_matrix(y_test, dt_pred)
    }

    # --- KNN (tuned) ---
    knn_params = {'n_neighbors': list(range(3, 11))}
    grid_knn = GridSearchCV(KNeighborsClassifier(), knn_params, cv=3)
    grid_knn.fit(X_train, y_train)
    knn_pred = grid_knn.predict(X_test)
    results['KNN'] = {
        'model': grid_knn.best_estimator_,
        'accuracy': accuracy_score(y_test, knn_pred),
        'report': classification_report(y_test, knn_pred, output_dict=True),
        'confusion': confusion_matrix(y_test, knn_pred)
    }

    # --- ANN (tuned) ---
    ann_params = {
        'hidden_layer_sizes': [(64,), (64, 32)],
        'activation': ['relu', 'tanh'],
        'solver': ['adam'],
        'max_iter': [500]
    }
    grid_ann = GridSearchCV(MLPClassifier(random_state=random_state), ann_params, cv=3)
    grid_ann.fit(X_train, y_train)
    ann_pred = grid_ann.predict(X_test)
    results['ANN'] = {
        'model': grid_ann.best_estimator_,
        'accuracy': accuracy_score(y_test, ann_pred),
        'report': classification_report(y_test, ann_pred, output_dict=True),
        'confusion': confusion_matrix(y_test, ann_pred)
    }

    # --- Random Forest ---
    rf = RandomForestClassifier(n_estimators=100, random_state=random_state)
    rf.fit(X_train, y_train)
    rf_pred = rf.predict(X_test)
    results['RandomForest'] = {
        'model': rf,
        'accuracy': accuracy_score(y_test, rf_pred),
        'report': classification_report(y_test, rf_pred, output_dict=True),
        'confusion': confusion_matrix(y_test, rf_pred)
    }

    # --- VotingClassifier Ensemble ---
    ensemble = VotingClassifier(
        estimators=[
            ('svm', grid_svm.best_estimator_),
            ('dt', grid_dt.best_estimator_),
            ('knn', grid_knn.best_estimator_),
            ('ann', grid_ann.best_estimator_),
            ('rf', rf)
        ],
        voting='soft'
    )
    ensemble.fit(X_train, y_train)
    ensemble_pred = ensemble.predict(X_test)
    results['Ensemble'] = {
        'model': ensemble,
        'accuracy': accuracy_score(y_test, ensemble_pred),
        'report': classification_report(y_test, ensemble_pred, output_dict=True),
        'confusion': confusion_matrix(y_test, ensemble_pred)
    }

    return results

