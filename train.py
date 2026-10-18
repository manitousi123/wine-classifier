import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.30
N_NEIGHBORS = 5
CV_FOLDS = 5

def load_dataset():
    wine = load_wine(as_frame=True)
    X = wine.data
    y = wine.target

    print("Dataset preview:")
    print(X.head())
    print(f"\nFeature shape: {X.shape}")
    print("\nClass counts:")
    print(y.value_counts(sort=False))

    return wine, X, y


def split_dataset(X, y):
    return train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

def evaluate_baseline_and_pipeline(X_train, X_test, y_train, y_test):
    baseline_model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    scaled_model = make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=N_NEIGHBORS),
    )
    scaled_model.fit(X_train, y_train)
    scaled_predictions = scaled_model.predict(X_test)
    scaled_accuracy = accuracy_score(y_test, scaled_predictions)

    print(f"\nUnscaled KNN test accuracy: {baseline_accuracy:.3f}")
    print(f"Scaled pipeline test accuracy: {scaled_accuracy:.3f}")
    print(
        "Accuracy improvement from scaling: "
        f"{scaled_accuracy - baseline_accuracy:+.3f}"
    )

    return scaled_model, scaled_predictions

def report_model(scaled_model, X_train, y_train, y_test, predictions, target_names):
    cv_scores = cross_val_score(
        scaled_model,
        X_train,
        y_train,
        cv=CV_FOLDS,
    )
    mean_cv_accuracy = sum(cv_scores) / len(cv_scores)

    print(f"\n{CV_FOLDS}-fold cross-validation scores: {cv_scores}")
    print(f"Mean cross-validation accuracy: {mean_cv_accuracy:.3f}")
    print("\nClassification report:")
    print(
        classification_report(
            y_test,
            predictions,
            target_names=target_names,
            zero_division=0,
        )
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        predictions,
        display_labels=target_names,
    )
    plt.show()  


def main():
    wine, X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    scaled_model, scaled_predictions = evaluate_baseline_and_pipeline(
        X_train,
        X_test,
        y_train,
        y_test,
    )
    report_model(
        scaled_model,
        X_train,
        y_train,
        y_test,
        scaled_predictions,
        wine.target_names,
    )
    
    
if __name__ == "__main__":
    main()