import matplotlib.pyplot as plt

from sklearn.datasets import load_wine
from sklearn.metrics import ConfusionMatrixDisplay, accuracy_score, classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_STATE = 42
TEST_SIZE = 0.30
FIXED_N_NEIGHBORS = 5
CANDIDATE_NEIGHBORS = (1, 3, 5, 7, 9, 11)
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


def build_scaled_model(n_neighbors):
    return make_pipeline(
        StandardScaler(),
        KNeighborsClassifier(n_neighbors=n_neighbors),
    )


def compare_fixed_models(X_train, X_test, y_train, y_test):
    baseline_model = KNeighborsClassifier(n_neighbors=FIXED_N_NEIGHBORS)
    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    scaled_model = build_scaled_model(FIXED_N_NEIGHBORS)
    scaled_model.fit(X_train, y_train)
    scaled_predictions = scaled_model.predict(X_test)
    scaled_accuracy = accuracy_score(y_test, scaled_predictions)

    print(f"\nUnscaled KNN test accuracy: {baseline_accuracy:.3f}")
    print(f"Scaled pipeline test accuracy: {scaled_accuracy:.3f}")
    print(
        "Accuracy improvement from scaling: "
        f"{scaled_accuracy - baseline_accuracy:+.3f}"
    )


def tune_neighbor_count(X_train, y_train):
    mean_scores = {}

    print("\nTraining-only neighbor search:")
    for n_neighbors in CANDIDATE_NEIGHBORS:
        candidate_model = build_scaled_model(n_neighbors)
        scores = cross_val_score(
            candidate_model,
            X_train,
            y_train,
            cv=CV_FOLDS,
        )
        mean_score = sum(scores) / len(scores)
        mean_scores[n_neighbors] = mean_score
        print(f"n_neighbors={n_neighbors}: mean CV accuracy={mean_score:.3f}")

    best_n_neighbors = max(mean_scores, key=mean_scores.get)
    print(f"Selected n_neighbors: {best_n_neighbors}")
    return best_n_neighbors


def evaluate_tuned_model(
    best_n_neighbors,
    X_train,
    X_test,
    y_train,
    y_test,
    target_names,
):
    tuned_model = build_scaled_model(best_n_neighbors)
    tuned_model.fit(X_train, y_train)
    tuned_predictions = tuned_model.predict(X_test)
    tuned_accuracy = accuracy_score(y_test, tuned_predictions)

    print(f"Tuned holdout accuracy: {tuned_accuracy:.3f}")
    print("\nTuned classification report:")
    print(
        classification_report(
            y_test,
            tuned_predictions,
            target_names=target_names,
            zero_division=0,
        )
    )

    ConfusionMatrixDisplay.from_predictions(
        y_test,
        tuned_predictions,
        display_labels=target_names,
    )
    plt.show() 


def main():
    wine, X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)
    compare_fixed_models(X_train, X_test, y_train, y_test)
    best_n_neighbors = tune_neighbor_count(X_train, y_train)
    evaluate_tuned_model(
        best_n_neighbors,
        X_train,
        X_test,
        y_train,
        y_test,
        wine.target_names,
    )  
    
if __name__ == "__main__":
    main()