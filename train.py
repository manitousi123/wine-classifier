from sklearn.datasets import load_wine
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier

RANDOM_STATE = 42
TEST_SIZE = 0.30
N_NEIGHBORS = 5

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


def main():
    wine, X, y = load_dataset()
    X_train, X_test, y_train, y_test = split_dataset(X, y)

    baseline_model = KNeighborsClassifier(n_neighbors=N_NEIGHBORS)
    baseline_model.fit(X_train, y_train)
    baseline_predictions = baseline_model.predict(X_test)
    baseline_accuracy = accuracy_score(y_test, baseline_predictions)

    print(f"\nUnscaled KNN test accuracy: {baseline_accuracy:.3f}")


if __name__ == "__main__":
    main()