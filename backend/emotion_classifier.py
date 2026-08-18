# -*- coding: utf-8 -*-

"""
Sakina Emotion Classifier — v13

Trains and persists the Arabic emotion classifier used by Sakina.

Architecture:
    Arabic normalization
        ↓
    FeatureUnion
        ├── Word TF-IDF (1–2 grams)
        └── Character TF-IDF (3–5 grams)
        ↓
    Linear SVM
        ↓
    Probability calibration

The classifier supports 12 emotional categories:

1. قلق وتوتر
2. حزن
3. فقدان أمل
4. وحدة
5. حيرة وتشتت
6. ذنب وتقصير
7. فقد وشوق
8. فرح وشكر
9. خوف
10. غضب
11. ضغط نفسي
12. شعور بالظلم

The classifier is responsible only for emotion prediction.

High-confidence lexical anchors and ambiguous-input handling
are implemented separately in the inference layer.
"""

import json
import pickle
import sys

from pathlib import Path

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    classification_report,
    accuracy_score,
    f1_score,
    confusion_matrix,
)


# ============================================================
# PROJECT PATHS
# ============================================================

ROOT = Path(__file__).resolve().parent.parent

DATA_DIR = ROOT / "data"
CHECKPOINT_DIR = ROOT / "checkpoints"
EVALUATION_DIR = ROOT / "evaluation"

DATA_FILE = DATA_DIR / "emotions_dataset.json"

MODEL_FILE = (
    CHECKPOINT_DIR
    / "sakina_emotion_classifier.pkl"
)

REPORT_FILE = (
    EVALUATION_DIR
    / "classifier_report.txt"
)


# ============================================================
# IMPORT ARABIC NORMALIZER
# ============================================================

sys.path.insert(
    0,
    str(DATA_DIR)
)

from normalize import (  # noqa: E402
    ArabicNormalizer,
)


# ============================================================
# EXPECTED EMOTIONS
# ============================================================

EXPECTED_EMOTIONS = [
    "قلق وتوتر",
    "حزن",
    "فقدان أمل",
    "وحدة",
    "حيرة وتشتت",
    "ذنب وتقصير",
    "فقد وشوق",
    "فرح وشكر",
    "خوف",
    "غضب",
    "ضغط نفسي",
    "شعور بالظلم",
]


# ============================================================
# LOAD DATA
# ============================================================

def load_data():
    """
    Load labeled emotion examples.

    Expected JSON structure:

    [
        {
            "user_input": "...",
            "detected_emotion": "..."
        }
    ]
    """

    if not DATA_FILE.exists():

        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )


    with open(
        DATA_FILE,
        "r",
        encoding="utf-8",
    ) as f:

        rows = json.load(f)


    X = []
    y = []


    for row in rows:

        text = (
            row
            .get("user_input", "")
            .strip()
        )

        emotion = (
            row
            .get("detected_emotion", "")
            .strip()
        )


        if not text or not emotion:
            continue


        X.append(text)
        y.append(emotion)


    if not X:

        raise ValueError(
            "Dataset contains no valid training examples."
        )


    return X, y


# ============================================================
# DATASET VALIDATION
# ============================================================

def validate_emotions(y):
    """
    Verify that the final dataset contains all
    12 Sakina emotion categories.
    """

    present_emotions = set(y)

    expected_set = set(EXPECTED_EMOTIONS)


    missing = (
        expected_set
        - present_emotions
    )


    unexpected = (
        present_emotions
        - expected_set
    )


    if missing:

        print(
            "\nWARNING: Missing emotion classes:"
        )

        for emotion in sorted(missing):
            print(f"  - {emotion}")


    if unexpected:

        print(
            "\nWARNING: Unexpected emotion classes:"
        )

        for emotion in sorted(unexpected):
            print(f"  - {emotion}")


    if missing:

        raise ValueError(
            "Dataset does not contain all "
            "12 required Sakina emotions."
        )


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

def print_class_distribution(y):

    from collections import Counter

    counts = Counter(y)


    print("\nClass distribution:")
    print("-" * 50)


    for emotion in EXPECTED_EMOTIONS:

        print(
            f"{emotion:<20} "
            f"{counts.get(emotion, 0)}"
        )


    print("-" * 50)
    print(
        f"Total examples: {len(y)}"
    )


# ============================================================
# MODEL PIPELINE
# ============================================================

def build_pipeline(
    char_ngram=(3, 5),
    class_weight="balanced",
):
    """
    Build Sakina Arabic emotion classifier.

    Word-level TF-IDF captures semantic phrasing.

    Character-level TF-IDF improves robustness
    to Arabic dialect spelling variations such as:

        سعيدة / سعيده
        متوترة / متوتره
        قلقانة / قلقانه
        مظلومة / مظلومه
        معصبة / معصبه
    """

    word_vectorizer = TfidfVectorizer(
        analyzer="word",
        ngram_range=(1, 2),
        min_df=1,
        sublinear_tf=True,
        lowercase=False,
    )


    char_vectorizer = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=char_ngram,
        min_df=2,
        sublinear_tf=True,
        lowercase=False,
    )


    features = FeatureUnion([
        (
            "word",
            word_vectorizer,
        ),
        (
            "char",
            char_vectorizer,
        ),
    ])


    base_svm = LinearSVC(
        C=1.0,
        class_weight=class_weight,
        random_state=42,
    )


    # CalibratedClassifierCV provides predict_proba(),
    # which Sakina uses for confidence and ambiguity logic.

    classifier = CalibratedClassifierCV(
        base_svm,
        cv=3,
    )


    pipeline = Pipeline([
        (
            "normalize",
            ArabicNormalizer(),
        ),
        (
            "features",
            features,
        ),
        (
            "clf",
            classifier,
        ),
    ])


    return pipeline


# ============================================================
# TRAIN
# ============================================================

def main():

    print(
        "\n"
        "=============================================="
    )

    print(
        "SAKINA EMOTION CLASSIFIER — v13"
    )

    print(
        "=============================================="
    )


    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    X, y = load_data()


    # --------------------------------------------------------
    # Validate emotion taxonomy
    # --------------------------------------------------------

    validate_emotions(y)


    print_class_distribution(y)


    # --------------------------------------------------------
    # Train / held-out split
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = (
        train_test_split(
            X,
            y,
            test_size=0.15,
            random_state=42,
            stratify=y,
        )
    )


    print(
        f"\nTrain examples: "
        f"{len(X_train)}"
    )

    print(
        f"Test examples:  "
        f"{len(X_test)}"
    )


    # --------------------------------------------------------
    # Build model
    # --------------------------------------------------------

    print(
        "\nBuilding classifier..."
    )


    pipeline = build_pipeline()


    # --------------------------------------------------------
    # Train
    # --------------------------------------------------------

    print(
        "Training classifier..."
    )


    pipeline.fit(
        X_train,
        y_train,
    )


    # --------------------------------------------------------
    # Predict held-out test
    # --------------------------------------------------------

    print(
        "Evaluating classifier..."
    )


    predictions = pipeline.predict(
        X_test
    )


    # --------------------------------------------------------
    # Metrics
    # --------------------------------------------------------

    accuracy = accuracy_score(
        y_test,
        predictions,
    )


    macro_f1 = f1_score(
        y_test,
        predictions,
        average="macro",
    )


    weighted_f1 = f1_score(
        y_test,
        predictions,
        average="weighted",
    )


    report = classification_report(
        y_test,
        predictions,
        labels=EXPECTED_EMOTIONS,
        zero_division=0,
    )


    matrix = confusion_matrix(
        y_test,
        predictions,
        labels=EXPECTED_EMOTIONS,
    )


    # --------------------------------------------------------
    # Print results
    # --------------------------------------------------------

    print(
        "\n"
        "=============================================="
    )

    print(
        "HELD-OUT EVALUATION"
    )

    print(
        "=============================================="
    )


    print(
        f"Accuracy:    "
        f"{accuracy:.4f}"
    )

    print(
        f"Macro F1:    "
        f"{macro_f1:.4f}"
    )

    print(
        f"Weighted F1: "
        f"{weighted_f1:.4f}"
    )


    print(
        "\nClassification Report:\n"
    )

    print(report)


    print(
        "\nConfusion Matrix:\n"
    )

    print(matrix)


    # ========================================================
    # SAVE MODEL
    # ========================================================

    CHECKPOINT_DIR.mkdir(
        exist_ok=True,
        parents=True,
    )


    with open(
        MODEL_FILE,
        "wb",
    ) as f:

        pickle.dump(
            pipeline,
            f,
        )


    # ========================================================
    # SAVE REPORT
    # ========================================================

    EVALUATION_DIR.mkdir(
        exist_ok=True,
        parents=True,
    )


    with open(
        REPORT_FILE,
        "w",
        encoding="utf-8",
    ) as f:

        f.write(
            "SAKINA EMOTION CLASSIFIER — v13\n"
        )

        f.write(
            "12 Emotion Classes\n"
        )

        f.write(
            "=" * 60
            + "\n\n"
        )


        f.write(
            f"Dataset: {DATA_FILE.name}\n"
        )

        f.write(
            f"Total examples: "
            f"{len(X)}\n"
        )

        f.write(
            f"Train examples: "
            f"{len(X_train)}\n"
        )

        f.write(
            f"Test examples: "
            f"{len(X_test)}\n\n"
        )


        f.write(
            f"Accuracy: "
            f"{accuracy:.4f}\n"
        )

        f.write(
            f"Macro F1: "
            f"{macro_f1:.4f}\n"
        )

        f.write(
            f"Weighted F1: "
            f"{weighted_f1:.4f}\n\n"
        )


        f.write(
            "Classification Report\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )

        f.write(report)


        f.write(
            "\n\nConfusion Matrix\n"
        )

        f.write(
            "-" * 60
            + "\n"
        )


        for row in matrix:

            f.write(
                " ".join(
                    str(value)
                    for value in row
                )
            )

            f.write("\n")


        f.write(
            "\nClass order:\n"
        )


        for index, emotion in enumerate(
            EXPECTED_EMOTIONS
        ):

            f.write(
                f"{index}: "
                f"{emotion}\n"
            )


    # ========================================================
    # COMPLETE
    # ========================================================

    print(
        "\n"
        "=============================================="
    )

    print(
        "TRAINING COMPLETE"
    )

    print(
        "=============================================="
    )


    print(
        f"Saved classifier -> "
        f"{MODEL_FILE}"
    )

    print(
        f"Saved report -> "
        f"{REPORT_FILE}"
    )


if __name__ == "__main__":
    main()