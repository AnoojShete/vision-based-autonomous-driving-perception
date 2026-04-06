import os
import time
from datetime import datetime

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, precision_recall_fscore_support
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier


TRAFFIC_SIGN_FAMILY = {
    "speed_limit": list(range(0, 9)),
    "prohibitory": list(range(9, 18)),
    "warning": list(range(18, 32)),
    "mandatory": list(range(32, 41)),
    "other": [41, 42],
}


def _algorithm_factory(algorithm):
    models = {
        "Decision Tree": DecisionTreeClassifier(random_state=42),
        "Naive Bayes": GaussianNB(),
        "SVM": SVC(kernel="rbf", gamma="scale"),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42, n_jobs=-1),
    }
    return models.get(algorithm, DecisionTreeClassifier(random_state=42))


def _safe_mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def _extract_sign_code(raw_value):
    raw = str(raw_value)
    tokens = []
    for token in raw.replace(".", " ").replace("-", " ").split():
        if token.isdigit():
            tokens.append(int(token))
    return sum(tokens) if tokens else 0


def _get_sign_family(class_id):
    for family, class_ids in TRAFFIC_SIGN_FAMILY.items():
        if int(class_id) in class_ids:
            return family
    return "unknown"


def _compute_image_features(image_path):
    image = cv2.imread(image_path)
    if image is None:
        return None

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    brightness = float(np.mean(gray))
    contrast = float(np.std(gray))
    blur_score = float(cv2.Laplacian(gray, cv2.CV_64F).var())

    return {
        "brightness": brightness,
        "contrast": contrast,
        "blur_score": blur_score,
    }


def load_driving_dataset(task="metadata", data_dir="data", max_samples=5000):
    """
    Load traffic-sign-oriented tabular datasets.

    task:
        - metadata: use Meta.csv fields (ShapeId, ColorId, SignId) -> ClassId
        - robustness: use Train/Test csv geometry + image quality features -> ClassId
    """
    if task == "metadata":
        meta_path = os.path.join(data_dir, "Meta.csv")
        train_path = os.path.join(data_dir, "Train.csv")
        meta_df = pd.read_csv(meta_path)
        train_df = pd.read_csv(train_path)

        meta_required = ["ClassId", "ShapeId", "ColorId", "SignId"]
        train_required = ["ClassId"]
        missing = [col for col in meta_required if col not in meta_df.columns]
        missing.extend([col for col in train_required if col not in train_df.columns])
        if missing:
            raise ValueError(f"Missing required columns for metadata task: {missing}")

        class_features = meta_df[meta_required].copy()
        class_features["SignCode"] = class_features["SignId"].apply(_extract_sign_code)
        class_features = class_features.drop(columns=["SignId"])

        model_df = train_df[["ClassId"]].copy().merge(class_features, on="ClassId", how="left")
        model_df = model_df.dropna(subset=["ShapeId", "ColorId", "SignCode"])
        if len(model_df) > max_samples:
            model_df = model_df.sample(n=max_samples, random_state=42)

        X = model_df[["ShapeId", "ColorId", "SignCode"]]
        y = model_df["ClassId"].astype(int)
        return X, y, model_df

    if task == "robustness":
        train_csv = os.path.join(data_dir, "Train.csv")
        df = pd.read_csv(train_csv)

        required = ["Width", "Height", "Roi.X1", "Roi.Y1", "Roi.X2", "Roi.Y2", "ClassId", "Path"]
        missing = [col for col in required if col not in df.columns]
        if missing:
            raise ValueError(f"Train.csv missing required columns: {missing}")

        rows = []
        robust_df = df
        if len(df) > max_samples:
            robust_df = df.sample(n=max_samples, random_state=42)

        for _, row in robust_df.iterrows():
            image_path = os.path.join(data_dir, str(row["Path"]))
            features = _compute_image_features(image_path)
            if features is None:
                continue

            width = float(row["Width"])
            height = float(row["Height"])
            roi_x1 = float(row["Roi.X1"])
            roi_y1 = float(row["Roi.Y1"])
            roi_x2 = float(row["Roi.X2"])
            roi_y2 = float(row["Roi.Y2"])
            roi_area = max(1.0, (roi_x2 - roi_x1) * (roi_y2 - roi_y1))
            full_area = max(1.0, width * height)

            rows.append(
                {
                    "ClassId": int(row["ClassId"]),
                    "width": width,
                    "height": height,
                    "roi_area_ratio": roi_area / full_area,
                    "aspect_ratio": width / max(height, 1.0),
                    **features,
                }
            )

        model_df = pd.DataFrame(rows)
        if model_df.empty:
            raise ValueError("Could not build robustness dataset. Check image paths under data/Train.")

        X = model_df.drop(columns=["ClassId"])
        y = model_df["ClassId"].astype(int)
        return X, y, model_df

    raise ValueError("Unsupported task. Use 'metadata' or 'robustness'.")


def get_driving_dataset_summary(task="metadata", data_dir="data", max_samples=5000):
    X, y, _ = load_driving_dataset(task=task, data_dir=data_dir, max_samples=max_samples)
    return {
        "task": task,
        "samples": int(len(X)),
        "features": int(X.shape[1]),
        "classes": int(y.nunique()),
    }


def _write_artifacts(y_test, y_pred, algorithm, task, report_dir="Public"):
    _safe_mkdir(report_dir)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    base = f"driving_{task}_{algorithm.replace(' ', '_')}_{stamp}"

    report_text = classification_report(y_test, y_pred, zero_division=0)
    report_path = os.path.join(report_dir, f"{base}_classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as handle:
        handle.write(report_text)

    matrix_path = os.path.join(report_dir, f"{base}_confusion_matrix.csv")
    matrix = confusion_matrix(y_test, y_pred)
    pd.DataFrame(matrix).to_csv(matrix_path, index=False)

    matrix_img_path = os.path.join(report_dir, f"{base}_confusion_matrix.png")
    fig = plt.figure(figsize=(10, 8))
    sns.heatmap(matrix, cmap="Blues", cbar=True)
    plt.title(f"Confusion Matrix: {task} | {algorithm}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    fig.tight_layout()
    fig.savefig(matrix_img_path)
    plt.close(fig)

    grouped = pd.DataFrame({"true": y_test, "pred": y_pred})
    grouped["true_family"] = grouped["true"].apply(_get_sign_family)
    grouped["pred_family"] = grouped["pred"].apply(_get_sign_family)
    grouped["is_correct"] = grouped["true"] == grouped["pred"]

    grouped_summary = (
        grouped.groupby("true_family", dropna=False)["is_correct"]
        .agg(["count", "sum"])
        .rename(columns={"sum": "correct"})
        .reset_index()
    )
    grouped_summary["error_count"] = grouped_summary["count"] - grouped_summary["correct"]
    grouped_summary["error_rate"] = 1 - (grouped_summary["correct"] / grouped_summary["count"])

    grouped_path = os.path.join(report_dir, f"{base}_grouped_errors.csv")
    grouped_summary.to_csv(grouped_path, index=False)

    return {
        "classification_report_path": report_path,
        "confusion_matrix_csv_path": matrix_path,
        "confusion_matrix_png_path": matrix_img_path,
        "grouped_error_report_path": grouped_path,
    }


def train_driving_model(task="metadata", algorithm="Decision Tree", data_dir="data", test_size=0.2, random_state=42, max_samples=5000):
    """
    Train and evaluate a driving-oriented tabular model.
    Returns metrics, runtime, and artifact paths.
    """
    t0 = time.perf_counter()
    X, y, _ = load_driving_dataset(task=task, data_dir=data_dir, max_samples=max_samples)
    class_counts = y.value_counts()
    use_stratify = None if class_counts.min() < 2 else y
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        random_state=random_state,
        stratify=use_stratify,
    )

    model = _algorithm_factory(algorithm)

    train_t0 = time.perf_counter()
    model.fit(X_train, y_train)
    train_seconds = time.perf_counter() - train_t0

    infer_t0 = time.perf_counter()
    y_pred = model.predict(X_test)
    infer_seconds = time.perf_counter() - infer_t0

    accuracy = float(accuracy_score(y_test, y_pred))
    precision_macro, recall_macro, f1_macro, _ = precision_recall_fscore_support(
        y_test, y_pred, average="macro", zero_division=0
    )
    precision_weighted, recall_weighted, f1_weighted, _ = precision_recall_fscore_support(
        y_test, y_pred, average="weighted", zero_division=0
    )

    artifacts = _write_artifacts(y_test, y_pred, algorithm=algorithm, task=task)

    total_seconds = time.perf_counter() - t0
    sample_count = len(X_test)
    ms_per_sample = (infer_seconds / max(sample_count, 1)) * 1000.0

    return {
        "task": task,
        "algorithm": algorithm,
        "samples": int(len(X)),
        "features": int(X.shape[1]),
        "classes": int(y.nunique()),
        "accuracy": accuracy,
        "precision_macro": float(precision_macro),
        "recall_macro": float(recall_macro),
        "f1_macro": float(f1_macro),
        "precision_weighted": float(precision_weighted),
        "recall_weighted": float(recall_weighted),
        "f1_weighted": float(f1_weighted),
        "train_seconds": float(train_seconds),
        "infer_seconds": float(infer_seconds),
        "ms_per_sample": float(ms_per_sample),
        "total_seconds": float(total_seconds),
        **artifacts,
    }


def train_model(df, target_column, algorithm):
    """
    Legacy generic trainer retained for optional advanced CSV sandbox.
    """
    df = df.dropna(subset=[target_column])
    X = df.drop(columns=[target_column]).copy()
    y = df[target_column].copy()

    X = X.fillna(0)
    for col in X.select_dtypes(include=["object"]).columns:
        X[col] = LabelEncoder().fit_transform(X[col].astype(str))

    if y.dtype == "object":
        y = LabelEncoder().fit_transform(y.astype(str))

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = _algorithm_factory(algorithm)
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    return float(accuracy_score(y_test, preds))