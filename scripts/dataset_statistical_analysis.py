from pathlib import Path
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


# ============================================================
# 1. DATASET PATH
# ============================================================

# Change this path to your dataset location
DATASET_DIR = Path(r"B:\fall_detection_system\data\dataset")
# Dataset splits
SPLITS = ["train", "valid", "test"]


# ============================================================
# 2. READ YOLO LABEL FILES
# ============================================================

records = []

for split in SPLITS:

    labels_dir = DATASET_DIR / split / "labels"

    if not labels_dir.exists():
        print(f"WARNING: Labels folder not found: {labels_dir}")
        continue

    label_files = list(labels_dir.glob("*.txt"))

    print(f"\n{split.upper()} DATASET")
    print("-" * 50)
    print(f"Label files found: {len(label_files)}")

    for label_file in label_files:

        with open(label_file, "r") as file:

            for line_number, line in enumerate(file, start=1):

                line = line.strip()

                # Ignore empty lines
                if not line:
                    continue

                values = line.split()

                # YOLO format must contain 5 values
                if len(values) != 5:
                    print(
                        f"WARNING: Invalid annotation in "
                        f"{label_file.name}, line {line_number}"
                    )
                    continue

                try:
                    class_id = int(values[0])

                    x_center = float(values[1])
                    y_center = float(values[2])
                    width = float(values[3])
                    height = float(values[4])

                except ValueError:
                    print(
                        f"WARNING: Invalid numeric value in "
                        f"{label_file.name}"
                    )
                    continue

                # ------------------------------------------------
                # Calculate additional features
                # ------------------------------------------------

                area = width * height

                if height != 0:
                    aspect_ratio = width / height
                else:
                    aspect_ratio = np.nan

                records.append({
                    "split": split,
                    "image": label_file.stem,
                    "class_id": class_id,
                    "x_center": x_center,
                    "y_center": y_center,
                    "width": width,
                    "height": height,
                    "area": area,
                    "aspect_ratio": aspect_ratio
                })


# ============================================================
# 3. CREATE DATAFRAME
# ============================================================

df = pd.DataFrame(records)

if df.empty:
    print("\nNo valid YOLO annotations were found.")
    print("Check your DATASET_DIR path and folder structure.")
    exit()


print("\n")
print("=" * 70)
print("COMPLETE DATASET STATISTICAL ANALYSIS")
print("=" * 70)


# ============================================================
# 4. BASIC DATASET INFORMATION
# ============================================================

print("\n1. DATASET SIZE")
print("-" * 50)

print(f"Total annotations : {len(df)}")
print(f"Unique images     : {df['image'].nunique()}")


# ============================================================
# 5. SPLIT DISTRIBUTION
# ============================================================

print("\n2. DATA SPLIT DISTRIBUTION")
print("-" * 50)

split_counts = df["split"].value_counts()

print(split_counts)

print("\nPercentage:")

split_percent = (
    df["split"].value_counts(normalize=True) * 100
)

print(split_percent.round(2))


# ============================================================
# 6. CLASS DISTRIBUTION
# ============================================================

print("\n3. CLASS DISTRIBUTION")
print("-" * 50)

class_counts = df["class_id"].value_counts().sort_index()

print(class_counts)

print("\nClass Percentage:")

class_percentage = (
    df["class_id"].value_counts(normalize=True)
    .sort_index() * 100
)

print(class_percentage.round(2))


# ============================================================
# 7. STATISTICAL CHARACTERISTICS
# ============================================================

features = [
    "width",
    "height",
    "area",
    "aspect_ratio"
]

print("\n4. STATISTICAL CHARACTERISTICS")
print("-" * 50)

statistics = []

for feature in features:

    values = df[feature].dropna()

    mean_value = values.mean()
    median_value = values.median()
    std_value = values.std()
    min_value = values.min()
    max_value = values.max()
    skew_value = values.skew()

    statistics.append({
        "Feature": feature,
        "Mean": mean_value,
        "Median": median_value,
        "Std_Deviation": std_value,
        "Minimum": min_value,
        "Maximum": max_value,
        "Skewness": skew_value
    })

    print(f"\n{feature.upper()}")

    print(f"Mean       : {mean_value:.4f}")
    print(f"Median     : {median_value:.4f}")
    print(f"Std Dev    : {std_value:.4f}")
    print(f"Minimum    : {min_value:.4f}")
    print(f"Maximum    : {max_value:.4f}")
    print(f"Skewness   : {skew_value:.4f}")


stats_df = pd.DataFrame(statistics)


# ============================================================
# 8. OUTLIER DETECTION USING IQR
# ============================================================

print("\n5. OUTLIER ANALYSIS")
print("-" * 50)

outlier_summary = []

for feature in features:

    values = df[feature].dropna()

    Q1 = values.quantile(0.25)
    Q3 = values.quantile(0.75)

    IQR = Q3 - Q1

    lower_limit = Q1 - 1.5 * IQR
    upper_limit = Q3 + 1.5 * IQR

    outliers = values[
        (values < lower_limit) |
        (values > upper_limit)
    ]

    outlier_percentage = (
        len(outliers) / len(values)
    ) * 100

    print(f"\n{feature.upper()}")

    print(f"Q1              : {Q1:.4f}")
    print(f"Q3              : {Q3:.4f}")
    print(f"IQR             : {IQR:.4f}")
    print(f"Lower limit     : {lower_limit:.4f}")
    print(f"Upper limit     : {upper_limit:.4f}")
    print(f"Outliers        : {len(outliers)}")
    print(f"Outlier %       : {outlier_percentage:.2f}%")

    outlier_summary.append({
        "Feature": feature,
        "Q1": Q1,
        "Q3": Q3,
        "IQR": IQR,
        "Lower_Limit": lower_limit,
        "Upper_Limit": upper_limit,
        "Outlier_Count": len(outliers),
        "Outlier_Percentage": outlier_percentage
    })


outlier_df = pd.DataFrame(outlier_summary)


# ============================================================
# 9. CORRELATION ANALYSIS
# ============================================================

print("\n6. CORRELATION MATRIX")
print("-" * 50)

correlation_matrix = df[
    ["width", "height", "area", "aspect_ratio"]
].corr()

print(correlation_matrix.round(3))


# ============================================================
# 10. CLASS-WISE STATISTICS
# ============================================================

print("\n7. CLASS-WISE STATISTICS")
print("-" * 50)

class_statistics = (
    df.groupby("class_id")[
        ["width", "height", "area", "aspect_ratio"]
    ]
    .agg(["mean", "median", "std", "min", "max"])
)

print(class_statistics.round(4))


# ============================================================
# 11. SPLIT-WISE STATISTICS
# ============================================================

print("\n8. SPLIT-WISE STATISTICS")
print("-" * 50)

split_statistics = (
    df.groupby("split")[
        ["width", "height", "area", "aspect_ratio"]
    ]
    .agg(["mean", "median", "std"])
)

print(split_statistics.round(4))


# ============================================================
# 12. PROBABILITY / FREQUENCY DISTRIBUTION
# ============================================================

print("\n9. FREQUENCY DISTRIBUTION")
print("-" * 50)

# Divide aspect ratio into intervals
bins = [
    0,
    0.5,
    0.75,
    1.0,
    1.25,
    1.5,
    2.0,
    3.0,
    np.inf
]

labels = [
    "<0.5",
    "0.5-0.75",
    "0.75-1.0",
    "1.0-1.25",
    "1.25-1.5",
    "1.5-2.0",
    "2.0-3.0",
    ">3.0"
]

df["aspect_ratio_range"] = pd.cut(
    df["aspect_ratio"],
    bins=bins,
    labels=labels,
    include_lowest=True
)

aspect_distribution = (
    df["aspect_ratio_range"]
    .value_counts(sort=False)
)

aspect_probability = (
    aspect_distribution / len(df)
)

print("\nAspect Ratio Frequency:")

print(aspect_distribution)

print("\nAspect Ratio Probability:")

print(aspect_probability.round(4))


# ============================================================
# 13. SAVE RESULTS
# ============================================================

OUTPUT_DIR = DATASET_DIR / "statistical_analysis"

OUTPUT_DIR.mkdir(
    parents=True,
    exist_ok=True
)


# Save complete annotation data
df.to_csv(
    OUTPUT_DIR / "annotation_statistics.csv",
    index=False
)

# Save statistical summary
stats_df.to_csv(
    OUTPUT_DIR / "statistical_summary.csv",
    index=False
)

# Save outlier information
outlier_df.to_csv(
    OUTPUT_DIR / "outlier_analysis.csv",
    index=False
)

# Save correlation matrix
correlation_matrix.to_csv(
    OUTPUT_DIR / "correlation_matrix.csv"
)

# Save class statistics
class_statistics.to_csv(
    OUTPUT_DIR / "class_wise_statistics.csv"
)

# Save split statistics
split_statistics.to_csv(
    OUTPUT_DIR / "split_wise_statistics.csv"
)


# ============================================================
# 14. GENERATE GRAPHS
# ============================================================

print("\n10. GENERATING GRAPHS")
print("-" * 50)


# Aspect Ratio Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["aspect_ratio"].dropna(),
    bins=50
)

plt.xlabel("Bounding Box Aspect Ratio")
plt.ylabel("Frequency")
plt.title("Aspect Ratio Distribution")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "aspect_ratio_distribution.png"
)

plt.close()


# Width Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["width"],
    bins=50
)

plt.xlabel("Normalized Bounding Box Width")
plt.ylabel("Frequency")
plt.title("Bounding Box Width Distribution")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "width_distribution.png"
)

plt.close()


# Height Distribution
plt.figure(figsize=(8, 5))

plt.hist(
    df["height"],
    bins=50
)

plt.xlabel("Normalized Bounding Box Height")
plt.ylabel("Frequency")
plt.title("Bounding Box Height Distribution")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "height_distribution.png"
)

plt.close()


# Class Distribution
plt.figure(figsize=(7, 5))

class_counts.plot(
    kind="bar"
)

plt.xlabel("Class ID")
plt.ylabel("Number of Annotations")
plt.title("Class Distribution")

plt.tight_layout()

plt.savefig(
    OUTPUT_DIR / "class_distribution.png"
)

plt.close()


# ============================================================
# 15. FINAL MESSAGE
# ============================================================

print("\n")
print("=" * 70)
print("ANALYSIS COMPLETED")
print("=" * 70)

print(f"\nResults saved in:")

print(OUTPUT_DIR)

print("\nGenerated files:")

for file in OUTPUT_DIR.iterdir():

    print(" -", file.name)