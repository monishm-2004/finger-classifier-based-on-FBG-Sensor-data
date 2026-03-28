# Finger Tapping Classification Project

## Overview
This project aims to classify finger tapping events using data from Fiber Bragg Grating (FBG) sensors. The goal is to distinguish between different fingers (thumb, index, middle, ring, small) based on the unique waveform patterns generated during tapping. The notebook covers data loading, preprocessing, feature extraction, and the application of various machine learning models for classification.

## Dataset
The dataset consists of FBG sensor readings for five different fingers: small, thumb, middle, index, and ring. Each finger's data is recorded in a separate Excel file (`T15 small.xlsx`, `T16 thumb.xlsx`, `T17 middle.xlsx`, `T18 index.xlsx`, `T19 ring.xlsx`). Each file contains `time` and five `sample` columns representing sensor readings.

## Methodology
The project follows these key steps:

### 1. Data Loading and Initial Inspection
- Five Excel files are loaded into pandas DataFrames (s, t, m, i, r).
- Initial check for null values is performed.

### 2. Data Preprocessing
- **Baseline Correction and Linear Detrending**: A `clean_fbg_data` function is applied to remove baseline drift and linear trends from the raw sensor data. This involves: 
  - Dropping rows with NaN values in sample columns.
  - Subtracting the mean of readings between 5-15 seconds (baseline).
  - Applying `scipy.signal.detrend` for linear detrending.
- **Segmentation**: The `segment_finger_taps` function segments the cleaned data. Each finger's data is divided into 25 segments (5 samples x 5 tap times), each representing a 21-second window around a tap event. The segments are then padded or truncated to a fixed length of 21 data points.
- **Consolidation**: All segmented data (s2, t2, m2, i2, r2) are concatenated into a single DataFrame `dataset_125` with 21 feature columns and a 'target' column for finger type.

### 3. Feature Engineering
- **Peak Centering and Resampling to 9 Points**: The 21-point segments are further processed to focus on the peak. The maximum value in each 21-point segment is identified, and a 9-point window centered around this peak is extracted. Edge cases are handled by padding with NaNs.
- **Cubic Spline Interpolation and FWHM Calculation**: 
  - The 9-point data is interpolated using `UnivariateSpline` to create 900 smooth data points, allowing for more detailed feature extraction.
  - The Full Width at Half Maximum (FWHM) is calculated from the interpolated data, serving as a key feature representing the width of the tap signal.
- **Additional Feature Extraction**: A `extract_master_features` function is applied to the 900-point smoothed data to derive 10 physical features:
  - `P2P`: Peak-to-Peak amplitude.
  - `RMS`: Root Mean Square of the signal.
  - `Max`: Maximum value of the signal.
  - `RiseTime`: Time taken to reach the peak.
  - `FallTime`: Time taken to fall from the peak.
  - `StdDev`: Standard deviation of the signal.
  - `Skew`: Skewness of the signal distribution.
  - `Kurtosis`: Kurtosis of the signal distribution.
  - `MaxSlope`: Maximum gradient of the signal.
  - `FFT_Mean`: Mean of the absolute Fast Fourier Transform values.
- These features, along with FWHM, form the `final_df` for model training.

### 4. Model Training and Evaluation
Several classification models are evaluated, initially for 5-class classification and then focusing on a 3-class problem ('index', 'middle', 'ring') after grouping 'thumb' and 'small' into an 'other' category or excluding them.

#### Models Explored:
- **Random Forest Classifier**: Initial 5-class classification and subsequent grid search with 5-fold cross-validation.
- **Linear Discriminant Analysis (LDA)**: Applied for 4-class (index, middle, ring, other) and 3-class classification, including cross-validation and PCA for dimensionality reduction.
- **K-Nearest Neighbors (KNN)**: Optimized using GridSearchCV with 5-fold cross-validation for the 3-class problem.
- **Support Vector Machine (SVM)**: Optimized using GridSearchCV with 5-fold cross-validation for the 3-class problem.
- **Gaussian Naive Bayes (GNB)**: Optimized using GridSearchCV with 5-fold cross-validation for the 3-class problem.
- **XGBoost**: Optimized using GridSearchCV with 5-fold cross-validation for the 3-class problem.

#### Evaluation Metrics:
- Accuracy (from cross-validation and hold-out test sets).
- Classification Report (Precision, Recall, F1-score).
- AUC Score (for multi-class using 'ovr' strategy).

## Results

A visual inspection of FWHM distribution for different fingers was performed, revealing potential differences.

**Model Performance Summary (3-Class Classification only, Best CV Accuracy)**:
| Model                | Classification Type   |   Accuracy |
|:---------------------|:----------------------|-----------:|
| Random Forest        | 5-class               |      52.8  |
| LDA                  | 4-class               |      62.4  |
| LDA                  | 3-class               |      60    |
| PCA + LDA            | 3-class               |      65.33 |
| KNN                  | 3-class               |      77.33 |
| SVM                  | 3-class               |      76    |
| Gaussian Naive Bayes | 3-class               |      54.67 |
| XGBoost              | 3-class               |      72    |

**Detailed Performance Reports (Hold-out Test Sets)**:

**Best KNN Model (Best CV Accuracy: 77.33%)**:
```
              precision    recall  f1-score   support

       index       0.50      0.60      0.55         5
      middle       0.60      0.60      0.60         5
        ring       0.75      0.60      0.67         5

    accuracy                           0.60        15
   macro avg       0.62      0.60      0.60        15
weighted avg       0.62      0.60      0.60        15

AUC Score for Best KNN: 0.79
```

**Best SVM Model (Best CV Accuracy: 76.00%)**:
```
              precision    recall  f1-score   support

       index       0.50      0.60      0.55         5
      middle       0.67      0.80      0.73         5
        ring       1.00      0.60      0.75         5

    accuracy                           0.67        15
   macro avg       0.72      0.67      0.67        15
weighted avg       0.72      0.67      0.67        15

AUC Score for Best SVM: 0.66
```

## Conclusion
Among the tested models, **K-Nearest Neighbors (KNN)** demonstrated the highest cross-validation accuracy of 77.33% for the 3-class finger classification problem ('index', 'middle', 'ring'), closely followed by **Support Vector Machine (SVM)** at 76.00%. The detailed classification reports provide insights into precision, recall, and F1-scores for each class, indicating varying performance across different finger types. The use of custom feature engineering, including FWHM and other waveform characteristics, played a crucial role in achieving these classification results.
