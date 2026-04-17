# DS 4320 Project 2: Predicting Air Quality Using PM2.5 Measurements
**Author:** Trevor Xu  
**NetID:** pxg6af  
**Data Source:** EPA AQS Parameter 88502 — Acceptable PM2.5 AQI & Speciation Mass (2023)  
**Database:** MongoDB Atlas — `air_quality_db.pm25_daily`

---

## Pipeline Overview
This notebook implements a full end-to-end pipeline to solve the specific problem:
> *Can we predict the AQI category (Good / Moderate / Unhealthy / etc.) for a U.S. monitoring site using historical PM2.5 measurements and temporal features?*

**Steps:**
1. Setup — logging, imports, MongoDB connection
2. Data preparation — query MongoDB into a DataFrame
3. Feature engineering — temporal + rolling features
4. Model training — Random Forest classifier
5. Evaluation — accuracy, classification report, confusion matrix
6. Visualization — publication-quality figures

## 1. Setup


```python
# ── Imports ───────────────────────────────────────────────────────────────
import logging
import os
import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns

from pymongo import MongoClient
from urllib.parse import quote_plus

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix
)

# ── Logging setup — writes to logs/pipeline.log AND notebook output ───────
os.makedirs('logs', exist_ok=True)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.FileHandler('logs/pipeline.log'),   # persists to disk
        logging.StreamHandler()                      # shows in notebook
    ]
)
log = logging.getLogger(__name__)
log.info('Pipeline started — imports and logging ready.')
```

    2026-04-17 15:56:06,592 [INFO] Pipeline started — imports and logging ready.


## 2. Data Preparation — Query MongoDB into a DataFrame

**Rationale:** We query all documents from the `pm25_daily` collection and flatten the nested subdocuments (`location`, `air_quality`) into a flat DataFrame. Querying from MongoDB rather than re-reading the CSV ensures the pipeline always uses the cleaned, validated data that was ingested — not a raw local file that may differ.


```python
def connect_to_mongo(uri: str, db_name: str, col_name: str):
    """
    Connect to MongoDB Atlas and return the target collection.
    Raises ConnectionError if the ping fails within 15 seconds.
    """
    try:
        client = MongoClient(uri, serverSelectionTimeoutMS=15_000)
        client.admin.command('ping')   # verify connection is live
        log.info('Connected to MongoDB Atlas.')
        return client[db_name][col_name]
    except Exception as exc:
        log.error('MongoDB connection failed: %s', exc)
        raise ConnectionError(f'Could not connect to MongoDB: {exc}') from exc


def query_to_dataframe(col) -> pd.DataFrame:
    """
    Query all documents from the MongoDB collection and flatten nested
    subdocuments (location, air_quality) into a single-level DataFrame.
    Returns a cleaned DataFrame ready for feature engineering.
    """
    log.info('Querying MongoDB collection...')

    try:
        # Project only the fields we need — avoids pulling large unused fields
        cursor = col.find(
            {},   # no filter — get all documents
            {
                '_id': 0,
                'date': 1,
                'year': 1,
                'month': 1,
                'day_of_week': 1,
                'location.state': 1,
                'location.city': 1,
                'location.latitude': 1,
                'location.longitude': 1,
                'air_quality.pm25_mean': 1,
                'air_quality.pm25_max': 1,
                'air_quality.aqi': 1,
                'air_quality.aqi_category': 1,
            }
        )
        records = list(cursor)
        log.info('Retrieved %d documents from MongoDB.', len(records))
    except Exception as exc:
        log.error('Query failed: %s', exc)
        raise

    # Flatten nested dicts into top-level columns
    rows = []
    for doc in records:
        loc = doc.get('location', {})
        aq  = doc.get('air_quality', {})
        rows.append({
            'date':         doc.get('date'),
            'year':         doc.get('year'),
            'month':        doc.get('month'),
            'day_of_week':  doc.get('day_of_week'),
            'state':        loc.get('state'),
            'city':         loc.get('city'),
            'latitude':     loc.get('latitude'),
            'longitude':    loc.get('longitude'),
            'pm25_mean':    aq.get('pm25_mean'),
            'pm25_max':     aq.get('pm25_max'),
            'aqi':          aq.get('aqi'),
            'aqi_category': aq.get('aqi_category'),
        })

    df = pd.DataFrame(rows)

    # Coerce types
    df['date']      = pd.to_datetime(df['date'])
    df['pm25_mean'] = pd.to_numeric(df['pm25_mean'], errors='coerce')
    df['pm25_max']  = pd.to_numeric(df['pm25_max'],  errors='coerce')
    df['aqi']       = pd.to_numeric(df['aqi'],       errors='coerce')

    # Drop rows where the target label is missing
    before = len(df)
    df = df.dropna(subset=['aqi_category', 'pm25_mean'])
    log.info('Dropped %d rows with missing target/features. Remaining: %d',
             before - len(df), len(df))

    return df


# ── Run the query ──────────────────────────────────────────────────────────
USERNAME = quote_plus('')
PASSWORD = quote_plus('')
MONGO_URI = f'mongodb+srv://{USERNAME}:{PASSWORD}@nosqlfinalproject.vkq6zwb.mongodb.net/?appName=NoSQLFinalProject'

col = connect_to_mongo(MONGO_URI, 'air_quality_db', 'pm25_daily')
df  = query_to_dataframe(col)

print(f'\nDataFrame shape: {df.shape}')
df.head()
```

    2026-04-17 16:00:56,242 [INFO] Connected to MongoDB Atlas.
    2026-04-17 16:00:56,243 [INFO] Querying MongoDB collection...
    2026-04-17 16:00:57,627 [INFO] Retrieved 96242 documents from MongoDB.
    2026-04-17 16:00:58,343 [INFO] Dropped 0 rows with missing target/features. Remaining: 96242


    
    DataFrame shape: (96242, 12)





<div>
<style scoped>
    .dataframe tbody tr th:only-of-type {
        vertical-align: middle;
    }

    .dataframe tbody tr th {
        vertical-align: top;
    }

    .dataframe thead th {
        text-align: right;
    }
</style>
<table border="1" class="dataframe">
  <thead>
    <tr style="text-align: right;">
      <th></th>
      <th>date</th>
      <th>year</th>
      <th>month</th>
      <th>day_of_week</th>
      <th>state</th>
      <th>city</th>
      <th>latitude</th>
      <th>longitude</th>
      <th>pm25_mean</th>
      <th>pm25_max</th>
      <th>aqi</th>
      <th>aqi_category</th>
    </tr>
  </thead>
  <tbody>
    <tr>
      <th>0</th>
      <td>2023-02-08</td>
      <td>2023</td>
      <td>2</td>
      <td>Wednesday</td>
      <td>Alabama</td>
      <td>Birmingham</td>
      <td>33.553056</td>
      <td>-86.815</td>
      <td>10.5</td>
      <td>10.5</td>
      <td>54</td>
      <td>Moderate</td>
    </tr>
    <tr>
      <th>1</th>
      <td>2023-05-03</td>
      <td>2023</td>
      <td>5</td>
      <td>Wednesday</td>
      <td>Alabama</td>
      <td>Birmingham</td>
      <td>33.553056</td>
      <td>-86.815</td>
      <td>8.4</td>
      <td>8.4</td>
      <td>47</td>
      <td>Good</td>
    </tr>
    <tr>
      <th>2</th>
      <td>2023-01-12</td>
      <td>2023</td>
      <td>1</td>
      <td>Thursday</td>
      <td>Alabama</td>
      <td>Birmingham</td>
      <td>33.553056</td>
      <td>-86.815</td>
      <td>9.3</td>
      <td>9.3</td>
      <td>51</td>
      <td>Moderate</td>
    </tr>
    <tr>
      <th>3</th>
      <td>2023-02-06</td>
      <td>2023</td>
      <td>2</td>
      <td>Monday</td>
      <td>Alabama</td>
      <td>Birmingham</td>
      <td>33.553056</td>
      <td>-86.815</td>
      <td>16.4</td>
      <td>16.4</td>
      <td>65</td>
      <td>Moderate</td>
    </tr>
    <tr>
      <th>4</th>
      <td>2023-02-20</td>
      <td>2023</td>
      <td>2</td>
      <td>Monday</td>
      <td>Alabama</td>
      <td>Birmingham</td>
      <td>33.553056</td>
      <td>-86.815</td>
      <td>19.6</td>
      <td>19.6</td>
      <td>71</td>
      <td>Moderate</td>
    </tr>
  </tbody>
</table>
</div>




```python
# ── Basic data summary ─────────────────────────────────────────────────────
log.info('Generating data summary.')

print('=== AQI Category Distribution ===')
print(df['aqi_category'].value_counts())
print()
print('=== PM2.5 Mean Stats ===')
print(df['pm25_mean'].describe().round(2))
print()
print('=== States in dataset ===')
print(df['state'].value_counts())
```

    2026-04-17 16:01:08,441 [INFO] Generating data summary.


    === AQI Category Distribution ===
    aqi_category
    Good                              73120
    Moderate                          22357
    Unhealthy for Sensitive Groups      396
    Unhealthy                           268
    Very Unhealthy                       94
    Hazardous                             7
    Name: count, dtype: int64
    
    === PM2.5 Mean Stats ===
    count    96242.0
    mean         7.2
    std          8.5
    min          0.0
    25%          3.3
    50%          5.6
    75%          8.9
    max        299.0
    Name: pm25_mean, dtype: float64
    
    === States in dataset ===
    state
    Washington        15684
    Oregon            14829
    California        11720
    Idaho              7608
    Florida            6918
    New York           5918
    Iowa               3528
    Arizona            2864
    Texas              2311
    Ohio               2301
    Montana            2282
    Louisiana          2129
    Georgia            1896
    Illinois           1813
    Colorado           1660
    Alabama            1492
    Oklahoma           1134
    New Mexico          937
    Tennessee           934
    Maine               915
    Alaska              692
    North Carolina      681
    Utah                576
    Wyoming             468
    Puerto Rico         430
    Virginia            418
    New Hampshire       353
    Minnesota           337
    Vermont             310
    Michigan            239
    Nevada              239
    Kansas              237
    North Dakota        234
    Massachusetts       230
    South Dakota        230
    Missouri            215
    Arkansas            212
    Hawaii              212
    Pennsylvania        121
    South Carolina      120
    West Virginia       120
    New Jersey          119
    Maryland            118
    Connecticut         117
    Virgin Islands      116
    Wisconsin           115
    Kentucky            110
    Name: count, dtype: int64


## 3. Feature Engineering

**Rationale:** The raw dataset contains daily PM2.5 means per monitoring site. To predict *next-day* AQI category, we engineer features that capture:
- **Temporal patterns** — month and day-of-week encode seasonal and weekly cycles in human activity and weather.
- **Rolling statistics** — 7-day rolling mean and standard deviation capture short-term pollution trends and volatility at each site. A rising rolling mean suggests worsening conditions.
- **State encoding** — state captures regional climate, industrial activity, and regulatory stringency.

We use `pm25_mean` shifted by 1 day as the primary predictor — this reflects what we would realistically know *before* the prediction day.


```python
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Engineer temporal and rolling features for the classification model.

    Features created:
      - pm25_lag1:    PM2.5 mean from the previous day (same city)
      - pm25_roll7:   7-day rolling mean of PM2.5 (same city)
      - pm25_roll7sd: 7-day rolling standard deviation of PM2.5 (same city)
      - month:        calendar month (1-12, captures seasonality)
      - day_of_week_enc: day of week encoded as integer (0=Monday)
      - state_enc:    label-encoded state
    """
    log.info('Engineering features...')

    # Sort by city and date so rolling calculations are correct
    df = df.sort_values(['city', 'date']).copy()

    # Lag feature: PM2.5 from the previous day at the same city
    # Rationale: yesterday's pollution strongly predicts today's AQI
    df['pm25_lag1'] = df.groupby('city')['pm25_mean'].shift(1)

    # Rolling mean over the past 7 days — captures sustained pollution events
    df['pm25_roll7'] = (
        df.groupby('city')['pm25_mean']
        .transform(lambda x: x.shift(1).rolling(7, min_periods=3).mean())
    )

    # Rolling std over past 7 days — captures volatility (e.g. wildfire onset)
    df['pm25_roll7sd'] = (
        df.groupby('city')['pm25_mean']
        .transform(lambda x: x.shift(1).rolling(7, min_periods=3).std())
    )

    # Encode day of week as integer (Monday=0 ... Sunday=6)
    dow_map = {'Monday':0,'Tuesday':1,'Wednesday':2,'Thursday':3,
               'Friday':4,'Saturday':5,'Sunday':6}
    df['day_of_week_enc'] = df['day_of_week'].map(dow_map)

    # Label-encode state — converts state names to integers for the model
    le_state = LabelEncoder()
    df['state_enc'] = le_state.fit_transform(df['state'].astype(str))

    # Drop rows where lag/rolling features are NaN (first days of each city)
    before = len(df)
    df = df.dropna(subset=['pm25_lag1', 'pm25_roll7', 'pm25_roll7sd'])
    log.info('Feature engineering complete. Dropped %d rows lacking lag/rolling data. Remaining: %d',
             before - len(df), len(df))

    return df, le_state


df_feat, le_state = engineer_features(df)

print(f'Feature-engineered DataFrame shape: {df_feat.shape}')
print(df_feat[['date','city','pm25_mean','pm25_lag1','pm25_roll7','pm25_roll7sd','aqi_category']].head(10))
```

    2026-04-17 16:01:15,091 [INFO] Engineering features...
    2026-04-17 16:01:15,409 [INFO] Feature engineering complete. Dropped 672 rows lacking lag/rolling data. Remaining: 95570


    Feature-engineered DataFrame shape: (95570, 17)
                date      city  pm25_mean  pm25_lag1  pm25_roll7  pm25_roll7sd  \
    86027 2023-01-04  Aberdeen        2.6        3.6    4.800000      1.044031   
    86251 2023-01-05  Aberdeen        3.1        2.6    4.250000      1.391642   
    86339 2023-01-06  Aberdeen        3.5        3.1    4.020000      1.310343   
    86406 2023-01-07  Aberdeen        2.9        3.5    3.933333      1.191078   
    86275 2023-01-08  Aberdeen        3.4        2.9    3.785714      1.155319   
    86028 2023-01-09  Aberdeen        2.9        3.4    3.485714      0.874507   
    86213 2023-01-10  Aberdeen        3.4        2.9    3.142857      0.369040   
    86468 2023-01-11  Aberdeen        2.7        3.4    3.114286      0.333809   
    86377 2023-01-12  Aberdeen        2.8        2.7    3.128571      0.309377   
    86214 2023-01-13  Aberdeen        2.7        2.8    3.085714      0.333809   
    
          aqi_category  
    86027         Good  
    86251         Good  
    86339         Good  
    86406         Good  
    86275         Good  
    86028         Good  
    86213         Good  
    86468         Good  
    86377         Good  
    86214         Good  


## 4. Model Training — Random Forest Classifier

**Rationale:** We use a **Random Forest classifier**, an ensemble method covered in DS 3021/4021. Random Forest is well-suited for this task because:
- It handles non-linear relationships between PM2.5 history and AQI category naturally.
- It is robust to outliers in PM2.5 readings (e.g., wildfire days).
- Feature importance scores provide interpretable insight into which predictors drive the model.
- It performs well with moderate-sized tabular datasets without extensive hyperparameter tuning.

We use an 80/20 train/test split stratified by AQI category to ensure all classes are represented in both sets, which is important given the class imbalance ("Good" dominates).


```python
def train_model(df: pd.DataFrame):
    """
    Train a Random Forest classifier to predict AQI category.

    Returns the trained model, test features, test labels,
    and predictions for downstream evaluation.
    """
    # Define feature columns and target
    FEATURES = ['pm25_lag1', 'pm25_roll7', 'pm25_roll7sd',
                'month', 'day_of_week_enc', 'state_enc']
    TARGET = 'aqi_category'

    X = df[FEATURES]
    y = df[TARGET]

    log.info('Class distribution in target:\n%s', y.value_counts().to_string())

    # Encode target labels to integers for sklearn
    le_target = LabelEncoder()
    y_enc = le_target.fit_transform(y)

    # 80/20 stratified split — ensures rare classes (Unhealthy, Hazardous)
    # appear in both train and test sets
    try:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_enc,
            test_size=0.2,
            random_state=42,
            stratify=y_enc
        )
    except Exception as exc:
        log.error('Train/test split failed: %s', exc)
        raise

    log.info('Train size: %d | Test size: %d', len(X_train), len(X_test))

    # Random Forest — 200 trees, max depth 12
    # n_estimators=200: enough trees for stable predictions without excess compute
    # max_depth=12: limits overfitting while allowing complex decision boundaries
    # class_weight='balanced': compensates for the heavy class imbalance
    #   ("Good" days far outnumber "Hazardous" days)
    clf = RandomForestClassifier(
        n_estimators=200,
        max_depth=12,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1   # use all CPU cores
    )

    log.info('Training Random Forest...')
    try:
        clf.fit(X_train, y_train)
    except Exception as exc:
        log.error('Model training failed: %s', exc)
        raise

    log.info('Training complete.')
    return clf, X_train, X_test, y_train, y_test, le_target, FEATURES


clf, X_train, X_test, y_train, y_test, le_target, FEATURES = train_model(df_feat)
```

    2026-04-17 16:01:20,909 [INFO] Class distribution in target:
    aqi_category
    Good                              72640
    Moderate                          22169
    Unhealthy for Sensitive Groups      394
    Unhealthy                           267
    Very Unhealthy                       94
    Hazardous                             6
    2026-04-17 16:01:20,996 [INFO] Train size: 76456 | Test size: 19114
    2026-04-17 16:01:20,998 [INFO] Training Random Forest...
    2026-04-17 16:01:33,013 [INFO] Training complete.


## 5. Model Evaluation


```python
def evaluate_model(clf, X_test, y_test, le_target):
    """
    Evaluate the trained classifier on the held-out test set.
    Logs accuracy and prints a full classification report.
    Returns predictions and class names for visualization.
    """
    try:
        y_pred = clf.predict(X_test)
    except Exception as exc:
        log.error('Prediction failed: %s', exc)
        raise

    acc = accuracy_score(y_test, y_pred)
    log.info('Test accuracy: %.4f', acc)

    class_names = le_target.classes_

    print(f'Test Accuracy: {acc:.4f}\n')
    print('Classification Report:')
    print(classification_report(y_test, y_pred, target_names=class_names))

    return y_pred, class_names


y_pred, class_names = evaluate_model(clf, X_test, y_test, le_target)
```

    2026-04-17 16:01:33,315 [INFO] Test accuracy: 0.7939


    Test Accuracy: 0.7939
    
    Classification Report:
                                    precision    recall  f1-score   support
    
                              Good       0.92      0.81      0.86     14528
                         Hazardous       0.00      0.00      0.00         1
                          Moderate       0.55      0.74      0.63      4434
                         Unhealthy       0.38      0.30      0.34        53
    Unhealthy for Sensitive Groups       0.16      0.29      0.21        79
                    Very Unhealthy       0.47      0.37      0.41        19
    
                          accuracy                           0.79     19114
                         macro avg       0.41      0.42      0.41     19114
                      weighted avg       0.83      0.79      0.80     19114
    


## 6. Visualization

**Rationale:** Three publication-quality figures are produced:
1. **Confusion Matrix** — shows exactly where the model makes mistakes. This is the primary diagnostic for a multi-class classifier.
2. **Feature Importance** — reveals which predictors drive the model, supporting the analytical narrative.
3. **PM2.5 Distribution by State** — contextualizes the prediction problem by showing the geographic variation in air quality that the model must learn.

All figures use a consistent color palette, labeled axes, and are sized for journal inclusion (300 dpi).


```python
def plot_confusion_matrix(y_test, y_pred, class_names):
    """
    Plot a normalized confusion matrix heatmap.
    Normalization (row-wise) shows recall per class, which is more
    informative than raw counts when class sizes differ greatly.
    """
    log.info('Plotting confusion matrix.')

    cm = confusion_matrix(y_test, y_pred)
    # Normalize row-wise so each cell shows fraction of true-class predictions
    cm_norm = cm.astype(float) / cm.sum(axis=1, keepdims=True)

    fig, ax = plt.subplots(figsize=(9, 7))

    sns.heatmap(
        cm_norm,
        annot=True,
        fmt='.2f',
        cmap='Blues',
        xticklabels=class_names,
        yticklabels=class_names,
        linewidths=0.5,
        ax=ax
    )

    ax.set_title('Normalized Confusion Matrix — AQI Category Prediction',
                 fontsize=14, fontweight='bold', pad=15)
    ax.set_xlabel('Predicted Category', fontsize=12, labelpad=10)
    ax.set_ylabel('True Category', fontsize=12, labelpad=10)
    ax.tick_params(axis='x', rotation=30)
    ax.tick_params(axis='y', rotation=0)

    plt.tight_layout()
    plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
    log.info('Saved confusion_matrix.png')
    plt.show()


plot_confusion_matrix(y_test, y_pred, class_names)
```

    2026-04-17 16:01:33,337 [INFO] Plotting confusion matrix.
    2026-04-17 16:01:33,915 [INFO] Saved confusion_matrix.png



    
![png](pipeline_files/pipeline_13_1.png)
    



```python
def plot_feature_importance(clf, feature_names):
    """
    Plot a horizontal bar chart of Random Forest feature importances.
    Mean Decrease in Impurity (MDI) scores show each feature's contribution
    to reducing uncertainty across all 200 trees.
    """
    log.info('Plotting feature importance.')

    importances = clf.feature_importances_
    # Sort features by importance (descending) for a clean ranked display
    idx = np.argsort(importances)[::-1]
    sorted_names  = [feature_names[i] for i in idx]
    sorted_values = importances[idx]

    # Human-readable feature labels
    label_map = {
        'pm25_lag1':      'PM2.5 (previous day)',
        'pm25_roll7':     'PM2.5 7-day rolling mean',
        'pm25_roll7sd':   'PM2.5 7-day rolling std dev',
        'month':          'Month of year',
        'day_of_week_enc':'Day of week',
        'state_enc':      'State'
    }
    labels = [label_map.get(n, n) for n in sorted_names]

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = sns.color_palette('viridis', len(labels))
    bars = ax.barh(labels[::-1], sorted_values[::-1], color=colors[::-1], edgecolor='white')

    # Annotate each bar with its importance value
    for bar, val in zip(bars, sorted_values[::-1]):
        ax.text(bar.get_width() + 0.002, bar.get_y() + bar.get_height() / 2,
                f'{val:.3f}', va='center', fontsize=10)

    ax.set_title('Feature Importance — Random Forest (Mean Decrease in Impurity)',
                 fontsize=13, fontweight='bold', pad=12)
    ax.set_xlabel('Importance Score', fontsize=11)
    ax.set_xlim(0, sorted_values.max() + 0.06)
    ax.spines[['top', 'right']].set_visible(False)

    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    log.info('Saved feature_importance.png')
    plt.show()


plot_feature_importance(clf, FEATURES)
```

    2026-04-17 16:01:37,698 [INFO] Plotting feature importance.
    2026-04-17 16:01:38,035 [INFO] Saved feature_importance.png



    
![png](pipeline_files/pipeline_14_1.png)
    



```python
def plot_pm25_by_state(df: pd.DataFrame):
    """
    Plot a boxplot of daily PM2.5 mean concentrations by state.
    States are ordered by median PM2.5 (ascending) to reveal geographic
    air quality gradients. This directly contextualizes the prediction
    problem: states with higher and more variable PM2.5 are harder to
    classify correctly.
    """
    log.info('Plotting PM2.5 distribution by state.')

    # Order states by median PM2.5
    order = (
        df.groupby('state')['pm25_mean']
        .median()
        .sort_values()
        .index.tolist()
    )

    fig, ax = plt.subplots(figsize=(12, 6))

    sns.boxplot(
        data=df,
        x='state',
        y='pm25_mean',
        order=order,
        palette='coolwarm',
        flierprops=dict(marker='o', markersize=2, alpha=0.3),
        linewidth=0.8,
        ax=ax
    )

    # WHO 24-hour PM2.5 guideline: 15 µg/m³
    ax.axhline(15, color='red', linestyle='--', linewidth=1.2, label='WHO guideline (15 µg/m³)')

    ax.set_title('Daily PM2.5 Concentration by State — EPA AQS 2023',
                 fontsize=14, fontweight='bold', pad=12)
    ax.set_xlabel('State', fontsize=12, labelpad=8)
    ax.set_ylabel('PM2.5 Mean (µg/m³)', fontsize=12, labelpad=8)
    ax.tick_params(axis='x', rotation=40)
    ax.legend(fontsize=10)
    ax.spines[['top', 'right']].set_visible(False)
    ax.yaxis.set_minor_locator(mticker.AutoMinorLocator())

    plt.tight_layout()
    plt.savefig('pm25_by_state.png', dpi=300, bbox_inches='tight')
    log.info('Saved pm25_by_state.png')
    plt.show()


plot_pm25_by_state(df_feat)
```

    2026-04-17 16:01:42,133 [INFO] Plotting PM2.5 distribution by state.
    2026-04-17 16:01:43,737 [INFO] Saved pm25_by_state.png



    
![png](pipeline_files/pipeline_15_1.png)
    


## 7. Summary and Conclusions

### Does the Pipeline Solve the Problem?

Yes. The Random Forest classifier achieves **79.4% accuracy** on the held-out test set, successfully predicting next-day AQI categories from MongoDB-stored PM2.5 measurements using only historical pollution data and temporal features. This demonstrates that the document-oriented database design, feature engineering strategy, and machine learning pipeline together produce a working end-to-end solution to the specific problem defined at the start of this project.

### Key Findings from the Model

**1. Historical PM2.5 dominates prediction.** The top three features — previous day's PM2.5 (0.278), 7-day rolling standard deviation (0.245), and 7-day rolling mean (0.214) — together account for **74% of the model's predictive power**. This confirms the core hypothesis that air quality exhibits strong temporal autocorrelation and that recent pollution history is the best predictor of near-term conditions.

**2. Volatility matters as much as level.** The 7-day rolling *standard deviation* (0.245) outranks the 7-day rolling *mean* (0.214) in importance. This is an informative finding: the model learns that **unstable pollution patterns** — indicative of active wildfires, weather front passages, or episodic industrial events — are a stronger signal of category change than a simple level average. A week with stable moderate readings behaves differently from a week with large day-to-day swings at the same mean.

**3. Location and calendar contribute meaningfully but secondarily.** State (0.113), day of week (0.082), and month (0.068) together contribute 26% of importance. State captures regional climate and pollution-source profiles; temporal features capture seasonality such as summer wildfire activity in Western states and winter inversion events.

### Class-Level Performance

| Class | Precision | Recall | F1 | Support |
|-------|-----------|--------|------|---------|
| Good | 0.92 | 0.81 | 0.86 | 14,528 |
| Moderate | 0.55 | 0.74 | 0.63 | 4,434 |
| Unhealthy for Sensitive Groups | 0.16 | 0.29 | 0.21 | 79 |
| Unhealthy | 0.38 | 0.30 | 0.34 | 53 |
| Very Unhealthy | 0.47 | 0.37 | 0.41 | 19 |
| Hazardous | 0.00 | 0.00 | 0.00 | 1 |
| **Weighted avg** | **0.83** | **0.79** | **0.80** | **19,114** |

The model performs strongly on the two dominant classes ("Good" and "Moderate"), which together make up over 99% of observations. Performance degrades on rare unhealthy categories — a well-known consequence of severe class imbalance, where the "Hazardous" class has only 1 test sample and cannot be meaningfully evaluated. Using `class_weight='balanced'` during training helped the model attempt non-trivial recall on rare classes (e.g., 37% recall on "Very Unhealthy") that would otherwise be ignored entirely in favor of the majority class.

### Limitations

- **No meteorological features.** Wind speed, humidity, and temperature are known drivers of PM2.5 dispersion. Adding NOAA weather data would likely improve prediction on high-pollution days.
- **Single-year training data.** The 2023 snapshot includes certain wildfire events but may not generalize to years with unusually severe or mild seasons.
- **Class imbalance for rare categories.** Extreme pollution events ("Hazardous", "Very Unhealthy") are so rare that evaluating the model on them is statistically unreliable. Collecting multi-year data or oversampling rare events would be needed for robust performance on these categories.
- **Rural coverage gaps.** EPA monitoring sites concentrate in urban/suburban areas, so predictions for rural regions rely on sparse nearby sites.

### Conclusion

The pipeline demonstrates that historical PM2.5 measurements, when stored in a well-structured MongoDB collection and combined with simple temporal features, can be used to train a machine learning model that correctly predicts next-day AQI category **about 4 out of every 5 days** across major U.S. cities. The finding that volatility (rolling standard deviation) is among the top predictors is practically meaningful: it suggests real-world early-warning systems should monitor not only average concentrations but also day-to-day variability as a leading indicator of deteriorating air quality. This directly supports the project's original motivation of giving vulnerable communities — the elderly, children, and those with respiratory conditions — actionable advance warning to plan their daily activities.


```python
# ── Final log entry ────────────────────────────────────────────────────────
log.info('Pipeline complete. Figures saved: confusion_matrix.png, '
         'feature_importance.png, pm25_by_state.png')
print('Pipeline complete. Check logs/pipeline.log for full run details.')
```

    2026-04-17 16:01:52,883 [INFO] Pipeline complete. Figures saved: confusion_matrix.png, feature_importance.png, pm25_by_state.png


    Pipeline complete. Check logs/pipeline.log for full run details.

