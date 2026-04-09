# Data Science Handbook — Complete Reference
> Python 3.10+ · 20+ Sections · 50+ Algorithms · 300+ Snippets  
> A comprehensive, copy-paste-ready reference for the full data science workflow.

---

## Table of Contents

### 📥 Data Ingestion
- [Reading CSV, Excel, JSON & More](#1-reading-csv-excel-json--more)
- [SQL & Databases](#2-sql--databases)
- [Parquet, Feather, APIs & Cloud](#3-parquet-feather-apis--cloud)

### 🧹 Data Cleaning
- [Missing Values & Duplicates](#4-missing-values--duplicates)
- [Dtypes, Type Conversion & Dates](#5-dtypes-type-conversion--dates)
- [Outlier Detection & Treatment](#6-outlier-detection--treatment)
- [String Cleaning & Regex](#7-string-cleaning--regex)

### 🔍 Exploration & EDA
- [Data Exploration](#8-data-exploration)
- [Full EDA Workflow](#9-full-eda-workflow)

### ⚙️ Preprocessing
- [Scaling, Encoding & Normalization](#10-scaling-encoding--normalization)
- [Advanced Imputation & Train/Test Split](#11-advanced-imputation--traintest-split)

### 🔧 Feature Engineering
- [Creating New Features](#12-creating-new-features)
- [Feature Selection](#13-feature-selection)
- [Dimensionality Reduction](#14-dimensionality-reduction)

### 📊 Visualization
- [Matplotlib](#15-matplotlib--low-level-control)
- [Seaborn](#16-seaborn--statistical-plots)
- [Plotly](#17-plotly--interactive-charts)

### 📐 Statistics
- [Descriptive Statistics](#18-descriptive-statistics)
- [Hypothesis Testing](#19-hypothesis-testing)
- [Statistical Modeling (statsmodels)](#20-statistical-modeling-statsmodels)

### 🤖 Machine Learning
- [Sklearn API & Pipelines](#21-sklearn-api--pipelines)
- [Linear Models](#22-linear-models)
- [Tree-Based Models](#23-tree-based-models)
- [Gradient Boosting & Ensembles](#24-gradient-boosting--ensembles)
- [SVM, KNN & Naive Bayes](#25-svm-knn--naive-bayes)
- [Clustering (Unsupervised)](#26-clustering-unsupervised)
- [Anomaly Detection](#27-anomaly-detection)
- [Time Series Analysis & Forecasting](#28-time-series-analysis--forecasting)
- [NLP Basics](#29-nlp-basics)

### ✅ Evaluation
- [Metrics — Classification & Regression](#30-metrics--classification--regression)
- [Cross Validation](#31-cross-validation)
- [Hyperparameter Tuning](#32-hyperparameter-tuning)
- [SHAP — Model Explainability](#33-shap--model-explainability)

### 💾 Production
- [Saving & Loading Models](#34-saving--loading-models)
- [Pandas Power Operations](#35-pandas-power-operations)
- [NumPy Quick Reference](#36-numpy-quick-reference)
- [Imbalanced Data Toolkit](#37-imbalanced-data-toolkit)
- [Algorithm Selection Guide](#38-algorithm-selection-guide)
- [Environment Setup](#39-environment-setup)

---

## 1. Reading CSV, Excel, JSON & More

### CSV Files — `pd.read_csv`

```python
import pandas as pd

# Basic read
df = pd.read_csv("data.csv")

# With options
df = pd.read_csv(
    "data.csv",
    sep=",",               # delimiter (use \t for TSV)
    header=0,              # row number for column names
    index_col="id",        # set a column as index
    usecols=["a","b","c"], # load only these columns
    dtype={"age": int},    # force dtype per column
    na_values=["NA","--"], # treat as NaN
    parse_dates=["date"],  # auto-parse date columns
    nrows=1000,            # read first N rows
    skiprows=2,            # skip first N rows
    encoding="utf-8",      # file encoding
    low_memory=False,      # better dtype inference
    chunksize=10000,       # iterator for large files
)

# Read in chunks (large files)
chunks = []
for chunk in pd.read_csv("big.csv", chunksize=50000):
    chunk = chunk[chunk["value"] > 0]   # filter per chunk
    chunks.append(chunk)
df = pd.concat(chunks, ignore_index=True)

# Read from URL
df = pd.read_csv("https://example.com/data.csv")

# Read compressed
df = pd.read_csv("data.csv.gz", compression="gzip")
```

### Excel Files — `pd.read_excel`

```python
import pandas as pd

# Basic (first sheet)
df = pd.read_excel("data.xlsx")

# Specific sheet
df = pd.read_excel("data.xlsx", sheet_name="Sheet2")

# Multiple sheets → dict of DataFrames
dfs = pd.read_excel("data.xlsx", sheet_name=None)
df1 = dfs["Revenue"]
df2 = dfs["Expenses"]

# With options
df = pd.read_excel(
    "data.xlsx",
    sheet_name=0,
    header=1,          # second row as header
    usecols="A:F",     # Excel column range
    skiprows=[0,2],    # skip specific rows
    na_values=["N/A"],
    dtype={"col": str},
)

# Write back to Excel
with pd.ExcelWriter("output.xlsx", engine="openpyxl") as w:
    df1.to_excel(w, sheet_name="Sheet1", index=False)
    df2.to_excel(w, sheet_name="Sheet2", index=False)

# Install: pip install openpyxl xlrd
```

### JSON Files

```python
import pandas as pd
import json

# Flat JSON
df = pd.read_json("data.json")

# Nested JSON — normalize it
with open("nested.json") as f:
    data = json.load(f)

df = pd.json_normalize(
    data,
    record_path="orders",       # nested list key
    meta=["user_id","name"],    # parent fields to keep
    sep="_",                    # flatten separator
)

# JSON Lines (one JSON per line)
df = pd.read_json("data.jsonl", lines=True)

# From a JSON string
df = pd.DataFrame(json.loads(json_string))

# To JSON
df.to_json("out.json", orient="records", indent=2)
# orient options: records, split, index, columns, values
```

### Other Text Formats

```python
import pandas as pd

# TSV (tab-separated)
df = pd.read_csv("data.tsv", sep="\t")

# Fixed-width format
df = pd.read_fwf(
    "data.txt",
    widths=[10, 8, 12, 6],   # column widths
    colspecs=[(0,10),(10,18)] # or explicit positions
)

# From clipboard (after Ctrl+C on spreadsheet)
df = pd.read_clipboard()

# HTML tables from a webpage
tables = pd.read_html("https://example.com/table.html")
df = tables[0]  # first table found

# Space-separated
df = pd.read_csv("data.txt", sep=r"\s+", engine="python")

# Multiple CSVs → one DataFrame
import glob
files = glob.glob("data/month_*.csv")
df = pd.concat(
    [pd.read_csv(f) for f in files],
    ignore_index=True
)
```

---

## 2. SQL & Databases

### SQLAlchemy + pandas

```python
import pandas as pd
from sqlalchemy import create_engine, text

# SQLite
engine = create_engine("sqlite:///mydb.db")

# PostgreSQL
engine = create_engine(
    "postgresql+psycopg2://user:pass@host:5432/dbname"
)

# MySQL
engine = create_engine(
    "mysql+pymysql://user:pass@host:3306/dbname"
)

# Read with SQL query
df = pd.read_sql(
    "SELECT * FROM users WHERE active = 1",
    con=engine
)

# Parameterized query (safe from SQL injection)
query = text("SELECT * FROM orders WHERE year = :yr")
df = pd.read_sql(query, con=engine, params={"yr": 2024})

# Read entire table
df = pd.read_sql_table("customers", con=engine)

# Write to SQL
df.to_sql("table_name", con=engine,
          if_exists="replace",  # or 'append', 'fail'
          index=False, chunksize=500)

# Using context manager
with engine.connect() as conn:
    result = conn.execute(text("SELECT COUNT(*) FROM t"))
    print(result.fetchone())
```

### sqlite3, MongoDB, Snowflake, BigQuery

```python
import sqlite3, pandas as pd

# Raw sqlite3
conn = sqlite3.connect("mydb.db")
df = pd.read_sql_query("SELECT * FROM sales", conn)
conn.close()

# Context manager
with sqlite3.connect("mydb.db") as conn:
    df = pd.read_sql_query("SELECT * FROM t", conn)

# MongoDB (pymongo)
from pymongo import MongoClient
client = MongoClient("mongodb://localhost:27017/")
db = client["mydb"]
cursor = db["collection"].find({"active": True})
df = pd.DataFrame(list(cursor))

# Snowflake (snowflake-connector-python)
import snowflake.connector
conn = snowflake.connector.connect(
    user="USER", password="PASS",
    account="account.region", warehouse="WH",
    database="DB", schema="PUBLIC"
)
df = pd.read_sql("SELECT * FROM t LIMIT 100", conn)

# BigQuery (google-cloud-bigquery)
from google.cloud import bigquery
client = bigquery.Client()
df = client.query("SELECT * FROM dataset.table").to_dataframe()
```

---

## 3. Parquet, Feather, APIs & Cloud

### Parquet & Feather (Fast Formats)

```python
import pandas as pd

# Parquet — columnar, compressed, fast (pip install pyarrow)
df.to_parquet("data.parquet", engine="pyarrow", index=False)
df = pd.read_parquet("data.parquet", engine="pyarrow")

# Read specific columns
df = pd.read_parquet("data.parquet", columns=["a","b"])

# Feather — ultra-fast for IPC
df.to_feather("data.feather")
df = pd.read_feather("data.feather")

# Pickle (Python objects)
df.to_pickle("data.pkl")
df = pd.read_pickle("data.pkl")

# HDF5 — large scientific data
df.to_hdf("data.h5", key="df", mode="w")
df = pd.read_hdf("data.h5", key="df")

# Parquet with partitioning (using pyarrow directly)
import pyarrow as pa, pyarrow.parquet as pq
table = pa.Table.from_pandas(df)
pq.write_to_dataset(table, root_path="data/",
                    partition_cols=["year","month"])
```

### REST APIs & Web Scraping

```python
import requests, pandas as pd

# GET request → DataFrame
resp = requests.get("https://api.example.com/data",
                    headers={"Authorization": "Bearer TOKEN"},
                    params={"limit": 100, "page": 1})
resp.raise_for_status()
df = pd.DataFrame(resp.json()["results"])

# POST request
resp = requests.post("https://api.example.com/query",
                     json={"filter": "active"})
data = resp.json()

# Pagination loop
all_data, page = [], 1
while True:
    r = requests.get(url, params={"page": page}).json()
    if not r["data"]: break
    all_data.extend(r["data"])
    page += 1
df = pd.DataFrame(all_data)

# Web Scraping with BeautifulSoup
from bs4 import BeautifulSoup
html = requests.get("https://example.com").text
soup = BeautifulSoup(html, "html.parser")
rows = soup.select("table tbody tr")
records = [[td.text.strip() for td in row.select("td")]
           for row in rows]
df = pd.DataFrame(records)
```

---

## 4. Missing Values & Duplicates

### Detecting & Removing Nulls

```python
import pandas as pd
import numpy as np

# Detect nulls
df.isnull()               # boolean mask
df.isnull().sum()         # null count per column
df.isnull().mean() * 100  # null % per column
df.info()                 # dtypes + non-null counts
df.notnull().all()        # columns with no nulls

# Rows with any null
df[df.isnull().any(axis=1)]

# Heatmap of nulls
import seaborn as sns, matplotlib.pyplot as plt
sns.heatmap(df.isnull(), cbar=False, cmap="viridis")

# Drop nulls
df.dropna()                        # any null in row
df.dropna(axis=1)                  # any null in col
df.dropna(how="all")               # only all-null rows
df.dropna(subset=["age","salary"]) # nulls in these cols
df.dropna(thresh=5)                # keep rows with ≥5 non-null

# Fill nulls
df.fillna(0)                       # fill all with 0
df["col"].fillna(df["col"].mean()) # column mean
df.ffill()                         # forward fill (pandas 2.x)
df.bfill()                         # backward fill
df.fillna({"a": 0, "b": "unknown"}) # per column
```

### Duplicates & Data Integrity

```python
import pandas as pd

# Detect duplicates
df.duplicated()                     # boolean Series
df.duplicated().sum()               # count
df[df.duplicated(keep=False)]       # all duplicate rows
df.duplicated(subset=["id","date"]) # on specific cols

# Remove duplicates
df.drop_duplicates()
df.drop_duplicates(subset=["email"])
df.drop_duplicates(keep="last")     # keep last occurrence
df.drop_duplicates(keep=False)      # drop ALL dupes

# Reset index after cleaning
df = df.drop_duplicates().reset_index(drop=True)

# Check unique values
df["col"].nunique()           # number of unique values
df["col"].unique()            # array of unique values
df["col"].value_counts()      # frequency table
df["col"].value_counts(normalize=True)  # proportions

# Validate: no nulls in key columns
assert df["id"].notnull().all(), "IDs contain nulls!"
assert df["id"].is_unique, "IDs are not unique!"

# Data consistency
df["date"] = pd.to_datetime(df["date"], errors="coerce")
invalid_dates = df[df["date"].isna()]
```

---

## 5. Dtypes, Type Conversion & Dates

### Type Conversions

```python
import pandas as pd

# Check dtypes
df.dtypes
df.select_dtypes(include=["number"])
df.select_dtypes(include=["object", "category"])

# Cast types
df["age"]   = df["age"].astype(int)
df["price"] = df["price"].astype(float)
df["flag"]  = df["flag"].astype(bool)

# Safe numeric conversion (coerce errors to NaN)
df["col"] = pd.to_numeric(df["col"], errors="coerce")

# Categorical — saves memory on low-cardinality cols
df["city"] = df["city"].astype("category")

# Optimize memory usage
for col in df.select_dtypes("int64").columns:
    df[col] = pd.to_numeric(df[col], downcast="integer")
for col in df.select_dtypes("float64").columns:
    df[col] = pd.to_numeric(df[col], downcast="float")

print(df.memory_usage(deep=True).sum() / 1e6, "MB")
```

### Date & Time Handling

```python
import pandas as pd

# Parse dates
df["date"] = pd.to_datetime(df["date"])
df["date"] = pd.to_datetime(df["date"],
                             format="%Y-%m-%d",
                             errors="coerce")
# Common formats: "%d/%m/%Y", "%m-%d-%Y %H:%M:%S"

# Extract components
df["year"]    = df["date"].dt.year
df["month"]   = df["date"].dt.month
df["day"]     = df["date"].dt.day
df["weekday"] = df["date"].dt.day_name()
df["quarter"] = df["date"].dt.quarter
df["week"]    = df["date"].dt.isocalendar().week
df["hour"]    = df["date"].dt.hour
df["is_wknd"] = df["date"].dt.dayofweek >= 5

# Timedelta
df["days_since"] = (pd.Timestamp.today() - df["date"]).dt.days
df["age_years"]  = df["days_since"] / 365.25

# Resample (time series)
df = df.set_index("date")
monthly = df.resample("ME").sum()  # month-end
weekly  = df.resample("W").mean()
daily   = df.resample("D").ffill() # forward fill gaps

# Date ranges
date_range = pd.date_range("2024-01-01", periods=12, freq="ME")
pd.date_range("2024-01-01", "2024-12-31", freq="B")  # business days
```

---

## 6. Outlier Detection & Treatment

### IQR & Z-Score Methods

```python
import pandas as pd
import numpy as np
from scipy import stats

# IQR Method
Q1 = df["col"].quantile(0.25)
Q3 = df["col"].quantile(0.75)
IQR = Q3 - Q1
lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

# Flag outliers
df["is_outlier"] = ~df["col"].between(lower, upper)

# Remove outliers
df_clean = df[df["col"].between(lower, upper)]

# Winsorize (clip instead of remove)
df["col_clipped"] = df["col"].clip(lower=lower, upper=upper)

# Z-Score Method
z_scores = np.abs(stats.zscore(df["col"].dropna()))
df_clean = df[z_scores < 3]  # keep within 3 std

# Multiple columns at once
from scipy.stats import zscore
numeric_cols = df.select_dtypes("number").columns
z = df[numeric_cols].apply(zscore)
df_clean = df[(np.abs(z) < 3).all(axis=1)]

# Modified Z-score (robust)
median = df["col"].median()
mad = (df["col"] - median).abs().median()
modified_z = 0.6745 * (df["col"] - median) / mad
df_clean = df[modified_z.abs() < 3.5]
```

### Isolation Forest, LOF & Elliptic Envelope

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.covariance import EllipticEnvelope
import numpy as np

X = df[["col1","col2","col3"]].values

# --- Isolation Forest ---
iso = IsolationForest(
    contamination=0.05,   # expected % of outliers
    n_estimators=100,
    random_state=42
)
df["anomaly_iso"] = iso.fit_predict(X)
# -1 = outlier, 1 = inlier
df_clean = df[df["anomaly_iso"] == 1]

# Anomaly scores (lower = more anomalous)
df["iso_score"] = iso.decision_function(X)

# --- Local Outlier Factor ---
lof = LocalOutlierFactor(
    n_neighbors=20,
    contamination=0.05
)
df["anomaly_lof"] = lof.fit_predict(X)

# --- Elliptic Envelope (Gaussian assumption) ---
ee = EllipticEnvelope(contamination=0.05)
df["anomaly_ee"] = ee.fit_predict(X)
```

---

## 7. String Cleaning & Regex

### String Operations (str accessor)

```python
import pandas as pd

s = df["text"]

# Basic cleaning
s.str.strip()            # remove leading/trailing spaces
s.str.lower()            # lowercase
s.str.upper()
s.str.title()

# Replace
s.str.replace("$", "", regex=False)
s.str.replace(r"\s+", " ", regex=True)       # collapse spaces
s.str.replace(r"[^\w\s]", "", regex=True)    # remove punctuation

# Extract
s.str.extract(r"(\d+)")          # first group → column
s.str.extractall(r"(\d+)")       # all matches
s.str.findall(r"\d+")            # list of all matches

# Split → multiple columns
df[["first","last"]] = df["name"].str.split(" ", n=1, expand=True)

# Contains / match
df[s.str.contains("error", na=False)]
df[s.str.startswith("A")]
df[s.str.endswith(".com")]
df[s.str.match(r"^\d{4}-\d{2}-\d{2}$")]

# Length and padding
s.str.len()
s.str.pad(10, side="left", fillchar="0")
s.str.slice(0, 50)    # truncate to 50 chars
```

### Advanced String Cleaning & Fuzzy Matching

```python
import pandas as pd, re, unicodedata

# Normalize unicode
def normalize(text):
    if pd.isna(text): return text
    return unicodedata.normalize("NFKD", str(text))
df["text"] = df["text"].apply(normalize)

# Standardize phone numbers
df["phone"] = (df["phone"]
    .str.replace(r"[\s\-\(\)\.]", "", regex=True)
    .str.replace(r"^(\+91|91|0)", "", regex=True))

# Email validation
email_pattern = r"^[\w\.-]+@[\w\.-]+\.\w+$"
df["email_valid"] = df["email"].str.match(email_pattern)

# Fuzzy matching — pip install thefuzz
from thefuzz import fuzz, process

choices = ["New York", "Los Angeles", "Chicago"]
best = process.extractOne("new york", choices)
# → ("New York", 90, 0)

# Vectorized fuzzy match
df["match"] = df["city"].apply(
    lambda x: process.extractOne(x, choices)[0]
             if pd.notna(x) else None
)

# Standardize categories
mapping = {
    "yes": 1, "y": 1, "true": 1, "1": 1,
    "no": 0, "n": 0, "false": 0, "0": 0
}
df["bool_col"] = df["bool_col"].str.lower().map(mapping)
```

---

## 8. Data Exploration

### First Look at Any Dataset

```python
import pandas as pd

df.head()           # first 5 rows
df.tail(10)         # last 10 rows
df.sample(5)        # random 5 rows
df.shape            # (rows, cols)
df.ndim             # number of dimensions
df.size             # total elements
df.columns.tolist() # column names
df.index            # row index
df.dtypes           # data types
df.info()           # summary: nulls, dtypes, memory
df.memory_usage(deep=True).sum() / 1e6  # MB

# Statistical summary
df.describe()                   # numeric cols
df.describe(include="all")      # all cols
df.describe(include="object")   # string cols only
df.describe(percentiles=[.1,.25,.75,.9,.95,.99])

# Categorical counts
for col in df.select_dtypes("object").columns:
    print(f"\n{col}:")
    print(df[col].value_counts().head(10))
```

### Correlations & Relationships

```python
import pandas as pd
import numpy as np

# Pearson correlation (linear)
df.corr(numeric_only=True)

# Spearman (monotonic, robust to outliers)
df.corr(method="spearman", numeric_only=True)

# Kendall (rank-based)
df.corr(method="kendall", numeric_only=True)

# Correlation with target
df.corrwith(df["target"]).sort_values()

# Cross-tabulation (categorical vs categorical)
pd.crosstab(df["gender"], df["churn"],
            normalize="index")   # row percentages

# Pivot tables
df.pivot_table(
    values="sales",
    index="region",
    columns="product",
    aggfunc="sum",
    fill_value=0
)

# GroupBy exploration
df.groupby("category")["value"].agg(
    ["mean","median","std","count","min","max"]
)

# Multi-level groupby
df.groupby(["year","month"])["revenue"].sum().unstack()
```

---

## 9. Full EDA Workflow

### Automated EDA Libraries

```python
# --- ydata-profiling (formerly pandas-profiling) ---
# pip install ydata-profiling
from ydata_profiling import ProfileReport

profile = ProfileReport(df,
    title="My Dataset EDA",
    explorative=True,
    correlations={"pearson":{"calculate":True}},
    missing_diagrams={"matrix":True}
)
profile.to_file("eda_report.html")
profile.to_notebook_iframe()   # in Jupyter

# --- Sweetviz ---
# pip install sweetviz
import sweetviz as sv
report = sv.analyze(df, target_feat="target")
report.show_html("sweetviz_report.html")

# Compare train vs test
report = sv.compare([train,"Train"], [test,"Test"])
report.show_html()

# --- D-Tale (interactive) ---
# pip install dtale
import dtale
d = dtale.show(df)
d.open_browser()
```

### Manual EDA Template

```python
import pandas as pd, numpy as np

def eda_report(df):
    print("="*60)
    print(f"Shape: {df.shape}")
    print(f"Memory: {df.memory_usage(deep=True).sum()/1e6:.2f} MB")
    print("\n--- DTYPES ---")
    print(df.dtypes.value_counts())
    print("\n--- NULLS (%) ---")
    null_pct = df.isnull().mean().mul(100).round(2)
    print(null_pct[null_pct > 0].sort_values(ascending=False))
    print("\n--- DUPLICATES ---")
    print(f"Duplicate rows: {df.duplicated().sum()}")
    print("\n--- NUMERIC SUMMARY ---")
    print(df.describe().T)
    print("\n--- CATEGORICAL ---")
    for col in df.select_dtypes("object"):
        print(f"{col}: {df[col].nunique()} unique")

eda_report(df)
```

---

## 10. Scaling, Encoding & Normalization

### Feature Scaling

```python
from sklearn.preprocessing import (
    StandardScaler, MinMaxScaler, RobustScaler,
    MaxAbsScaler, Normalizer, PowerTransformer, QuantileTransformer
)

X = df[["age","income","score"]].values

# StandardScaler — zero mean, unit variance (most common)
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
X_orig   = scaler.inverse_transform(X_scaled)  # inverse

# MinMaxScaler — scales to [0, 1] range
mm = MinMaxScaler(feature_range=(0, 1))
X_mm = mm.fit_transform(X)

# RobustScaler — uses IQR, robust to outliers
rb = RobustScaler()
X_rb = rb.fit_transform(X)

# Normalizer — scales each SAMPLE (row) to unit norm
norm = Normalizer(norm="l2")   # l1, l2, max
X_norm = norm.fit_transform(X)

# PowerTransformer — make data more Gaussian
pt = PowerTransformer(method="yeo-johnson")   # or "box-cox" (pos only)
X_pt = pt.fit_transform(X)

# QuantileTransformer — maps to uniform or normal
qt = QuantileTransformer(output_distribution="normal", n_quantiles=1000)
X_qt = qt.fit_transform(X)

# IMPORTANT: Always fit on TRAIN, transform TEST
scaler.fit(X_train)
X_train_s = scaler.transform(X_train)
X_test_s  = scaler.transform(X_test)   # use train stats!
```

### Categorical Encoding

```python
import pandas as pd
from sklearn.preprocessing import (
    LabelEncoder, OrdinalEncoder, OneHotEncoder
)

# --- Label Encoding (binary or ordinal) ---
le = LabelEncoder()
df["encoded"] = le.fit_transform(df["category"])
# Decode: le.inverse_transform([0,1,2])
# Classes: le.classes_

# --- Ordinal Encoding ---
oe = OrdinalEncoder(
    categories=[["low","medium","high"]],
    handle_unknown="use_encoded_value",
    unknown_value=-1
)
df[["rank_enc"]] = oe.fit_transform(df[["rank"]])

# --- One-Hot Encoding (sklearn) ---
ohe = OneHotEncoder(
    drop="first",        # avoid multicollinearity
    sparse_output=False, # dense array
    handle_unknown="ignore"
)
encoded = ohe.fit_transform(df[["city"]])
cols = ohe.get_feature_names_out()

# --- Pandas get_dummies ---
df = pd.get_dummies(df, columns=["city","gender"],
                    drop_first=True, dtype=int)

# --- Target Encoding (high cardinality) ---
# pip install category_encoders
import category_encoders as ce
te = ce.TargetEncoder(cols=["city"])
df["city_enc"] = te.fit_transform(df["city"], df["target"])

# --- Binary Encoding ---
be = ce.BinaryEncoder(cols=["city"])
df = be.fit_transform(df)
```

---

## 11. Advanced Imputation & Train/Test Split

### SimpleImputer, KNN Imputer & Iterative Imputer

```python
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer
from sklearn.impute import IterativeImputer
import numpy as np

# SimpleImputer
imp = SimpleImputer(strategy="mean")
# strategies: mean, median, most_frequent, constant
imp = SimpleImputer(strategy="constant", fill_value=0)
X_imp = imp.fit_transform(X)

# KNN Imputer — uses K nearest rows
knn_imp = KNNImputer(n_neighbors=5, weights="uniform")
X_knn = knn_imp.fit_transform(X)

# Iterative Imputer (MICE) — models each feature
iter_imp = IterativeImputer(
    max_iter=10,
    random_state=42,
    initial_strategy="mean"
)
X_iter = iter_imp.fit_transform(X)

# Add missingness indicator as feature
for col in df.columns[df.isnull().any()]:
    df[f"{col}_missing"] = df[col].isnull().astype(int)
df.fillna(df.median(numeric_only=True), inplace=True)
```

### Train / Validation / Test Split

```python
from sklearn.model_selection import train_test_split

X = df.drop("target", axis=1)
y = df["target"]

# Basic split
X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,       # 20% test
    random_state=42,     # reproducibility
    stratify=y           # preserve class balance
)

# Three-way split: 60/20/20
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

# Time-series split (no shuffling!)
split = int(len(df) * 0.8)
train = df.iloc[:split]
test  = df.iloc[split:]

# Group-aware split (prevent leakage)
from sklearn.model_selection import GroupShuffleSplit
gss = GroupShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, test_idx = next(gss.split(X, y, groups=df["user_id"]))
X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
```

---

## 12. Creating New Features

### Numeric Feature Creation

```python
import pandas as pd, numpy as np

# Arithmetic interactions
df["price_per_sqft"] = df["price"] / df["sqft"]
df["age_income"]     = df["age"] * df["income"]
df["bmi"]            = df["weight"] / (df["height"] ** 2)

# Polynomial features
from sklearn.preprocessing import PolynomialFeatures
poly = PolynomialFeatures(degree=2, include_bias=False,
                          interaction_only=False)
X_poly = poly.fit_transform(X[["a","b"]])
names  = poly.get_feature_names_out(["a","b"])

# Log / sqrt transformations (right-skewed data)
df["log_income"] = np.log1p(df["income"])   # log(x+1), safe for 0
df["sqrt_area"]  = np.sqrt(df["area"])
df["cbrt_value"] = np.cbrt(df["value"])

# Binning / discretization
df["age_bin"] = pd.cut(df["age"],
    bins=[0,18,35,55,100],
    labels=["<18","18-35","35-55","55+"])

df["decile"] = pd.qcut(df["income"], q=10,
    labels=False, duplicates="drop")   # quantile-based

# Rolling statistics (time series)
df = df.sort_values("date")
df["rolling_mean_7"]  = df["sales"].rolling(7).mean()
df["rolling_std_7"]   = df["sales"].rolling(7).std()
df["rolling_max_30"]  = df["sales"].rolling(30).max()
df["ewm_7"]           = df["sales"].ewm(span=7).mean()

# Lag features
df["lag_1"]  = df["sales"].shift(1)
df["lag_7"]  = df["sales"].shift(7)
df["lag_30"] = df["sales"].shift(30)
```

### Aggregation & Group Features

```python
import pandas as pd, numpy as np

# Group statistics as features
agg = df.groupby("user_id")["amount"].agg(
    user_total="sum",
    user_mean="mean",
    user_count="count",
    user_max="max",
    user_std="std"
).reset_index()
df = df.merge(agg, on="user_id", how="left")

# Frequency encoding
freq = df["city"].value_counts().to_dict()
df["city_freq"] = df["city"].map(freq)

# Ratio features
df["pct_of_total"] = df["amount"] / df["total"]
df["rank_in_group"] = df.groupby("group")["score"] \
                        .rank(method="dense", ascending=False)

# Cumulative features
df["cumsum_sales"] = df.groupby("store")["sales"].cumsum()
df["cumax_sales"]  = df.groupby("store")["sales"].cummax()

# Date-based features
df["date"]       = pd.to_datetime(df["date"])
df["days_to_event"] = (df["event_date"] - df["date"]).dt.days
df["is_holiday"]    = df["date"].isin(holiday_list).astype(int)
df["is_month_end"]  = df["date"].dt.is_month_end.astype(int)

# Cyclical encoding (for periodic features)
df["sin_month"] = np.sin(2 * np.pi * df["month"] / 12)
df["cos_month"] = np.cos(2 * np.pi * df["month"] / 12)
```

---

## 13. Feature Selection

### Filter & Wrapper Methods

```python
from sklearn.feature_selection import (
    SelectKBest, f_classif, f_regression,
    mutual_info_classif, mutual_info_regression,
    RFE, RFECV, VarianceThreshold
)

# Remove near-zero variance features
vt = VarianceThreshold(threshold=0.01)
X_vt = vt.fit_transform(X)

# Univariate — select top K features
sel = SelectKBest(f_classif, k=10)
X_new  = sel.fit_transform(X, y)
selected = X.columns[sel.get_support()].tolist()

# Mutual Information (non-linear)
sel_mi = SelectKBest(mutual_info_classif, k=10)
X_mi   = sel_mi.fit_transform(X, y)

# Recursive Feature Elimination (RFE)
from sklearn.ensemble import RandomForestClassifier
rfe = RFE(estimator=RandomForestClassifier(n_estimators=50),
          n_features_to_select=10)
rfe.fit(X_train, y_train)
selected = X.columns[rfe.support_].tolist()
ranking  = pd.Series(rfe.ranking_, index=X.columns).sort_values()

# RFECV — auto-select optimal number
rfecv = RFECV(RandomForestClassifier(n_estimators=50),
              cv=5, scoring="roc_auc", min_features_to_select=5)
rfecv.fit(X_train, y_train)
print(f"Optimal features: {rfecv.n_features_}")
```

### Model-Based & Regularization Selection

```python
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LassoCV
from sklearn.ensemble import RandomForestClassifier
import numpy as np

# L1 Regularization (Lasso) — forces sparse coefficients
lasso = LassoCV(cv=5, random_state=42)
lasso.fit(X_train_scaled, y_train)
selected_lasso = X.columns[lasso.coef_ != 0].tolist()
print(f"Lasso selected {len(selected_lasso)} features")

# Feature importances from Random Forest
rf = RandomForestClassifier(n_estimators=200, random_state=42)
rf.fit(X_train, y_train)
fi = pd.Series(rf.feature_importances_, index=X.columns)
fi.sort_values(ascending=False).head(20)

# SelectFromModel
sfm = SelectFromModel(rf, threshold="mean")
sfm.fit(X_train, y_train)
X_new = sfm.transform(X_train)

# SHAP-based selection — pip install shap
import shap
explainer = shap.TreeExplainer(rf)
shap_vals  = explainer.shap_values(X_train)
shap_imp   = pd.Series(
    np.abs(shap_vals[1]).mean(0),
    index=X.columns
).sort_values(ascending=False)

# Correlation-based removal (drop highly correlated features)
corr  = df.corr().abs()
upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
drop_cols = [c for c in upper.columns if any(upper[c] > 0.95)]
df.drop(columns=drop_cols, inplace=True)
```

---

## 14. Dimensionality Reduction

### PCA, SVD & LDA

```python
from sklearn.decomposition import PCA, TruncatedSVD, NMF, KernelPCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

# --- PCA ---
pca = PCA(n_components=0.95, random_state=42)  # keep 95% variance
# or: PCA(n_components=10)
X_pca = pca.fit_transform(X_scaled)

print(pca.explained_variance_ratio_)            # per component
print(pca.explained_variance_ratio_.cumsum())   # cumulative
print(f"Reduced: {X.shape[1]} → {X_pca.shape[1]} features")

# Elbow plot
import matplotlib.pyplot as plt, numpy as np
pca_full = PCA().fit(X_scaled)
plt.plot(np.cumsum(pca_full.explained_variance_ratio_))
plt.xlabel("Components"); plt.ylabel("Variance Explained")

# --- TruncatedSVD (works on sparse matrices) ---
svd = TruncatedSVD(n_components=50, random_state=42)
X_svd = svd.fit_transform(X_sparse)

# --- LDA (supervised, maximizes class separation) ---
lda = LinearDiscriminantAnalysis(n_components=2)
X_lda = lda.fit_transform(X_scaled, y)

# --- NMF (non-negative, for topics/parts) ---
nmf = NMF(n_components=10, random_state=42)
W = nmf.fit_transform(X_nonneg)   # sample weights
H = nmf.components_               # feature weights

# --- Kernel PCA (non-linear) ---
kpca = KernelPCA(n_components=10, kernel="rbf", gamma=0.01)
X_kpca = kpca.fit_transform(X_scaled)
```

### t-SNE & UMAP (Visualization)

```python
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt

# --- t-SNE (for visualization ONLY, not ML input) ---
tsne = TSNE(
    n_components=2,
    perplexity=30,       # 5–50, more = more global structure
    learning_rate=200,
    n_iter=1000,
    random_state=42,
    metric="euclidean"
)
X_tsne = tsne.fit_transform(X_scaled)
plt.scatter(X_tsne[:,0], X_tsne[:,1], c=y, cmap="tab10", s=5)

# --- UMAP (faster, better structure preservation) ---
# pip install umap-learn
import umap
reducer = umap.UMAP(
    n_components=2,
    n_neighbors=15,   # local vs global balance
    min_dist=0.1,     # how tightly packed
    metric="euclidean",
    random_state=42
)
X_umap = reducer.fit_transform(X_scaled)

# UMAP can also transform test data
reducer.fit(X_train)
X_train_umap = reducer.transform(X_train)
X_test_umap  = reducer.transform(X_test)
```

---

## 15. Matplotlib — Low-Level Control

### Core Plots & Customization

```python
import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np

# Figure setup
fig, axes = plt.subplots(2, 3, figsize=(15, 8))
ax = axes[0, 0]

# Style
plt.style.use("seaborn-v0_8-whitegrid")
mpl.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.titlesize": 14,
    "axes.labelsize": 12,
    "figure.dpi": 120
})

# Line plot
ax.plot(x, y, color="#2196f3", linewidth=2,
        linestyle="--", marker="o", markersize=4, label="Series A")
ax.legend(); ax.set_title("Title"); ax.set_xlabel("X"); ax.set_ylabel("Y")

# Bar chart
ax.bar(categories, values, color="steelblue",
       edgecolor="white", width=0.6)

# Horizontal bar
ax.barh(categories, values)

# Scatter
sc = ax.scatter(x, y, c=z, cmap="viridis", s=50, alpha=0.6)
plt.colorbar(sc, ax=ax, label="Z value")

# Histogram
ax.hist(data, bins=30, density=True, alpha=0.7, color="teal")

# Box plot
ax.boxplot([group1, group2, group3], labels=["A","B","C"],
           patch_artist=True, notch=True)

# Annotations
ax.axhline(y=0, color="red", linestyle="-", linewidth=1)
ax.axvline(x=threshold, color="orange", linestyle="--")
ax.text(x, y, "Label", fontsize=9, ha="center")
ax.annotate("Peak", xy=(x_peak, y_peak),
            xytext=(x_peak+1, y_peak+5),
            arrowprops=dict(arrowstyle="->"))

plt.tight_layout()
plt.savefig("plot.png", dpi=150, bbox_inches="tight")
plt.show()
```

### Subplots & Advanced Layouts

```python
import matplotlib.pyplot as plt

# GridSpec for custom layouts
fig = plt.figure(figsize=(14, 8))
gs  = fig.add_gridspec(2, 3, hspace=0.4, wspace=0.3)
ax1 = fig.add_subplot(gs[0, :2])   # spans 2 cols
ax2 = fig.add_subplot(gs[0, 2])
ax3 = fig.add_subplot(gs[1, 0])
ax4 = fig.add_subplot(gs[1, 1:])   # spans 2 cols

# Twin axes (dual y-axis)
fig, ax1 = plt.subplots(figsize=(10,5))
ax2 = ax1.twinx()
ax1.plot(x, y1, "b-", label="Revenue")
ax2.plot(x, y2, "r--", label="Growth %")
ax1.set_ylabel("Revenue", color="b")
ax2.set_ylabel("Growth %", color="r")

# Pie / Donut
fig, ax = plt.subplots()
ax.pie(sizes, labels=labels, autopct="%1.1f%%",
       wedgeprops=dict(width=0.5))  # donut if width < 1

# 3D plot
from mpl_toolkits.mplot3d import Axes3D
fig = plt.figure()
ax  = fig.add_subplot(111, projection="3d")
ax.scatter(x, y, z, c=z, cmap="plasma")

# Multiple lines in a loop
for col in df.columns:
    ax.plot(df.index, df[col], label=col, alpha=0.8)
ax.legend(loc="upper left", ncols=3)
```

---

## 16. Seaborn — Statistical Plots

### Distribution & Categorical Plots

```python
import seaborn as sns
import matplotlib.pyplot as plt

sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)

# Distribution
sns.histplot(df["col"], bins=30, kde=True, color="steelblue")
sns.kdeplot(df["col"], fill=True, bw_adjust=0.5)
sns.ecdfplot(df["col"])   # empirical CDF

# Box / Violin / Strip
sns.boxplot(x="category", y="value", data=df, hue="group",
            palette="Set2", width=0.6)
sns.violinplot(x="category", y="value", data=df,
               inner="box", cut=0)
sns.stripplot(x="category", y="value", data=df,
              jitter=True, alpha=0.4, size=3)

# Combined: box + strip
ax = sns.boxplot(x="cat", y="val", data=df, fliersize=0)
sns.stripplot(x="cat", y="val", data=df, color=".3",
              jitter=True, size=3, ax=ax)

# Bar plot with CI
sns.barplot(x="category", y="value", data=df,
            errorbar="ci", capsize=0.1)

# Count plot
sns.countplot(x="category", data=df,
              order=df["category"].value_counts().index)
```

### Correlation, Pair & Heatmaps

```python
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Correlation heatmap
corr = df.corr(numeric_only=True)
mask = np.triu(np.ones_like(corr), k=1)

fig, ax = plt.subplots(figsize=(12, 9))
sns.heatmap(corr, mask=mask, annot=True, fmt=".2f",
            cmap="RdYlGn", center=0, vmin=-1, vmax=1,
            linewidths=0.5, ax=ax, cbar_kws={"shrink":0.8})

# Pairplot (all vs all)
sns.pairplot(df, hue="target", diag_kind="kde",
             plot_kws={"alpha": 0.5, "s": 20},
             diag_kws={"fill": True})

# Scatter + regression
sns.regplot(x="x", y="y", data=df, scatter_kws={"s":10})
sns.lmplot(x="x", y="y", data=df, hue="group",
           col="region", col_wrap=3)

# FacetGrid
g = sns.FacetGrid(df, col="category", row="region",
                  height=3, aspect=1.2)
g.map(sns.histplot, "value", kde=True)
g.add_legend()

# Cluster heatmap
sns.clustermap(df.corr(), cmap="vlag", figsize=(10,10),
               method="ward", metric="euclidean")
```

---

## 17. Plotly — Interactive Charts

### Plotly Express (Quick)

```python
import plotly.express as px

# Scatter
fig = px.scatter(df, x="gdp", y="life_exp", color="continent",
                 size="population", hover_name="country",
                 log_x=True, title="GDP vs Life Expectancy",
                 trendline="ols")
fig.show()

# Line
fig = px.line(df, x="date", y="value", color="series", markers=True)

# Bar
fig = px.bar(df, x="category", y="value", color="group",
             barmode="group",   # or 'stack', 'overlay'
             text_auto=True)

# Histogram
fig = px.histogram(df, x="age", nbins=40, color="gender",
                   barmode="overlay", marginal="box", opacity=0.7)

# Box
fig = px.box(df, x="category", y="value", color="group",
             notched=True, points="outliers")

# Heatmap / Correlation
fig = px.imshow(corr, text_auto=".2f",
                color_continuous_scale="RdYlGn",
                zmin=-1, zmax=1, aspect="auto")

# Choropleth map
fig = px.choropleth(df, locations="iso_alpha",
                    color="value", hover_name="country",
                    color_continuous_scale="Viridis")

fig.update_layout(template="plotly_dark", height=500)
fig.write_html("chart.html")   # save as interactive HTML
```

### Plotly Graph Objects (Advanced)

```python
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Subplots
fig = make_subplots(rows=2, cols=2,
    subplot_titles=["Line","Bar","Scatter","Pie"],
    shared_xaxes=True, vertical_spacing=0.1)

fig.add_trace(go.Scatter(x=x, y=y1, name="A",
                          line=dict(width=2)), row=1, col=1)
fig.add_trace(go.Bar(x=cats, y=vals, name="B",
                      marker_color="steelblue"), row=1, col=2)
fig.add_trace(go.Scatter(x=x, y=y2, mode="markers",
                          marker=dict(size=6, color=z,
                                      colorscale="Viridis",
                                      showscale=True)), row=2, col=1)
fig.add_trace(go.Pie(labels=labels, values=values, hole=0.4), row=2, col=2)

# Candlestick (finance)
fig = go.Figure(go.Candlestick(
    x=df["date"], open=df["open"],
    high=df["high"], low=df["low"], close=df["close"]
))

# 3D Scatter
fig = go.Figure(go.Scatter3d(
    x=df.x, y=df.y, z=df.z, mode="markers",
    marker=dict(size=3, color=df.color, colorscale="Plasma")
))
fig.show()
```

---

## 18. Descriptive Statistics

### Scipy & Numpy Stats

```python
import numpy as np
from scipy import stats

x = df["col"].dropna().values

# Central tendency
np.mean(x); np.median(x)
stats.mode(x, keepdims=True).mode[0]

# Spread
np.std(x); np.var(x)
np.percentile(x, [25, 50, 75])
stats.iqr(x)   # interquartile range

# Shape
stats.skew(x)      # >0 right-skewed, <0 left-skewed
stats.kurtosis(x)  # excess kurtosis (0 = normal)

# Normality tests
stat, p = stats.shapiro(x[:5000])   # best for n < 5000
print(f"Shapiro p={p:.4f} {'normal' if p>0.05 else 'NOT normal'}")

stat, p = stats.normaltest(x)       # D'Agostino
stat, p = stats.kstest(x, "norm", args=(x.mean(), x.std()))

# Full description
stats.describe(x)
# → nobs, minmax, mean, variance, skewness, kurtosis

# Bootstrap confidence interval
from scipy.stats import bootstrap
rng = np.random.default_rng(42)
res = bootstrap((x,), np.mean, n_resamples=10000,
                confidence_level=0.95, random_state=rng)
print(res.confidence_interval)  # (low, high)
```

### Distributions Reference

```python
from scipy import stats
import numpy as np

# Normal
dist = stats.norm(loc=0, scale=1)
dist.pdf(x);  dist.cdf(x)
dist.ppf(0.95)       # inverse CDF (quantile)
dist.rvs(size=1000)  # random samples

# Common distributions
stats.t(df=10)           # Student's t
stats.chi2(df=5)         # Chi-squared
stats.f(dfn=3, dfd=20)  # F-distribution
stats.binom(n=10, p=0.3) # Binomial
stats.poisson(mu=5)      # Poisson
stats.expon(scale=2)     # Exponential
stats.lognorm(s=0.5)     # Log-normal

# Fit distribution to data
params = stats.norm.fit(data)       # returns (loc, scale)
params = stats.lognorm.fit(data, floc=0)

# QQ-plot
import matplotlib.pyplot as plt
stats.probplot(x, dist="norm", plot=plt)
plt.title("Normal Q-Q Plot"); plt.show()

# KDE
kde = stats.gaussian_kde(x, bw_method="scott")
x_grid  = np.linspace(x.min(), x.max(), 200)
density = kde(x_grid)
```

---

## 19. Hypothesis Testing

### Parametric Tests

```python
from scipy import stats
import numpy as np

α = 0.05  # significance level

# --- One-sample t-test ---
stat, p = stats.ttest_1samp(sample, popmean=100)
print("Reject H0" if p < α else "Fail to reject H0")

# --- Two-sample t-test ---
stat, p = stats.ttest_ind(a, b, equal_var=True)   # Student's t
stat, p = stats.ttest_ind(a, b, equal_var=False)  # Welch's t

# --- Paired t-test ---
stat, p = stats.ttest_rel(before, after)

# --- One-way ANOVA ---
stat, p = stats.f_oneway(g1, g2, g3, g4)

# Post-hoc: Tukey HSD
from statsmodels.stats.multicomp import pairwise_tukeyhsd
tukey = pairwise_tukeyhsd(df["value"], df["group"], alpha=0.05)
print(tukey)

# --- Chi-squared test (independence) ---
ct = pd.crosstab(df["var1"], df["var2"])
stat, p, dof, expected = stats.chi2_contingency(ct)

# --- Proportions z-test ---
from statsmodels.stats.proportion import proportions_ztest
count = np.array([45, 55])
nobs  = np.array([100, 100])
stat, p = proportions_ztest(count, nobs)
```

### Non-Parametric & Effect Size

```python
from scipy import stats
import pingouin as pg  # pip install pingouin

# --- Mann-Whitney U (non-parametric t-test) ---
stat, p = stats.mannwhitneyu(a, b, alternative="two-sided")

# --- Wilcoxon signed-rank (non-parametric paired t) ---
stat, p = stats.wilcoxon(before, after)

# --- Kruskal-Wallis (non-parametric ANOVA) ---
stat, p = stats.kruskal(g1, g2, g3)

# --- Spearman rank correlation ---
r, p = stats.spearmanr(x, y)

# --- Pearson correlation with p-value ---
r, p = stats.pearsonr(x, y)

# --- Effect sizes ---
d = pg.compute_effsize(a, b, eftype="cohen")
# d=0.2 small, 0.5 medium, 0.8 large

# Cramér's V (categorical association strength)
def cramers_v(x, y):
    ct = pd.crosstab(x, y)
    chi2 = stats.chi2_contingency(ct)[0]
    n = ct.sum().sum()
    min_dim = min(ct.shape) - 1
    return np.sqrt(chi2 / (n * min_dim))

# --- Multiple testing correction ---
from statsmodels.stats.multitest import multipletests
pvals = [0.001, 0.04, 0.08, 0.15]
reject, pvals_corr, _, _ = multipletests(pvals, method="fdr_bh")
# methods: bonferroni, holm, fdr_bh, fdr_by
```

---

## 20. Statistical Modeling (statsmodels)

### OLS Regression & GLM

```python
import statsmodels.api as sm
import statsmodels.formula.api as smf

# --- OLS with formula interface ---
model  = smf.ols("target ~ age + income + C(city)", data=df)
result = model.fit()
print(result.summary())   # full stats report

# Key attributes
result.params        # coefficients
result.pvalues       # p-values
result.rsquared      # R²
result.rsquared_adj  # adjusted R²
result.aic; result.bic
result.fittedvalues  # predictions
result.resid         # residuals

# Predict on new data
new_X = pd.DataFrame({"age":[30],"income":[50000],"city":["NY"]})
result.predict(new_X)

# Heteroscedasticity-robust standard errors
result_robust = model.fit(cov_type="HC3")   # White's SE

# --- GLM (Logistic, Poisson, etc.) ---
glm = smf.glm("target ~ age + income",
               data=df, family=sm.families.Binomial())
res = glm.fit()

# Poisson (count data)
glm = smf.glm("count ~ x1 + x2",
               data=df, family=sm.families.Poisson())

# Negative Binomial (overdispersed counts)
glm = smf.glm("count ~ x1", data=df,
               family=sm.families.NegativeBinomial())
res = glm.fit()
```

### Diagnostics & Residual Checks

```python
import statsmodels.api as sm
from statsmodels.stats.diagnostic import het_breuschpagan
from statsmodels.stats.stattools import durbin_watson
import matplotlib.pyplot as plt, numpy as np

result = smf.ols("y ~ x1 + x2", data=df).fit()
resid  = result.resid
fitted = result.fittedvalues

# Heteroscedasticity (Breusch-Pagan)
bp_stat, bp_p, _, _ = het_breuschpagan(resid, result.model.exog)
print(f"Breusch-Pagan p={bp_p:.4f}")

# Autocorrelation — 2=none, <1 or >3 = concern
dw = durbin_watson(resid)
print(f"Durbin-Watson: {dw:.3f}")

# Influential observations
influence = result.get_influence()
leverage  = influence.hat_matrix_diag
cooks_d   = influence.cooks_distance[0]

# Normality of residuals
from scipy import stats
stat, p = stats.shapiro(resid)

# Residual plots
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
axes[0,0].scatter(fitted, resid, alpha=0.5)
axes[0,0].axhline(0, color="r"); axes[0,0].set_title("Residuals vs Fitted")
sm.qqplot(resid, line="s", ax=axes[0,1]); axes[0,1].set_title("Q-Q")
plt.tight_layout()
```

---

## 21. Sklearn API & Pipelines

### Complete Pipeline Example

```python
import pandas as pd, numpy as np
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report

numeric_features     = ["age","income","balance"]
categorical_features = ["city","occupation"]

X = df[numeric_features + categorical_features]
y = df["target"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y)

# Preprocessing pipelines
numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler",  StandardScaler()),
])

categorical_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("ohe",     OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe,     numeric_features),
    ("cat", categorical_pipe, categorical_features),
], remainder="drop")

# Full pipeline
full_pipe = Pipeline([
    ("prep",  preprocessor),
    ("model", GradientBoostingClassifier(n_estimators=200, random_state=42)),
])

full_pipe.fit(X_train, y_train)
y_pred = full_pipe.predict(X_test)
print(classification_report(y_test, y_pred))

# Cross-validation on full pipeline (no leakage!)
cv_scores = cross_val_score(full_pipe, X, y, cv=5,
                             scoring="roc_auc", n_jobs=-1)
print(f"CV AUC: {cv_scores.mean():.4f} ± {cv_scores.std():.4f}")
```

---

## 22. Linear Models

### Linear & Ridge / Lasso Regression

```python
from sklearn.linear_model import (
    LinearRegression, Ridge, Lasso, ElasticNet,
    BayesianRidge, HuberRegressor, QuantileRegressor
)

# OLS
lr = LinearRegression(fit_intercept=True)
lr.fit(X_train, y_train)
y_pred = lr.predict(X_test)
print(lr.coef_, lr.intercept_)

# Ridge (L2 regularization)
ridge = Ridge(alpha=1.0)   # higher alpha = more regularization
ridge.fit(X_train, y_train)

# Lasso (L1 — sparse coefficients)
lasso = Lasso(alpha=0.01, max_iter=5000)
lasso.fit(X_train, y_train)
# Non-zero coefficients = selected features
pd.Series(lasso.coef_, index=feat_names)[lasso.coef_!=0]

# ElasticNet (L1+L2)
en = ElasticNet(alpha=0.01, l1_ratio=0.5)

# Auto-tune alpha via cross-validation
from sklearn.linear_model import RidgeCV, LassoCV
ridge = RidgeCV(alphas=[0.01,0.1,1,10,100], cv=5)
ridge.fit(X_train_s, y_train)
print(f"Best alpha: {ridge.alpha_}")

# Bayesian Ridge — outputs uncertainty estimates
br = BayesianRidge()
br.fit(X_train, y_train)
y_pred, y_std = br.predict(X_test, return_std=True)

# Huber — robust to outliers
huber = HuberRegressor(epsilon=1.35, alpha=0.0001)
huber.fit(X_train, y_train)

# Quantile Regression
qr = QuantileRegressor(quantile=0.9, alpha=0.1, solver="highs")
qr.fit(X_train, y_train)
upper_bound = qr.predict(X_test)
```

### Logistic Regression

```python
from sklearn.linear_model import LogisticRegression, SGDClassifier

lr = LogisticRegression(
    C=1.0,              # 1/alpha (inverse regularization)
    penalty="l2",       # l1, l2, elasticnet, None
    solver="lbfgs",     # lbfgs, liblinear, saga, sag
    max_iter=1000,
    class_weight="balanced",  # handle imbalance
    multi_class="auto",
    random_state=42
)
lr.fit(X_train, y_train)
y_pred  = lr.predict(X_test)
y_proba = lr.predict_proba(X_test)[:, 1]   # P(class=1)

# Coefficients
pd.Series(lr.coef_[0], index=feat_names).sort_values()

# SGD (fast for large datasets, online learning)
sgd = SGDClassifier(loss="log_loss", penalty="l2",
                    alpha=0.0001, max_iter=100,
                    class_weight="balanced", random_state=42)
sgd.partial_fit(X_chunk, y_chunk, classes=[0,1])
```

---

## 23. Tree-Based Models

### Decision Tree

```python
from sklearn.tree import (
    DecisionTreeClassifier, DecisionTreeRegressor,
    plot_tree, export_text
)
import matplotlib.pyplot as plt

dt = DecisionTreeClassifier(
    max_depth=5,
    min_samples_split=20,
    min_samples_leaf=10,
    max_features="sqrt",
    criterion="gini",        # gini or entropy
    class_weight="balanced",
    random_state=42
)
dt.fit(X_train, y_train)

# Visualize tree
plt.figure(figsize=(20,10))
plot_tree(dt, feature_names=feat_names,
          class_names=["0","1"], filled=True,
          rounded=True, fontsize=8)

# Text representation
print(export_text(dt, feature_names=feat_names))

# Feature importance
pd.Series(dt.feature_importances_,
          index=feat_names).sort_values(ascending=False)
```

### Random Forest

```python
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor

rf = RandomForestClassifier(
    n_estimators=300,
    max_depth=None,        # None = grow until pure
    min_samples_split=5,
    min_samples_leaf=2,
    max_features="sqrt",   # sqrt for cls, 1/3 for reg
    max_samples=0.8,       # bagging fraction
    bootstrap=True,
    oob_score=True,        # OOB generalization estimate
    class_weight="balanced",
    n_jobs=-1,
    random_state=42
)
rf.fit(X_train, y_train)
print(f"OOB score: {rf.oob_score_:.4f}")

# Feature importance
fi = pd.Series(rf.feature_importances_,
               index=feat_names).sort_values(ascending=False)

# Permutation importance (more reliable)
from sklearn.inspection import permutation_importance
r  = permutation_importance(rf, X_test, y_test,
                             n_repeats=30, random_state=42, n_jobs=-1)
pi = pd.Series(r.importances_mean, index=feat_names).sort_values()
```

---

## 24. Gradient Boosting & Ensembles

### XGBoost

```python
import xgboost as xgb
# pip install xgboost

xgb_clf = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.05,
    max_depth=6,
    min_child_weight=1,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    reg_alpha=0,          # L1
    reg_lambda=1,         # L2
    scale_pos_weight=neg/pos,  # class imbalance
    eval_metric="auc",
    early_stopping_rounds=50,
    tree_method="hist",
    device="cuda",        # GPU (if available)
    random_state=42, n_jobs=-1
)
xgb_clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            verbose=50)

# Native API
dtrain = xgb.DMatrix(X_train, label=y_train)
dval   = xgb.DMatrix(X_val,   label=y_val)
params = {"max_depth":6, "eta":0.05, "objective":"binary:logistic",
          "eval_metric":"auc", "tree_method":"hist"}
model  = xgb.train(params, dtrain, num_boost_round=500,
                   evals=[(dval,"val")], early_stopping_rounds=50,
                   verbose_eval=50)
```

### LightGBM

```python
import lightgbm as lgb
# pip install lightgbm

lgb_clf = lgb.LGBMClassifier(
    n_estimators=1000,
    learning_rate=0.05,
    num_leaves=63,       # 2^max_depth - 1 rule of thumb
    max_depth=-1,
    min_child_samples=20,
    subsample=0.8,
    colsample_bytree=0.8,
    reg_alpha=0.1,
    reg_lambda=0.1,
    is_unbalance=True,
    n_jobs=-1,
    random_state=42, verbose=-1
)
lgb_clf.fit(X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50),
                       lgb.log_evaluation(100)])

# Native API
train_ds = lgb.Dataset(X_train, label=y_train)
val_ds   = lgb.Dataset(X_val,   label=y_val)
params   = {"objective":"binary","metric":"auc",
            "num_leaves":63,"learning_rate":0.05,"verbose":-1}
model    = lgb.train(params, train_ds, num_boost_round=1000,
                     valid_sets=[val_ds],
                     callbacks=[lgb.early_stopping(50)])
```

### CatBoost

```python
from catboost import CatBoostClassifier, Pool
# pip install catboost

cat_features = ["city","occupation","gender"]

model = CatBoostClassifier(
    iterations=1000,
    learning_rate=0.05,
    depth=6,
    l2_leaf_reg=3,
    bagging_temperature=1,
    eval_metric="AUC",
    early_stopping_rounds=50,
    random_seed=42,
    verbose=100
)
model.fit(X_train, y_train,
          cat_features=cat_features,   # no encoding needed!
          eval_set=(X_val, y_val))

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:,1]
```

### HistGradientBoosting, Voting & Stacking

```python
from sklearn.ensemble import (
    HistGradientBoostingClassifier,
    VotingClassifier, StackingClassifier
)

# HistGBM (fast, handles NaN natively)
hgb = HistGradientBoostingClassifier(
    max_iter=500, learning_rate=0.05,
    max_leaf_nodes=63,
    min_samples_leaf=20,
    l2_regularization=0.1,
    early_stopping=True, validation_fraction=0.1,
    n_iter_no_change=20, random_state=42
)

# Voting Classifier
voting = VotingClassifier(
    estimators=[("rf", rf), ("lgbm", lgbm), ("lr", lr)],
    voting="soft",     # uses probabilities (better)
    weights=[2,2,1]
)

# Stacking (meta-learner)
stacking = StackingClassifier(
    estimators=[("rf", rf), ("xgb", xgb_clf), ("lgbm", lgbm)],
    final_estimator=LogisticRegression(C=0.1),
    cv=5, stack_method="predict_proba", n_jobs=-1
)
stacking.fit(X_train, y_train)
```

---

## 25. SVM, KNN & Naive Bayes

### Support Vector Machine

```python
from sklearn.svm import SVC, SVR, LinearSVC

# SVC (classification)
svc = SVC(
    C=1.0,            # margin penalty (higher=less margin)
    kernel="rbf",     # linear, poly, rbf, sigmoid
    gamma="scale",    # kernel coefficient
    degree=3,         # for poly kernel
    probability=True, # enable predict_proba (slower)
    class_weight="balanced",
    random_state=42
)
svc.fit(X_train_scaled, y_train)   # Note: must scale features!

# SVR (regression)
svr = SVR(C=10, kernel="rbf", gamma="scale", epsilon=0.1)

# LinearSVC (fast for large datasets)
lsvc = LinearSVC(C=1.0, max_iter=5000, class_weight="balanced")
```

### KNN & Naive Bayes

```python
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB, MultinomialNB, BernoulliNB

# KNN Classifier (scale features!)
knn = KNeighborsClassifier(
    n_neighbors=5,
    weights="uniform",    # uniform or distance
    metric="euclidean",
    algorithm="auto",
    n_jobs=-1
)
knn.fit(X_train_scaled, y_train)

# Find optimal K
k_scores = []
for k in range(1, 30):
    knn = KNeighborsClassifier(n_neighbors=k)
    sc  = cross_val_score(knn, X_scaled, y, cv=5).mean()
    k_scores.append(sc)
best_k = np.argmax(k_scores) + 1

# Gaussian Naive Bayes (continuous features)
gnb = GaussianNB(var_smoothing=1e-9)
gnb.fit(X_train, y_train)

# Multinomial NB (text counts, must be non-negative)
mnb = MultinomialNB(alpha=1.0)   # Laplace smoothing
mnb.fit(X_train_counts, y_train)
```

---

## 26. Clustering (Unsupervised)

### K-Means

```python
from sklearn.cluster import KMeans, MiniBatchKMeans
from sklearn.metrics import silhouette_score, davies_bouldin_score

kmeans = KMeans(
    n_clusters=5,
    init="k-means++",   # smart initialization
    n_init=10,          # multiple restarts
    max_iter=300,
    random_state=42
)
labels  = kmeans.fit_predict(X_scaled)
centers = kmeans.cluster_centers_
inertia = kmeans.inertia_   # WCSS

# Elbow method
wcss    = []
K_range = range(2, 15)
for k in K_range:
    km = KMeans(n_clusters=k, n_init=10, random_state=42)
    km.fit(X_scaled)
    wcss.append(km.inertia_)
plt.plot(K_range, wcss, "bo-")

# Evaluation
sil = silhouette_score(X_scaled, labels)       # higher=better
db  = davies_bouldin_score(X_scaled, labels)   # lower=better

# MiniBatch KMeans (large datasets)
mb = MiniBatchKMeans(n_clusters=5, batch_size=1024, random_state=42)
```

### DBSCAN, HDBSCAN & Hierarchical

```python
from sklearn.cluster import DBSCAN, AgglomerativeClustering
import hdbscan   # pip install hdbscan

# DBSCAN (density-based, -1 = noise)
db = DBSCAN(eps=0.5, min_samples=5, metric="euclidean", n_jobs=-1)
labels    = db.fit_predict(X_scaled)
n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
n_noise    = (labels == -1).sum()

# Find eps: KNN distance elbow
from sklearn.neighbors import NearestNeighbors
nn = NearestNeighbors(n_neighbors=5)
nn.fit(X_scaled)
dists, _ = nn.kneighbors(X_scaled)
dists = np.sort(dists[:, -1])
plt.plot(dists)

# HDBSCAN (better than DBSCAN)
hdb    = hdbscan.HDBSCAN(min_cluster_size=30, min_samples=10)
labels = hdb.fit_predict(X_scaled)

# Agglomerative Hierarchical
agg    = AgglomerativeClustering(n_clusters=5, linkage="ward")
labels = agg.fit_predict(X_scaled)

# Dendrogram
from scipy.cluster.hierarchy import dendrogram, linkage
Z = linkage(X_scaled, method="ward")
dendrogram(Z, truncate_mode="lastp", p=20); plt.show()
```

### Gaussian Mixture Models

```python
from sklearn.mixture import GaussianMixture, BayesianGaussianMixture
import numpy as np

gmm = GaussianMixture(
    n_components=5,
    covariance_type="full",   # full, tied, diag, spherical
    init_params="kmeans",
    n_init=5, max_iter=100, random_state=42
)
gmm.fit(X_scaled)
labels    = gmm.predict(X_scaled)
proba     = gmm.predict_proba(X_scaled)   # soft membership
log_lik   = gmm.score(X_scaled)

# Model selection via BIC (lower=better)
bics  = []
for k in range(1, 15):
    gmm = GaussianMixture(n_components=k, n_init=5)
    gmm.fit(X_scaled)
    bics.append(gmm.bic(X_scaled))
best_k = np.argmin(bics) + 1

# Bayesian GMM (auto-selects components)
bgmm = BayesianGaussianMixture(n_components=10, n_init=5)
bgmm.fit(X_scaled)
```

---

## 27. Anomaly Detection

```python
from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.svm import OneClassSVM
from sklearn.covariance import EllipticEnvelope
from sklearn.preprocessing import StandardScaler
from scipy import stats
import numpy as np

X_scaled = StandardScaler().fit_transform(X)
contamination = 0.05

# Isolation Forest (recommended)
iso        = IsolationForest(contamination=contamination,
                              n_estimators=200, random_state=42)
iso_labels = iso.fit_predict(X_scaled)       # -1=anomaly, 1=normal
iso_scores = iso.decision_function(X_scaled) # higher=more normal

# Local Outlier Factor
lof        = LocalOutlierFactor(n_neighbors=20, contamination=contamination)
lof_labels = lof.fit_predict(X_scaled)

# One-Class SVM (train on normal data only)
ocsvm        = OneClassSVM(nu=contamination, kernel="rbf", gamma="scale")
ocsvm.fit(X_normal_scaled)
ocsvm_labels = ocsvm.predict(X_scaled)

# Elliptic Envelope (Gaussian assumption)
ee        = EllipticEnvelope(contamination=contamination)
ee_labels = ee.fit_predict(X_scaled)

# Ensemble: majority vote
all_preds      = np.vstack([iso_labels, lof_labels, ee_labels]).T
ensemble_labels, _ = stats.mode(all_preds, axis=1, keepdims=False)
```

### Autoencoder-Based Anomaly Detection

```python
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import StandardScaler
import numpy as np

# Train on normal data only
scaler         = StandardScaler()
X_normal_scaled = scaler.fit_transform(X_normal)

autoenc = MLPRegressor(
    hidden_layer_sizes=(64, 16, 64),   # bottleneck architecture
    activation="relu", max_iter=200, random_state=42
)
autoenc.fit(X_normal_scaled, X_normal_scaled)   # target = input!

# Reconstruction error
X_all_scaled  = scaler.transform(X_all)
X_recon       = autoenc.predict(X_all_scaled)
recon_error   = np.mean((X_all_scaled - X_recon)**2, axis=1)

threshold = np.percentile(recon_error[normal_mask], 95)
anomalies = recon_error > threshold
```

---

## 28. Time Series Analysis & Forecasting

### Decomposition & Stationarity

```python
import pandas as pd, numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose, STL
from statsmodels.tsa.stattools import adfuller, acf, pacf
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
import matplotlib.pyplot as plt

ts = df.set_index("date")["value"]

# Decomposition
decomp = seasonal_decompose(ts, model="additive", period=12)
decomp.plot(); plt.tight_layout()

# STL (robust)
stl    = STL(ts, period=12, robust=True)
result = stl.fit()
result.plot()

# ADF test — H0: non-stationary
adf_stat, adf_p, _, _, crit, _ = adfuller(ts.dropna())
print(f"ADF p-value: {adf_p:.4f}")
if adf_p >= 0.05:
    ts_diff = ts.diff().dropna()   # first difference

# ACF & PACF (determine AR/MA orders)
fig, (ax1, ax2) = plt.subplots(2,1, figsize=(12,6))
plot_acf(ts.dropna(),  lags=40, ax=ax1)
plot_pacf(ts.dropna(), lags=40, ax=ax2, method="ywm")
plt.tight_layout()
```

### ARIMA, SARIMA, Prophet & ML

```python
from pmdarima import auto_arima       # pip install pmdarima
from statsmodels.tsa.statespace.sarimax import SARIMAX
from prophet import Prophet           # pip install prophet

# --- Auto ARIMA ---
model    = auto_arima(ts_train, m=12, seasonal=True,
                      information_criterion="aic", stepwise=True,
                      suppress_warnings=True, n_jobs=-1)
forecast = model.predict(n_periods=12)

# --- SARIMA ---
sarima = SARIMAX(ts_train, order=(1,1,1), seasonal_order=(1,1,1,12))
result = sarima.fit(disp=False)
fc     = result.forecast(steps=12)
conf   = result.get_forecast(12).conf_int()

# --- Prophet ---
df_p = df.rename(columns={"date":"ds","value":"y"})
m    = Prophet(changepoint_prior_scale=0.05,
               yearly_seasonality=True, weekly_seasonality=True)
m.add_country_holidays(country_name="US")
m.fit(df_p)
future   = m.make_future_dataframe(periods=90, freq="D")
forecast = m.predict(future)
m.plot(forecast); m.plot_components(forecast)
```

---

## 29. NLP Basics

### Text Preprocessing & Vectorization

```python
import re, nltk, numpy as np, pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

nltk.download(["stopwords","punkt","wordnet"])
stop_words  = set(stopwords.words("english"))
lemmatizer  = WordNetLemmatizer()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"<[^>]+>",  " ", text)       # remove HTML
    text = re.sub(r"http\S+",  " ", text)        # remove URLs
    text = re.sub(r"[^a-z\s]", " ", text)        # keep letters
    text = re.sub(r"\s+",      " ", text).strip()
    return text

def preprocess(text):
    tokens = nltk.word_tokenize(text)
    tokens = [t for t in tokens if t not in stop_words and len(t) > 2]
    tokens = [lemmatizer.lemmatize(t) for t in tokens]
    return " ".join(tokens)

df["clean"]     = df["text"].apply(clean_text)
df["processed"] = df["clean"].apply(preprocess)

# TF-IDF (best general-purpose vectorizer)
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1,2),
                         sublinear_tf=True, min_df=2, max_df=0.9)
X_tfidf = tfidf.fit_transform(df["processed"])

# Word2Vec embeddings — pip install gensim
from gensim.models import Word2Vec
sentences = [text.split() for text in df["processed"]]
w2v = Word2Vec(sentences, vector_size=100, window=5,
               min_count=2, workers=4, epochs=10)

def sentence_vector(sentence):
    words = [w for w in sentence.split() if w in w2v.wv]
    return np.mean([w2v.wv[w] for w in words], axis=0) if words else np.zeros(100)
X_w2v = np.vstack(df["processed"].apply(sentence_vector))
```

### Transformers & Sentence Embeddings

```python
# pip install transformers sentence-transformers
from sentence_transformers import SentenceTransformer
from transformers import pipeline
from sklearn.metrics.pairwise import cosine_similarity

# Sentence Transformers (fast, high quality)
model      = SentenceTransformer("all-MiniLM-L6-v2")
embeddings = model.encode(df["text"].tolist(),
                           batch_size=64, show_progress_bar=True)
# embeddings.shape: (n_docs, 384)

# HuggingFace Pipelines
# Sentiment analysis
clf     = pipeline("sentiment-analysis",
                    model="distilbert-base-uncased-finetuned-sst-2-english")
results = clf(df["text"].tolist()[:10])

# Zero-shot classification (no training needed!)
zsc    = pipeline("zero-shot-classification",
                   model="facebook/bart-large-mnli")
result = zsc("This product is amazing!",
             candidate_labels=["positive","negative","neutral"])

# NER
ner      = pipeline("ner", aggregation_strategy="simple")
entities = ner("Apple Inc. is based in Cupertino, CA.")

# Semantic search
query_emb = model.encode(["best laptop for data science"])
scores    = cosine_similarity(query_emb, embeddings)[0]
top_idx   = scores.argsort()[::-1][:5]
```

---

## 30. Metrics — Classification & Regression

### Classification Metrics

```python
from sklearn.metrics import (
    accuracy_score, balanced_accuracy_score,
    precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score,
    classification_report, confusion_matrix,
    ConfusionMatrixDisplay, roc_curve,
    precision_recall_curve, matthews_corrcoef,
    log_loss
)
import matplotlib.pyplot as plt, numpy as np

y_pred = model.predict(X_test)
y_prob = model.predict_proba(X_test)[:, 1]

print(f"Accuracy:  {accuracy_score(y_test, y_pred):.4f}")
print(f"Bal. Acc:  {balanced_accuracy_score(y_test, y_pred):.4f}")
print(f"Precision: {precision_score(y_test, y_pred):.4f}")
print(f"Recall:    {recall_score(y_test, y_pred):.4f}")
print(f"F1:        {f1_score(y_test, y_pred):.4f}")
print(f"ROC-AUC:   {roc_auc_score(y_test, y_prob):.4f}")
print(f"PR-AUC:    {average_precision_score(y_test, y_prob):.4f}")
print(f"Log Loss:  {log_loss(y_test, y_prob):.4f}")
print(f"MCC:       {matthews_corrcoef(y_test, y_pred):.4f}")

# Full report
print(classification_report(y_test, y_pred,
      target_names=["Class0","Class1"], digits=4))

# Confusion matrix
cm   = confusion_matrix(y_test, y_pred)
disp = ConfusionMatrixDisplay(cm, display_labels=["0","1"])
disp.plot(cmap="Blues")

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_prob)
plt.plot(fpr, tpr, label=f"AUC={roc_auc_score(y_test,y_prob):.3f}")
plt.plot([0,1],[0,1],"k--")

# Threshold tuning
thresholds = np.arange(0.1, 0.9, 0.01)
f1_scores  = [f1_score(y_test, y_prob >= t) for t in thresholds]
best_thresh = thresholds[np.argmax(f1_scores)]
y_pred_tuned = (y_prob >= best_thresh).astype(int)
print(f"Best threshold: {best_thresh:.2f}")
```

### Regression Metrics

```python
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error,
    root_mean_squared_error, r2_score,
    mean_absolute_percentage_error,
    median_absolute_error
)
import numpy as np

y_pred = model.predict(X_test)

mae   = mean_absolute_error(y_test, y_pred)
mse   = mean_squared_error(y_test, y_pred)
rmse  = root_mean_squared_error(y_test, y_pred)   # sklearn 1.4+
mape  = mean_absolute_percentage_error(y_test, y_pred) * 100
r2    = r2_score(y_test, y_pred)
medae = median_absolute_error(y_test, y_pred)

# Adjusted R²
n, p  = len(y_test), X_test.shape[1]
adj_r2 = 1 - (1-r2) * (n-1) / (n-p-1)

print(f"MAE:    {mae:.4f}")
print(f"RMSE:   {rmse:.4f}")
print(f"MAPE:   {mape:.2f}%")
print(f"R²:     {r2:.4f}")
print(f"Adj R²: {adj_r2:.4f}")
```

---

## 31. Cross Validation

### CV Strategies

```python
from sklearn.model_selection import (
    cross_val_score, cross_validate, cross_val_predict,
    KFold, StratifiedKFold, RepeatedStratifiedKFold,
    GroupKFold, TimeSeriesSplit
)
import numpy as np

# Stratified KFold (classification)
skf    = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
scores = cross_val_score(model, X, y, cv=skf,
                          scoring="roc_auc", n_jobs=-1)
print(f"Mean AUC: {scores.mean():.4f} ± {scores.std():.4f}")

# Repeated Stratified (more stable)
rskf   = RepeatedStratifiedKFold(n_splits=5, n_repeats=10, random_state=42)

# Time Series Split (no future leakage)
tscv   = TimeSeriesSplit(n_splits=5, gap=0)
for fold, (train_idx, val_idx) in enumerate(tscv.split(X)):
    X_tr, X_vl = X.iloc[train_idx], X.iloc[val_idx]
    model.fit(X_tr, y.iloc[train_idx])
    score = roc_auc_score(y.iloc[val_idx],
                           model.predict_proba(X_vl)[:,1])
    print(f"Fold {fold+1}: AUC = {score:.4f}")

# Group KFold (prevent group leakage)
gkf    = GroupKFold(n_splits=5)
scores = cross_val_score(model, X, y, cv=gkf,
                          groups=df["user_id"])

# Multiple metrics at once
results = cross_validate(
    model, X, y, cv=skf,
    scoring=["roc_auc","f1","precision","recall"],
    return_train_score=True, n_jobs=-1
)

# Out-of-fold predictions (for stacking)
oof_preds = cross_val_predict(model, X, y, cv=skf,
                               method="predict_proba", n_jobs=-1)
```

### Learning Curves

```python
from sklearn.model_selection import learning_curve, validation_curve
import matplotlib.pyplot as plt, numpy as np

# Learning Curve (bias vs variance)
train_sizes, train_scores, val_scores = learning_curve(
    estimator=model, X=X, y=y,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="roc_auc",
    train_sizes=np.linspace(0.1, 1.0, 10),
    n_jobs=-1
)

train_mean = train_scores.mean(axis=1)
val_mean   = val_scores.mean(axis=1)

fig, ax = plt.subplots(figsize=(8,5))
ax.plot(train_sizes, train_mean, "o-", color="blue",   label="Train")
ax.plot(train_sizes, val_mean,   "o-", color="orange", label="Validation")
ax.fill_between(train_sizes, train_mean - train_scores.std(1),
                             train_mean + train_scores.std(1), alpha=0.2)
ax.set_xlabel("Training Size"); ax.set_ylabel("AUC"); ax.legend()
# High train, low val → overfitting. Low both → underfitting.

# Validation Curve (tune one hyperparameter)
param_range = [1, 2, 5, 10, 20, 50]
train_sc, val_sc = validation_curve(
    DecisionTreeClassifier(), X, y,
    param_name="max_depth",
    param_range=param_range,
    cv=5, scoring="roc_auc"
)
plt.plot(param_range, train_sc.mean(1), label="Train")
plt.plot(param_range, val_sc.mean(1),   label="Val")
```

---

## 32. Hyperparameter Tuning

### Grid Search & Random Search

```python
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV
from scipy.stats import randint, uniform, loguniform
from sklearn.ensemble import RandomForestClassifier

# Grid Search (exhaustive)
param_grid = {
    "n_estimators":   [100, 200, 300],
    "max_depth":      [None, 5, 10, 20],
    "min_samples_leaf": [1, 5, 10],
    "max_features":   ["sqrt", "log2"],
}
grid_search = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=param_grid,
    cv=StratifiedKFold(5, shuffle=True, random_state=42),
    scoring="roc_auc", n_jobs=-1, verbose=1,
    return_train_score=True
)
grid_search.fit(X_train, y_train)
print(f"Best params: {grid_search.best_params_}")
print(f"Best score:  {grid_search.best_score_:.4f}")
best_model = grid_search.best_estimator_

# Random Search (more efficient)
param_dist = {
    "n_estimators":  randint(100, 1000),
    "max_depth":     [None, 5, 10, 20, 30],
    "min_samples_leaf": randint(1, 20),
    "max_features":  uniform(0.1, 0.9),
}
rand_search = RandomizedSearchCV(
    RandomForestClassifier(random_state=42),
    param_distributions=param_dist,
    n_iter=100, cv=5, scoring="roc_auc",
    n_jobs=-1, verbose=1, random_state=42
)
rand_search.fit(X_train, y_train)
```

### Optuna — Bayesian Optimization

```python
import optuna
from sklearn.model_selection import cross_val_score, StratifiedKFold

optuna.logging.set_verbosity(optuna.logging.WARNING)

def objective(trial):
    params = {
        "n_estimators":    trial.suggest_int("n_estimators", 50, 1000),
        "max_depth":       trial.suggest_int("max_depth", 3, 10),
        "learning_rate":   trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample":       trial.suggest_float("subsample", 0.5, 1.0),
        "max_features":    trial.suggest_float("max_features", 0.3, 1.0),
        "min_samples_leaf":trial.suggest_int("min_samples_leaf", 1, 30),
    }
    model  = GradientBoostingClassifier(**params, random_state=42)
    cv     = StratifiedKFold(5, shuffle=True, random_state=42)
    scores = cross_val_score(model, X_train, y_train,
                              cv=cv, scoring="roc_auc", n_jobs=-1)
    return scores.mean()

study = optuna.create_study(direction="maximize",
                              sampler=optuna.samplers.TPESampler(seed=42))
study.optimize(objective, n_trials=100)

print(f"Best AUC:    {study.best_value:.4f}")
print(f"Best params: {study.best_params}")

# Visualization
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

---

## 33. SHAP — Model Explainability

```python
import shap
import numpy as np, pandas as pd
import matplotlib.pyplot as plt

# --- Tree models (fast) ---
explainer   = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# For binary classification, shap_values is a list [cls0, cls1]
sv = shap_values[1] if isinstance(shap_values, list) else shap_values

# Summary (global feature importance)
shap.summary_plot(sv, X_test, feature_names=feat_names)

# Bar plot (mean absolute SHAP)
shap.summary_plot(sv, X_test, plot_type="bar")

# Dependence plot (feature effect)
shap.dependence_plot("income", sv, X_test)

# Force plot (single prediction)
shap.force_plot(explainer.expected_value[1], sv[0], X_test.iloc[0])

# Waterfall plot
shap.plots.waterfall(explainer(X_test.iloc[[0]])[..., 1])

# --- Kernel SHAP (any model, slower) ---
background    = shap.sample(X_train, 100)
explainer_k   = shap.KernelExplainer(model.predict_proba, background)
sv_k          = explainer_k.shap_values(X_test[:10], nsamples=200)

# Feature importance from SHAP
shap_imp = pd.Series(
    np.abs(sv).mean(0), index=feat_names
).sort_values(ascending=False)
print(shap_imp.head(20))
```

---

## 34. Saving & Loading Models

### Joblib, Pickle & Native Formats

```python
import joblib, pickle

# Joblib (recommended for sklearn + numpy)
joblib.dump(model, "model.joblib")
joblib.dump(model, "model.joblib", compress=3)  # smaller file
model = joblib.load("model.joblib")

# Pickle
with open("model.pkl", "wb") as f:
    pickle.dump(model, f)
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

# Save full pipeline
joblib.dump(full_pipe, "pipeline.joblib")

# XGBoost native
xgb_model.save_model("xgb_model.json")
xgb_model = xgb.XGBClassifier()
xgb_model.load_model("xgb_model.json")

# LightGBM native
lgb_model.save_model("lgbm_model.txt")
lgbm = lgb.Booster(model_file="lgbm_model.txt")
```

### MLflow Experiment Tracking

```python
import mlflow

mlflow.set_experiment("my_experiment")
with mlflow.start_run():
    mlflow.log_params({"lr": 0.05, "depth": 6})
    mlflow.log_metric("roc_auc", 0.91)
    mlflow.sklearn.log_model(model, "model")
    mlflow.log_artifact("eda_report.html")
```

### FastAPI Model Serving

```python
# pip install fastapi uvicorn joblib pydantic
from fastapi import FastAPI
from pydantic import BaseModel
import joblib, pandas as pd

app   = FastAPI(title="ML Model API")
model = joblib.load("pipeline.joblib")

class Features(BaseModel):
    age: float
    income: float
    city: str
    occupation: str

class Prediction(BaseModel):
    prediction: int
    probability: float

@app.post("/predict", response_model=Prediction)
def predict(features: Features):
    data = pd.DataFrame([features.model_dump()])
    pred = model.predict(data)[0]
    prob = model.predict_proba(data)[0, 1]
    return Prediction(prediction=int(pred), probability=float(prob))

@app.get("/health")
def health():
    return {"status": "ok"}

# Run: uvicorn app:app --host 0.0.0.0 --port 8000
# Docs: http://localhost:8000/docs
```

---

## 35. Pandas Power Operations

### Merging, Reshaping & GroupBy

```python
import pandas as pd, numpy as np

# --- Merge / Join ---
pd.merge(df1, df2, on="id", how="left")           # left, right, inner, outer
pd.merge(df1, df2, left_on="uid", right_on="user_id")
pd.merge(df1, df2, on=["id","date"])               # multi-key
df.join(df2, how="left", rsuffix="_r")             # index-based

# --- Concat ---
pd.concat([df1, df2], axis=0, ignore_index=True)  # rows
pd.concat([df1, df2], axis=1)                      # columns

# --- Pivot & Melt ---
wide = df.pivot(index="date", columns="category", values="sales")
wide = df.pivot_table(index="date", columns="cat",
                       values="sales", aggfunc="sum")
long = wide.reset_index().melt(id_vars="date",
                                var_name="category",
                                value_name="sales")

# --- Apply / Map / Transform ---
df["new"]    = df["col"].apply(lambda x: x**2)
df["new"]    = df["col"].map({"A":1,"B":2})          # dict mapping
df["scaled"] = df.groupby("g")["v"].transform("mean") # group-aware

# --- Window Functions ---
df.groupby("category")["sales"].rolling(7).mean()
df.groupby("category")["sales"].expanding().sum()
df.groupby("category")["sales"].pct_change()
```

### Performance & Large Data

```python
import pandas as pd, numpy as np

# Vectorized ops (fast)
# BAD:  df.apply(lambda x: x["a"] + x["b"], axis=1)
# GOOD: df["a"] + df["b"]
# BAD:  for i, row in df.iterrows(): ...
# GOOD: vectorized or df.apply with simple function

# Filter with query (faster)
df.query("age > 30 & income < 100000")
df.query("city in @city_list")   # use variable with @

# Memory optimization with categoricals
for col in df.select_dtypes("object"):
    if df[col].nunique() / len(df) < 0.5:
        df[col] = df[col].astype("category")

# Chunking large files
results = []
for chunk in pd.read_csv("huge.csv", chunksize=100_000):
    results.append(chunk.groupby("category")["val"].sum())
final = pd.concat(results).groupby(level=0).sum()

# Dask (parallel pandas)
import dask.dataframe as dd
ddf    = dd.from_pandas(df, npartitions=4)
result = ddf.groupby("cat")["val"].mean().compute()

# Polars (ultra-fast) — pip install polars
import polars as pl
df_pl  = pl.read_csv("data.csv")
result = (df_pl
    .filter(pl.col("age") > 30)
    .group_by("city")
    .agg(pl.col("salary").mean().alias("avg_salary"))
    .sort("avg_salary", descending=True))

df_pl = pl.from_pandas(df_pd)   # convert to polars
df_pd = df_pl.to_pandas()        # convert back
```

---

## 36. NumPy Quick Reference

### Array Creation & Operations

```python
import numpy as np

# --- Creation ---
np.array([1,2,3])
np.zeros((3,4));  np.ones((3,4));  np.full((3,4), 5)
np.eye(4)                            # identity matrix
np.arange(0, 10, 0.5)               # start, stop, step
np.linspace(0, 1, 100)              # evenly spaced
np.random.seed(42)
np.random.rand(3,4)                  # uniform [0,1]
np.random.randn(3,4)                 # standard normal
np.random.randint(0, 10, size=(3,4))
np.random.choice(arr, size=10, replace=False)

# --- Shape ---
a.shape;  a.ndim;  a.size;  a.dtype
a.reshape(2,6);  a.ravel();  a.flatten()
a.T                                  # transpose
np.squeeze(a, axis=1)
np.expand_dims(a, axis=0)
np.concatenate([a,b], axis=0)
np.vstack([a,b]);  np.hstack([a,b])
np.split(a, 3, axis=0)

# --- Math ---
np.sum(a, axis=0);  np.mean(a);  np.std(a)
np.min(a, axis=1);  np.max(a);  np.cumsum(a)
np.dot(a, b);  a @ b                 # matrix multiplication
np.linalg.norm(a)                    # vector norm
np.linalg.inv(A)                     # matrix inverse
np.linalg.eig(A)                     # eigenvalues/vectors
np.linalg.svd(A)                     # SVD

# --- Boolean indexing ---
a[a > 0]
a[(a > 0) & (a < 10)]
np.where(a > 0, a, 0)               # conditional select
np.clip(a, 0, 1)                    # clamp values

# --- Vectorized functions ---
np.log(a);  np.exp(a);  np.sqrt(a)
np.abs(a);  np.sign(a)
np.floor(a);  np.ceil(a);  np.round(a, 2)
np.maximum(a, b);  np.minimum(a, b) # element-wise

# --- Memory ---
a = np.float32(a)                   # save memory vs float64
a.nbytes / 1e6                      # size in MB
```

---

## 37. Imbalanced Data Toolkit

```python
# pip install imbalanced-learn
from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
from imblearn.under_sampling import RandomUnderSampler, TomekLinks
from imblearn.combine import SMOTETomek, SMOTEENN
from imblearn.pipeline import Pipeline as ImbPipeline

# Applying Random Under Sampling
from imblearn.under_sampling import RandomUnderSampler

rus = RandomUnderSampler(random_state=42)
X_resampled, y_resampled = rus.fit_resample(X_train, y_train)

# Applying Random Over Sampling
from imblearn.over_sampling import RandomOverSampler

ros = RandomOverSampler(random_state=42)
X_resampled, y_resampled = ros.fit_resample(X_train, y_train)

# Applying Ensemble Method/ Balanced Random Forest (Recomanded)
from imblearn.ensemble import BalancedRandomForestClassifier

classifier = BalancedRandomForestClassifier(random_state=42)
classifier.fit(X_train, y_train)

# Cost Sensitive Learning/ Class Weights 
# Create a logistic regression model with class weights
model = LogisticRegression(class_weight={0:50,1:1}, solver='liblinear')   # Can take other models also

# Train the model
model.fit(X_train, y_train)

# SMOTE — synthetic minority oversampling (Recomanded)
sm = SMOTE(sampling_strategy="auto", k_neighbors=5, random_state=42)
X_res, y_res = sm.fit_resample(X_train, y_train)

# Combined over+under sampling
smt = SMOTETomek(random_state=42)
X_res, y_res = smt.fit_resample(X_train, y_train)

# In a pipeline (avoids leakage)
pipe = ImbPipeline([
    ("smote",  SMOTE(random_state=42)),
    ("scaler", StandardScaler()),
    ("model",  RandomForestClassifier()),
])
pipe.fit(X_train, y_train)

# Class weights (no resampling needed)
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
weights      = compute_class_weight("balanced",
                                    classes=np.unique(y), y=y)
class_weight = dict(zip(np.unique(y), weights))
```

### Probability Calibration

```python
from sklearn.calibration import CalibratedClassifierCV, calibration_curve
import matplotlib.pyplot as plt

# Calibrate probabilities
calibrated = CalibratedClassifierCV(
    estimator=model,
    method="sigmoid",  # platt scaling, or 'isotonic'
    cv=5
)
calibrated.fit(X_train, y_train)
proba_cal = calibrated.predict_proba(X_test)[:,1]

# Reliability diagram
frac_pos, mean_pred = calibration_curve(
    y_test, y_prob, n_bins=10, strategy="uniform")
plt.plot(mean_pred, frac_pos, "s-", label="Model")
plt.plot([0,1],[0,1], "k--", label="Perfect")
plt.xlabel("Mean Predicted Probability")
plt.ylabel("Fraction of Positives")

# Brier Score (lower=better, 0=perfect)
from sklearn.metrics import brier_score_loss
bs     = brier_score_loss(y_test, y_prob)
bs_cal = brier_score_loss(y_test, proba_cal)
print(f"Brier (uncal): {bs:.4f}")
print(f"Brier (cal):   {bs_cal:.4f}")
```

---

## 38. Algorithm Selection Guide

| Task | Algorithm | When to Use | Avoid When |
|------|-----------|-------------|------------|
| Binary Classification | `LogisticRegression` | Linear boundary, needs probabilities, baseline | Complex non-linear patterns |
| Binary / Multi-class | `RandomForest` | Robust baseline, no scaling needed, handles mixed types | Need fast inference, huge feature space |
| Any Tabular Task | `XGBoost / LightGBM` | Most competition-winning, regularized, handles missing | Very small datasets (<100 rows), need interpretability |
| Regression | `Ridge / Lasso` | Linear relationships, many features, regularization | Highly non-linear data |
| Text Classification | `TF-IDF + LogReg` | Fast baseline, sparse features | Semantic understanding required |
| Image / NLP | `Transformers` | Unstructured data, transfer learning | Small data without pretrained model |
| Clustering (known k) | `KMeans` | Spherical clusters, large data, fast | Arbitrary shapes, unknown k, outliers |
| Clustering (unknown k) | `HDBSCAN / DBSCAN` | Arbitrary shapes, noisy data | Very high dimensions, huge datasets |
| Anomaly Detection | `IsolationForest` | Multivariate, fast, scalable | Very low contamination |
| Time Series | `Prophet / SARIMA` | Seasonal data, trend, holidays | Very long series (use LSTM/TFT) |
| Dimensionality Reduction | `PCA` | Linear structure, preprocessing, speed | Non-linear manifolds (use UMAP) |
| Visualization | `t-SNE / UMAP` | Exploring cluster structure visually | ML preprocessing, large datasets (t-SNE) |
| Imbalanced Data | `class_weight="balanced"` | Easiest fix, no data modification | Extreme imbalance >100:1 (use SMOTE) |
| Feature Selection | `RFECV / LassoCV` | Automatic, cross-validated | Very many features (use importances first) |

---

## 39. Environment Setup

```bash
# Create virtual environment
python -m venv ds_env
source ds_env/bin/activate      # Linux/Mac
ds_env\Scripts\activate         # Windows

# Core data science stack
pip install numpy pandas scikit-learn matplotlib seaborn plotly

# Gradient boosting
pip install xgboost lightgbm catboost

# Bayesian optimization
pip install optuna hyperopt

# Deep Learning
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install tensorflow

# NLP
pip install transformers sentence-transformers datasets tokenizers
pip install spacy nltk gensim
python -m spacy download en_core_web_sm

# AutoML / Profiling
pip install ydata-profiling sweetviz dtale
pip install imbalanced-learn category_encoders shap

# Big data / performance
pip install polars dask pyarrow fastparquet

# Statistical modeling
pip install statsmodels pingouin scipy pmdarima prophet

# Serving / tracking
pip install fastapi uvicorn mlflow streamlit gradio

# Jupyter
pip install jupyterlab ipywidgets

# Save environment
pip freeze > requirements.txt

# Restore environment
pip install -r requirements.txt
```

---

> **Tip:** Press `Ctrl+F` in your editor or viewer to search any keyword (e.g., `IsolationForest`, `cross_val_score`, `pd.read_excel`) and jump directly to the right snippet.

*Data Science Handbook — Python 3.10+*
