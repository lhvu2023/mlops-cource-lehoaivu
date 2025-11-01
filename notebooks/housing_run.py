import os
import logging
from pathlib import Path
import pandas as pd

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s - %(message)s",
)
logger = logging.getLogger("housing")

# Paths
PROJECT_ROOT = Path("..")
DATA_PATH = PROJECT_ROOT / "data" / "housing.csv"
ARTIFACT_DIR = PROJECT_ROOT / "scripts" / "session_1"
ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
MODEL_PATH = ARTIFACT_DIR / "housing_linear.joblib"

logger.info(f"Data path: {DATA_PATH}")
logger.info(f"Artifact dir: {ARTIFACT_DIR}")

logger.info("Loading dataset...")
df = pd.read_csv(DATA_PATH)
logger.info(f"Loaded {len(df)} rows and {len(df.columns)} columns")

df.head()

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import numpy as np

logger.info("Preparing features and target...")
# Identify target and basic features from the CSV header
TARGET = "Price"
ALL_COLUMNS = df.columns.tolist()
NUM_FEATURES = [
    "Avg. Area Income",
    "Avg. Area House Age",
    "Avg. Area Number of Rooms",
    "Avg. Area Number of Bedrooms",
    "Area Population",
]
CAT_FEATURES = [
    # 'Address' exists but is high-cardinality; we'll drop it for a simple baseline
]

X = df[NUM_FEATURES]
y = df[TARGET]

logger.info("Splitting train/test...")
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

logger.info("Building pipeline...")
preprocessor = ColumnTransformer(
    transformers=[
        ("num", StandardScaler(), NUM_FEATURES),
        # ("cat", OneHotEncoder(handle_unknown="ignore"), CAT_FEATURES),
    ],
    remainder="drop",
)

model = Pipeline(
    steps=[
        ("preprocess", preprocessor),
        ("regressor", SGDRegressor(
            max_iter=5000,
            tol=1e-4,
            learning_rate="optimal",
            random_state=42,
            verbose=1
        )),
    ]
)

logger.info("Training model...")
model.fit(X_train, y_train)

logger.info("Evaluating model...")
y_pred = model.predict(X_test)
rmse = np.sqrt(mean_squared_error(y_test, y_pred))
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
logger.info(f"RMSE: {rmse:.2f} | MAE: {mae:.2f} | R2: {r2:.4f}")

rmse, mae, r2
