"""
FE_Pipeline.py — Feature Engineering Pipeline for Dormancy Fraud Detection

Transforms raw CSV into a clean, model-ready feature matrix.

Steps
-----
1.  Load & drop leaky/post-event columns
2.  Parse timestamp, extract temporal features
3.  Dormancy features  (is_first_transaction, dormancy_bucket, etc.)
4.  IP frequency feature
5.  Amount log-transform
6.  Encode categoricals (label encode low-card, frequency encode high-card)
7.  Scale numeric features (RobustScaler)
8.  Time-based train/test split
9.  Save outputs

Leakage safeguards
------------------
- fraud_type dropped on load (post-event label)
- is_fraud only used as y, never enters X
- timestamp dropped from X after all features extracted
- transaction_id dropped from X (pure identifier)
"""

import sys
import io
import logging
import joblib
import numpy as np
import pandas as pd
import gc
from collections import Counter
from pathlib import Path
from typing import Tuple, Dict, Any

from sklearn.preprocessing import LabelEncoder, RobustScaler

from FE_config import FEConfig

# Force UTF-8 stdout on Windows
if hasattr(sys.stdout, 'buffer'):
    sys.stdout = io.TextIOWrapper(
        sys.stdout.buffer, encoding='utf-8', errors='replace', line_buffering=True
    )


# ==============================================================================
# LOGGING
# ==============================================================================

def setup_logging(config: FEConfig) -> logging.Logger:
    logger = logging.getLogger('FE_Pipeline')
    logger.setLevel(logging.INFO)
    logger.handlers = []
    fmt = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')

    ch = logging.StreamHandler(sys.stdout)
    ch.setFormatter(fmt)
    logger.addHandler(ch)

    fh = logging.FileHandler(config.LOG_FILE, encoding='utf-8')
    fh.setFormatter(fmt)
    logger.addHandler(fh)

    return logger


# ==============================================================================
# FEATURE ENGINEERING PIPELINE
# ==============================================================================

class FeatureEngineeringPipeline:
    """
    End-to-end feature engineering pipeline for dormancy fraud detection.
    Produces a leak-free, model-ready feature matrix from the raw CSV.
    """

    def __init__(self, config: FEConfig = None):
        self.config   = config or FEConfig()
        self.config.validate()
        self.logger   = setup_logging(self.config)
        self.df       = None
        self.encoders: Dict[str, Any] = {}
        self.scaler   = None
        self.is_sampled = False

        Path("features").mkdir(exist_ok=True)
        self.logger.info("FeatureEngineeringPipeline initialised")

    # -------------------------------------------------------------------------
    # STEP 1 — LOAD
    # -------------------------------------------------------------------------

    def load(self) -> pd.DataFrame:
        """Load CSV and immediately drop post-event leaky columns."""
        self.logger.info(f"Loading: {self.config.INPUT_CSV}")
        try:
            self.df = self._read_csv_full()
            self.logger.info(f"Loaded: {len(self.df):,} rows x {len(self.df.columns)} cols")
        except MemoryError:
            if not self.config.MEMORY_SAFE_MODE:
                self.logger.error(
                    "MemoryError and MEMORY_SAFE_MODE is disabled. "
                    "Enable MEMORY_SAFE_MODE in FE_Config.py to use chunked fallback."
                )
                raise
            self.logger.warning(
                "MemoryError during full CSV read. Switching to memory-safe chunked sampling "
                f"(up to {self.config.FALLBACK_SAMPLE_ROWS:,} rows)."
            )
            self.df = self._read_csv_chunked_sampled(self.config.FALLBACK_SAMPLE_ROWS)
            self.is_sampled = True
            self.logger.info(
                f"Fallback load complete: {len(self.df):,} sampled rows x {len(self.df.columns)} cols"
            )

        # Free any intermediate memory
        gc.collect()

        # PRIMARY LEAKAGE SAFEGUARD
        to_drop = [c for c in self.config.DROP_ON_LOAD if c in self.df.columns]
        if to_drop:
            self.df.drop(columns=to_drop, inplace=True)
            self.logger.info(f"[LEAKAGE GUARD] Dropped on load: {to_drop}")

        return self.df

    def _dtype_map(self) -> Dict[str, Any]:
        return {
            'transaction_id': 'string',
            'sender_account': 'string',
            'receiver_account': 'string',
            'device_hash': 'string',
            'ip_address': 'string',
            'merchant_category': 'category',
            'location': 'category',
            'device_used': 'category',
            'payment_channel': 'category',
            'transaction_type': 'category',
            'amount': 'float32',
            'time_since_last_transaction': 'float32',
            'spending_deviation_score': 'float32',
            'velocity_score': 'float32',
            'geo_anomaly_score': 'float32',
            self.config.TARGET_COLUMN: 'int8',
        }

    def _read_csv_full(self) -> pd.DataFrame:
        return pd.read_csv(
            self.config.INPUT_CSV,
            dtype=self._dtype_map(),
            parse_dates=['timestamp'],
            memory_map=True,
            low_memory=False,
        )

    def _read_csv_chunked_sampled(self, max_rows: int) -> pd.DataFrame:
        """Read CSV in chunks and keep a bounded stratified sample."""
        chunks = []
        rows_kept = 0
        target = self.config.TARGET_COLUMN
        seed = self.config.RANDOM_STATE

        for idx, chunk in enumerate(
            pd.read_csv(
                self.config.INPUT_CSV,
                dtype=self._dtype_map(),
                parse_dates=['timestamp'],
                chunksize=self.config.CHUNK_SIZE,
                low_memory=False,
            )
        ):
            if rows_kept >= max_rows:
                break

            remaining = max_rows - rows_kept
            if len(chunk) > remaining:
                frac = remaining / len(chunk)
                if target in chunk.columns:
                    sampled = (
                        chunk.groupby(target, group_keys=False)
                             .apply(lambda x: x.sample(frac=frac, random_state=seed + idx))
                    )
                    if len(sampled) > remaining:
                        sampled = sampled.sample(n=remaining, random_state=seed + idx)
                    chunk = sampled.reset_index(drop=True)
                else:
                    chunk = chunk.sample(n=remaining, random_state=seed + idx)

            chunks.append(chunk)
            rows_kept += len(chunk)

            if (idx + 1) % 10 == 0:
                self.logger.info(
                    f"Chunked fallback progress: {rows_kept:,}/{max_rows:,} rows kept"
                )

        if not chunks:
            raise MemoryError("Chunked fallback could not load any rows from CSV.")

        return pd.concat(chunks, ignore_index=True)

    # -------------------------------------------------------------------------
    # STEP 2 — TEMPORAL FEATURES
    # -------------------------------------------------------------------------

    def build_temporal_features(self) -> pd.DataFrame:
        """
        Parse timestamp and extract dormancy-relevant temporal features.
        Raw timestamp string is kept temporarily for sort-based train/test split,
        then dropped before the feature matrix is finalised.
        """
        self.logger.info("Building temporal features...")
        self.df['timestamp'] = pd.to_datetime(self.df['timestamp'], format='ISO8601')

        # Basic temporal decomposition
        self.df['hour']        = self.df['timestamp'].dt.hour
        self.df['day_of_week'] = self.df['timestamp'].dt.dayofweek
        self.df['month']       = self.df['timestamp'].dt.month
        self.df['is_weekend']  = self.df['day_of_week'].isin(self.config.WEEKEND_DAYS).astype(int)

        # High-risk hour flag (late night / early morning — common for dormancy fraud)
        self.df['is_high_risk_hour'] = self.df['hour'].isin(
            self.config.HIGH_RISK_HOURS
        ).astype(int)

        self.logger.info(
            "Temporal features created: hour, day_of_week, month, "
            "is_weekend, is_high_risk_hour"
        )
        return self.df

    # -------------------------------------------------------------------------
    # STEP 3 — DORMANCY FEATURES
    # -------------------------------------------------------------------------

    def build_dormancy_features(self) -> pd.DataFrame:
        """
        Engineer dormancy-specific features from time_since_last_transaction.

        is_first_transaction
            Binary flag: 1 if time_since_last_transaction is null.
            Null means no prior transaction exists for this account —
            it is the account's very first transaction.
            Must be created BEFORE imputation so the signal is not lost.

        dormancy_bucket
            Categorical bucket of time_since_last_transaction:
            recent / moderate / dormant / long_dormant.
            Captures non-linear dormancy risk thresholds.

        dormancy_risk_score
            Numeric: 0 (recent) to 3 (long_dormant).
            Ordinal encoding of dormancy_bucket for tree-based models.

        time_since_last_transaction (imputed)
            After flagging, nulls imputed with the non-fraud median.
            Using non-fraud median avoids introducing fraud-class signal
            into the imputed values.
        """
        self.logger.info("Building dormancy features...")
        col = self.config.DORMANCY_IMPUTE_COL

        # --- is_first_transaction flag (MUST come before imputation) ----------
        self.df[self.config.DORMANCY_NULL_FLAG_COL] = (
            self.df[col].isnull().astype(int)
        )
        n_first = self.df[self.config.DORMANCY_NULL_FLAG_COL].sum()
        self.logger.info(
            f"is_first_transaction: {n_first:,} accounts "
            f"({n_first/len(self.df)*100:.2f}% of transactions)"
        )

        # --- Impute with non-fraud median (safe imputation) -------------------
        non_fraud_median = (
            self.df.loc[self.df[self.config.TARGET_COLUMN] == False, col]
            .median()
        )
        self.df[col] = self.df[col].fillna(non_fraud_median)
        self.logger.info(
            f"Imputed {col} nulls with non-fraud median: {non_fraud_median:.4f}"
        )

        # --- Dormancy bucket --------------------------------------------------
        self.df['dormancy_bucket'] = pd.cut(
            self.df[col],
            bins=self.config.DORMANCY_BINS,
            labels=self.config.DORMANCY_LABELS,
            right=False,
        ).astype(str)

        # --- Ordinal dormancy risk score (0=recent, 3=long_dormant) ----------
        risk_map = {
            'recent':       0,
            'moderate':     1,
            'dormant':      2,
            'long_dormant': 3,
        }
        self.df['dormancy_risk_score'] = (
            self.df['dormancy_bucket'].map(risk_map).fillna(0).astype(int)
        )

        bucket_counts = self.df['dormancy_bucket'].value_counts()
        self.logger.info(f"Dormancy bucket distribution:\n{bucket_counts}")

        return self.df

    # -------------------------------------------------------------------------
    # STEP 4 — IP ADDRESS FEATURE
    # -------------------------------------------------------------------------

    def build_ip_features(self) -> pd.DataFrame:
        """
        Frequency-encode ip_address.

        Rather than label-encoding (which assigns arbitrary order to IPs),
        we compute how often each IP appears in the dataset.
        A very low frequency IP on a dormant account is a strong fraud signal.
        The raw ip_address string is then dropped from X.
        """
        self.logger.info("Building IP frequency feature...")
        ip_freq = self.df['ip_address'].value_counts()
        self.df[self.config.IP_FREQ_COL] = (
            self.df['ip_address'].map(ip_freq).fillna(1).astype(int)
        )
        self.logger.info(
            f"IP frequency range: {self.df[self.config.IP_FREQ_COL].min()} "
            f"to {self.df[self.config.IP_FREQ_COL].max()}"
        )
        return self.df

    # -------------------------------------------------------------------------
    # STEP 5 — AMOUNT FEATURES
    # -------------------------------------------------------------------------

    def build_amount_features(self) -> pd.DataFrame:
        """
        Log-transform amount to reduce right skew (EDA: skewness=1.65).
        log1p used so amount=0 doesn't produce -inf.
        Both raw and log amount are kept — tree models can use either.
        """
        self.logger.info("Building amount features...")
        self.df['amount_log'] = np.log1p(self.df['amount'])
        self.logger.info(
            f"amount_log: mean={self.df['amount_log'].mean():.4f}, "
            f"skew={self.df['amount_log'].skew():.4f}"
        )
        return self.df

    # -------------------------------------------------------------------------
    # STEP 6 — ENCODE CATEGORICALS
    # -------------------------------------------------------------------------

    def encode_categoricals(self) -> pd.DataFrame:
        """
        Low-cardinality columns  (<= HIGH_CARDINALITY_THRESHOLD unique values)
            -> LabelEncoder  (integer codes, compact, works well with trees)

        High-cardinality columns (>  HIGH_CARDINALITY_THRESHOLD unique values)
            -> Frequency encoding (replace value with its count in the dataset)
               Better than label encoding for high-card because it preserves
               meaningful signal (rare accounts/devices = higher fraud risk).

        ip_address is already frequency-encoded in build_ip_features(),
        so it is excluded here.

        Encoders are stored in self.encoders for use at inference time.
        """
        self.logger.info("Encoding categorical features...")

        # Low-cardinality: LabelEncoder
        for col in self.config.LOW_CARDINALITY_COLS:
            if col not in self.df.columns:
                continue
            le = LabelEncoder()
            self.df[f'{col}_enc'] = le.fit_transform(self.df[col].astype(str))
            self.encoders[col] = le
            self.logger.info(
                f"Label encoded '{col}': {len(le.classes_)} classes"
            )

        # High-cardinality: Frequency encoding (exclude ip_address — done above)
        high_card = [
            c for c in self.config.HIGH_CARDINALITY_COLS
            if c != 'ip_address' and c in self.df.columns
        ]
        for col in high_card:
            freq_map = self.df[col].value_counts().to_dict()
            self.df[f'{col}_freq'] = (
                self.df[col].map(freq_map).fillna(1).astype(int)
            )
            self.encoders[f'{col}_freq_map'] = freq_map
            self.logger.info(
                f"Frequency encoded '{col}': "
                f"{self.df[col].nunique():,} unique values"
            )

        # dormancy_bucket is a string — label encode it
        le_bucket = LabelEncoder()
        self.df['dormancy_bucket_enc'] = le_bucket.fit_transform(
            self.df['dormancy_bucket'].astype(str)
        )
        self.encoders['dormancy_bucket'] = le_bucket

        return self.df

    # -------------------------------------------------------------------------
    # STEP 7 — BUILD FEATURE MATRIX
    # -------------------------------------------------------------------------

    def build_feature_matrix(self) -> Tuple[pd.DataFrame, pd.Series]:
        """
        Assemble the final leak-free feature matrix X and target y.

        Dropped from X:
            - transaction_id   (identifier)
            - timestamp        (raw string; all signal already extracted)
            - ip_address       (raw string; signal captured in ip_freq)
            - sender_account   (raw string; signal captured in sender_account_freq)
            - receiver_account (raw string; signal captured in receiver_account_freq)
            - device_hash      (raw string; signal captured in device_hash_freq)
            - dormancy_bucket  (raw string; signal captured in dormancy_bucket_enc)
            - is_fraud         (target)
            - any other LEAKY_COLUMNS
        """
        self.logger.info("Building feature matrix...")

        # Columns to drop from X
        raw_string_cols = [
            'ip_address', 'sender_account', 'receiver_account',
            'device_hash', 'dormancy_bucket',
        ]
        drop_cols = (
            self.config.DROP_FROM_X
            + raw_string_cols
            + self.config.LEAKY_COLUMNS
        )

        X = self.df.drop(
            columns=[c for c in drop_cols if c in self.df.columns]
        )
        y = self.df[self.config.TARGET_COLUMN].astype(int)

        self.logger.info(
            f"Feature matrix: {X.shape[0]:,} rows x {X.shape[1]} features"
        )
        self.logger.info(f"Features: {list(X.columns)}")
        self.logger.info(
            f"Class balance: {y.sum():,} fraud ({y.mean()*100:.2f}%)"
        )

        return X, y

    # -------------------------------------------------------------------------
    # STEP 8 — SCALE
    # -------------------------------------------------------------------------

    def scale_features(self, X: pd.DataFrame) -> pd.DataFrame:
        """
        Apply RobustScaler to numeric columns.
        RobustScaler uses median + IQR so outliers (8.25% in amount) 
        don't distort the scale the way StandardScaler would.
        Binary flags and encoded categoricals are not scaled.
        """
        if not self.config.SCALE_FEATURES:
            return X

        self.logger.info("Scaling numeric features with RobustScaler...")
        cols_to_scale = [
            c for c in self.config.COLS_TO_SCALE if c in X.columns
        ]

        self.scaler = RobustScaler()
        X = X.copy()
        X[cols_to_scale] = self.scaler.fit_transform(X[cols_to_scale])
        self.logger.info(f"Scaled columns: {cols_to_scale}")
        return X

    # -------------------------------------------------------------------------
    # STEP 9 — TRAIN / TEST SPLIT
    # -------------------------------------------------------------------------

    def split(
        self, X: pd.DataFrame, y: pd.Series
    ) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """
        Time-based train/test split.

        Random splitting is WRONG for this dataset because:
        - Transactions are time-ordered
        - Random split would allow future transactions to train the model
          and past transactions to be in the test set (temporal leakage)

        We sort by timestamp, take first 80% as train, last 20% as test.
        This mirrors real deployment: train on past, evaluate on future.
        """
        self.logger.info("Performing time-based train/test split...")

        # Sort by timestamp (still in self.df at this point)
        sorted_idx = self.df['timestamp'].argsort()
        X = X.iloc[sorted_idx].reset_index(drop=True)
        y = y.iloc[sorted_idx].reset_index(drop=True)

        split_idx = int(len(X) * (1 - self.config.TEST_SIZE))
        X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
        y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

        self.logger.info(
            f"Train: {len(X_train):,} rows "
            f"({y_train.sum():,} fraud, {y_train.mean()*100:.2f}%)"
        )
        self.logger.info(
            f"Test:  {len(X_test):,} rows  "
            f"({y_test.sum():,} fraud, {y_test.mean()*100:.2f}%)"
        )
        return X_train, X_test, y_train, y_test

    # -------------------------------------------------------------------------
    # STEP 10 — SAVE
    # -------------------------------------------------------------------------

    def save(
        self,
        X_train: pd.DataFrame,
        X_test: pd.DataFrame,
        y_train: pd.Series,
        y_test: pd.Series,
    ):
        """Save all outputs to the features/ directory."""
        self.logger.info("Saving outputs...")

        X_train.to_parquet(self.config.OUTPUT_X_TRAIN, index=False)
        X_test.to_parquet(self.config.OUTPUT_X_TEST,  index=False)
        y_train.to_frame().to_parquet(self.config.OUTPUT_Y_TRAIN, index=False)
        y_test.to_frame().to_parquet(self.config.OUTPUT_Y_TEST,   index=False)

        joblib.dump(self.encoders, self.config.OUTPUT_ENCODERS)
        if self.scaler:
            joblib.dump(self.scaler, self.config.OUTPUT_SCALER)

        self.logger.info(f"X_train saved : {self.config.OUTPUT_X_TRAIN}")
        self.logger.info(f"X_test  saved : {self.config.OUTPUT_X_TEST}")
        self.logger.info(f"y_train saved : {self.config.OUTPUT_Y_TRAIN}")
        self.logger.info(f"y_test  saved : {self.config.OUTPUT_Y_TEST}")
        self.logger.info(f"Encoders saved: {self.config.OUTPUT_ENCODERS}")
        if self.scaler:
            self.logger.info(f"Scaler  saved : {self.config.OUTPUT_SCALER}")

    # -------------------------------------------------------------------------
    # ORCHESTRATOR
    # -------------------------------------------------------------------------

    def run(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Run the full feature engineering pipeline end to end."""
        self.logger.info("=" * 80)
        self.logger.info("FEATURE ENGINEERING PIPELINE — START")
        self.logger.info("=" * 80)

        self.load()
        if self.is_sampled:
            self.logger.warning(
                "Pipeline running on sampled data due to memory fallback. "
                "Outputs are suitable for development/training iteration, not final benchmarking."
            )
        self.build_temporal_features()
        self.build_dormancy_features()
        self.build_ip_features()
        self.build_amount_features()
        self.encode_categoricals()

        X, y = self.build_feature_matrix()
        X    = self.scale_features(X)

        X_train, X_test, y_train, y_test = self.split(X, y)
        self.save(X_train, X_test, y_train, y_test)

        self.logger.info("=" * 80)
        self.logger.info("FEATURE ENGINEERING PIPELINE — COMPLETE")
        self.logger.info("=" * 80)

        return X_train, X_test, y_train, y_test
