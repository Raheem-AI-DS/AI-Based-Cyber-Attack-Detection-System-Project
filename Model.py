"""
AI-Based Cyber Attack Detection and Hacking Risk Prediction System
===================================================================
Model.py  -  Dataset generation, training, evaluation and prediction API

Usage
-----
    python Model.py                       # create dataset if missing, train, evaluate, save model
    python Model.py --fast                # smaller/faster training (good for a first test)
    python Model.py --tune                # + hyper-parameter search (RandomizedSearchCV)
    python Model.py --regenerate          # rebuild the dataset before training
    python Model.py --generate-only       # only write cyber_attack_dataset.csv
    python Model.py --demo                # print a few sample predictions with the saved model

Used by app.py:
    from Model import predict_attack
    predict_attack(features_dict) -> {attack_type, confidence, probabilities, risk_score, ...}

Machine-learning concepts used
------------------------------
 1. Feature engineering        log transforms, byte ratios, login-failure ratio, port-scan ratio, posture risk
 2. Unsupervised learning      Isolation Forest trained on NORMAL traffic only -> anomaly score used as a feature
 3. Feature scaling/encoding   StandardScaler + One-Hot encoding in a ColumnTransformer
 4. Feature selection          SelectFromModel (Extra-Trees importance)
 5. Ensemble learning          Random Forest (bagging), Extra-Trees, Gradient Boosting (boosting)
 6. Diverse base learners      SVM (RBF kernel), Logistic Regression, Neural Network (MLP)
 7. Stacking                   meta-learner (Logistic Regression) combines base-learner outputs via 5-fold CV
 8. Class-imbalance handling   class_weight="balanced"
 9. Cross-validation           Stratified K-Fold for model comparison
10. Hyper-parameter tuning     RandomizedSearchCV (optional --tune)
11. Regression                 Voting ensemble (GBM + Random Forest + Ridge) predicts the 0-100 hacking-risk score
12. Evaluation                 accuracy, precision, recall, F1, ROC-AUC, log-loss, confusion matrix, MAE, R2
"""

from __future__ import annotations

import argparse
import json
import threading
import time
import warnings
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
    IsolationForest,
    RandomForestClassifier,
    RandomForestRegressor,
    StackingClassifier,
    VotingRegressor,
)
from sklearn.feature_selection import SelectFromModel
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    mean_absolute_error,
    precision_score,
    r2_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import RandomizedSearchCV, StratifiedKFold, cross_val_score, train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.svm import SVC

warnings.filterwarnings("ignore")

# --------------------------------------------------------------------------
# Paths & schema (must match app.py)
# --------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
DATASET_PATH = BASE_DIR / "cyber_attack_dataset.csv"
MODEL_PATH = BASE_DIR / "cyber_model.joblib"
REPORT_PATH = BASE_DIR / "model_report.json"
CM_PATH = BASE_DIR / "confusion_matrix.png"
SEED = 42

CLASSES = ["Normal", "DDoS Attack", "Port Scan", "SQL Injection", "XSS Attack", "Brute Force"]
SERVICES = ["http", "https", "ssh", "ftp", "dns", "smtp", "telnet", "smb", "other"]
FLAGS = ["SF", "S0", "REJ", "RSTO", "SH", "OTH"]
PROTOCOLS = ["tcp", "udp", "icmp"]
SERVICE_PORT = {"http": 80, "https": 443, "ssh": 22, "ftp": 21, "dns": 53, "smtp": 25, "telnet": 23, "smb": 445}

RAW_CAT = ["protocol_type", "service", "flag"]
RAW_BOOL = ["outdated_software", "firewall_enabled", "mfa_enabled"]
RAW_NUM = [
    "duration", "src_bytes", "dst_bytes", "packet_count", "packets_per_sec", "src_port", "dst_port",
    "unique_dst_ports", "unique_dst_ips", "conn_count", "syn_error_rate", "error_rate", "failed_logins",
    "login_attempts", "payload_sql_tokens", "payload_script_tokens", "open_ports", "days_since_patch",
]
FEATURES = RAW_CAT + RAW_NUM + RAW_BOOL  # the 24 inputs app.py sends
TARGET_CLASS, TARGET_RISK = "label", "risk_score"
CLASS_TO_ID = {c: i for i, c in enumerate(CLASSES)}  # integer labels keep every sklearn estimator happy

DEFAULTS = {
    "protocol_type": "tcp", "service": "other", "flag": "SF", "duration": 1.0, "src_bytes": 500, "dst_bytes": 500,
    "packet_count": 20, "packets_per_sec": 20.0, "src_port": 49152, "dst_port": 443, "unique_dst_ports": 1,
    "unique_dst_ips": 1, "conn_count": 10, "syn_error_rate": 0.0, "error_rate": 0.0, "failed_logins": 0,
    "login_attempts": 1, "payload_sql_tokens": 0, "payload_script_tokens": 0, "open_ports": 5,
    "days_since_patch": 14, "outdated_software": 0, "firewall_enabled": 1, "mfa_enabled": 0,
}

# columns the models actually consume (all engineered / log-scaled so scale-sensitive models behave)
MODEL_NUM_BASE = [
    "log_duration", "log_src_bytes", "log_dst_bytes", "log_packets", "log_pps", "log_conn", "log_ports",
    "log_ips", "syn_error_rate", "error_rate", "log_failed", "log_attempts", "log_sql", "log_script",
    "open_ports", "days_since_patch", "bytes_ratio", "bytes_per_packet", "port_conn_ratio",
    "login_fail_ratio", "log_payload_tokens", "token_diff", "well_known_dst", "src_port_norm",
    "dst_port_norm", "posture_risk",
] + RAW_BOOL
MODEL_NUM_FINAL = MODEL_NUM_BASE + ["anomaly_score"]


# --------------------------------------------------------------------------
# 1) Synthetic dataset generator
# --------------------------------------------------------------------------
def _cat(rng, n, weights: dict):
    keys = list(weights)
    p = np.array([weights[k] for k in keys], float)
    return rng.choice(keys, size=n, p=p / p.sum())


def _ports(rng, service, random_all=False):
    base = np.array([SERVICE_PORT.get(s, -1) for s in service])
    rand = rng.integers(1, 65535, len(service)) if random_all else rng.integers(1024, 65535, len(service))
    return np.where(base > 0, base, rand)


def _gen_normal(rng, n):
    proto = _cat(rng, n, {"tcp": .70, "udp": .25, "icmp": .05})
    service = _cat(rng, n, dict(zip(SERVICES, [.22, .30, .06, .05, .14, .06, .03, .05, .09])))
    flag = _cat(rng, n, dict(zip(FLAGS, [.90, .02, .03, .02, .01, .02])))
    duration = rng.lognormal(0.8, 1.0, n)
    packets = rng.poisson(35, n) + 1
    pps = packets / np.maximum(duration, 0.2) * rng.uniform(.5, 1.5, n)
    conn = rng.poisson(15, n) + 1
    syn = rng.beta(1, 40, n)
    busy = rng.random(n) < 0.10  # flash-crowd / backup jobs: legitimate but heavy
    pps = np.where(busy, pps * rng.uniform(5, 40, n), pps)
    conn = np.where(busy, rng.poisson(150, n) + 1, conn)
    syn = np.where(busy, rng.beta(2, 25, n), syn)
    failed = np.where(rng.random(n) < 0.05, rng.poisson(4, n), rng.poisson(0.2, n))  # forgotten passwords
    return pd.DataFrame(dict(
        protocol_type=proto, service=service, flag=flag, duration=duration,
        src_bytes=rng.lognormal(7.0, 1.3, n), dst_bytes=rng.lognormal(8.0, 1.5, n), packet_count=packets,
        packets_per_sec=pps, src_port=rng.integers(1024, 65535, n), dst_port=_ports(rng, service),
        unique_dst_ports=1 + rng.poisson(1.2, n), unique_dst_ips=1 + rng.poisson(1.5, n), conn_count=conn,
        syn_error_rate=syn, error_rate=rng.beta(1, 60, n), failed_logins=failed,
        login_attempts=failed + rng.poisson(1.5, n) + 1, payload_sql_tokens=rng.poisson(.08, n),
        payload_script_tokens=rng.poisson(.08, n)))


def _gen_ddos(rng, n):
    proto = _cat(rng, n, {"tcp": .55, "udp": .30, "icmp": .15})
    service = _cat(rng, n, dict(zip(SERVICES, [.25, .25, 0, .04, .15, .04, 0, 0, .27])))
    flag = _cat(rng, n, dict(zip(FLAGS, [.15, .55, .15, .05, .05, .05])))
    udp_flood = rng.random(n) < 0.30
    src = np.where(udp_flood, rng.lognormal(8.0, 1.0, n), rng.lognormal(5.0, 1.0, n))
    weak = rng.random(n) < 0.10  # low-rate application-layer DDoS (harder)
    pps = np.where(weak, rng.lognormal(6.3, 0.6, n), rng.lognormal(8.5, 0.8, n))
    conn = np.where(weak, rng.lognormal(5.0, 0.5, n), rng.lognormal(6.0, 0.7, n))
    return pd.DataFrame(dict(
        protocol_type=proto, service=service, flag=flag, duration=rng.lognormal(-0.5, 0.9, n),
        src_bytes=src, dst_bytes=rng.lognormal(3.5, 1.5, n), packet_count=rng.lognormal(7.5, 0.9, n),
        packets_per_sec=pps, src_port=rng.integers(1024, 65535, n), dst_port=_ports(rng, service),
        unique_dst_ports=1 + rng.poisson(2, n), unique_dst_ips=1 + rng.poisson(1.5, n), conn_count=conn,
        syn_error_rate=rng.beta(6, 3, n), error_rate=rng.beta(3, 8, n), failed_logins=rng.poisson(.1, n),
        login_attempts=rng.poisson(1, n) + 1, payload_sql_tokens=rng.poisson(.1, n),
        payload_script_tokens=rng.poisson(.1, n)))


def _gen_portscan(rng, n):
    proto = _cat(rng, n, {"tcp": .80, "udp": .17, "icmp": .03})
    service = _cat(rng, n, dict(zip(SERVICES, [.10, .10, .10, .06, 0, 0, .06, .08, .50])))
    flag = _cat(rng, n, dict(zip(FLAGS, [.05, .40, .25, .15, .12, .03])))
    slow = rng.random(n) < 0.12  # slow / stealth scan: fewer ports per window
    ports = np.where(slow, rng.lognormal(3.0, 0.6, n), rng.lognormal(5.0, 0.9, n)) + 4
    return pd.DataFrame(dict(
        protocol_type=proto, service=service, flag=flag, duration=rng.lognormal(1.0, 0.9, n),
        src_bytes=rng.lognormal(3.8, 0.8, n), dst_bytes=rng.lognormal(3.0, 1.2, n),
        packet_count=rng.poisson(150, n) + 20, packets_per_sec=rng.lognormal(4.0, 0.8, n),
        src_port=rng.integers(1024, 65535, n), dst_port=_ports(rng, service, random_all=True),
        unique_dst_ports=np.minimum(ports, 1000), unique_dst_ips=1 + rng.poisson(2, n),
        conn_count=rng.lognormal(4.5, 0.7, n), syn_error_rate=rng.beta(4, 5, n), error_rate=rng.beta(3, 6, n),
        failed_logins=rng.poisson(.2, n), login_attempts=rng.poisson(1, n) + 1,
        payload_sql_tokens=rng.poisson(.1, n), payload_script_tokens=rng.poisson(.1, n)))


def _gen_web_injection(rng, n, kind):
    service = _cat(rng, n, dict(zip(SERVICES, [.50, .45, 0, 0, 0, 0, 0, 0, .05])))
    flag = _cat(rng, n, dict(zip(FLAGS, [.93, .01, .02, .02, .01, .01])))
    hit = 2 + rng.poisson(6 if kind == "sql" else 5, n)
    obfuscated = rng.random(n) < 0.12  # encoded / obfuscated payload evades token counting
    hit = np.where(obfuscated, rng.poisson(1.2, n), hit)
    other = rng.poisson(.3, n)
    failed = rng.poisson(.3, n)
    return pd.DataFrame(dict(
        protocol_type=_cat(rng, n, {"tcp": .97, "udp": .02, "icmp": .01}), service=service, flag=flag,
        duration=rng.lognormal(0.9, 0.8, n), src_bytes=rng.lognormal(7.8, 0.9, n),
        dst_bytes=rng.lognormal(8.4, 1.2, n), packet_count=rng.poisson(25, n) + 5,
        packets_per_sec=rng.lognormal(2.8, 0.8, n), src_port=rng.integers(1024, 65535, n),
        dst_port=_ports(rng, service), unique_dst_ports=1 + rng.poisson(.3, n), unique_dst_ips=1,
        conn_count=rng.poisson(10, n) + 1, syn_error_rate=rng.beta(1, 40, n), error_rate=rng.beta(2, 20, n),
        failed_logins=failed, login_attempts=failed + rng.poisson(2, n) + 1,
        payload_sql_tokens=hit if kind == "sql" else other,
        payload_script_tokens=hit if kind == "xss" else other))


def _gen_bruteforce(rng, n):
    service = _cat(rng, n, dict(zip(SERVICES, [.10, .15, .35, .15, 0, .05, .10, .10, 0])))
    flag = _cat(rng, n, dict(zip(FLAGS, [.60, .05, .15, .20, 0, 0])))
    failed = 6 + rng.lognormal(3.5, 0.9, n)
    slow = rng.random(n) < 0.12  # low-and-slow guessing, overlaps with "forgot my password"
    failed = np.where(slow, rng.poisson(5, n) + 3, failed)
    return pd.DataFrame(dict(
        protocol_type=_cat(rng, n, {"tcp": .96, "udp": .03, "icmp": .01}), service=service, flag=flag,
        duration=rng.lognormal(2.0, 0.9, n), src_bytes=rng.lognormal(6.5, 0.7, n),
        dst_bytes=rng.lognormal(6.0, 0.8, n), packet_count=rng.poisson(120, n) + 10,
        packets_per_sec=rng.lognormal(3.2, 0.7, n), src_port=rng.integers(1024, 65535, n),
        dst_port=_ports(rng, service), unique_dst_ports=1 + rng.poisson(.3, n),
        unique_dst_ips=1 + rng.poisson(.4, n), conn_count=rng.lognormal(3.2, 0.7, n),
        syn_error_rate=rng.beta(1, 25, n), error_rate=rng.beta(2, 15, n), failed_logins=failed,
        login_attempts=failed + rng.poisson(8, n) + 1, payload_sql_tokens=rng.poisson(.1, n),
        payload_script_tokens=rng.poisson(.1, n)))


def _posture_vec(d: pd.DataFrame) -> np.ndarray:
    """Security-posture component of hacking risk (0..40) - same formula as app.py."""
    return (np.minimum(d["open_ports"], 40) / 40 * 12 + 8 * d["outdated_software"]
            + 9 * (1 - d["firewall_enabled"]) + 4 * (1 - d["mfa_enabled"])
            + np.minimum(d["days_since_patch"], 180) / 180 * 7)


def generate_dataset(n: int = 15000, seed: int = SEED, label_noise: float = 0.015) -> pd.DataFrame:
    """Create a labelled, realistic-but-synthetic network-traffic dataset (overlapping classes + label noise)."""
    rng = np.random.default_rng(seed)
    share = {"Normal": .40, "DDoS Attack": .14, "Port Scan": .14, "SQL Injection": .11,
             "XSS Attack": .10, "Brute Force": .11}
    parts = []
    for cls, frac in share.items():
        m = int(round(n * frac))
        df = {"Normal": _gen_normal, "DDoS Attack": _gen_ddos, "Port Scan": _gen_portscan,
              "Brute Force": _gen_bruteforce}.get(cls, None)
        if cls == "SQL Injection":
            df = _gen_web_injection(rng, m, "sql")
        elif cls == "XSS Attack":
            df = _gen_web_injection(rng, m, "xss")
        else:
            df = df(rng, m)
        df[TARGET_CLASS] = cls
        parts.append(df)
    data = pd.concat(parts, ignore_index=True)

    attacked = (data[TARGET_CLASS] != "Normal").to_numpy()
    k = len(data)
    data["open_ports"] = 1 + rng.poisson(6, k) + attacked * rng.poisson(2, k) + (data[TARGET_CLASS] == "Port Scan") * rng.poisson(4, k)
    data["outdated_software"] = (rng.random(k) < np.where(attacked, .40, .25)).astype(int)
    data["firewall_enabled"] = (rng.random(k) < np.where(attacked, .80, .92)).astype(int)
    data["mfa_enabled"] = (rng.random(k) < np.where(attacked, .25, .40)).astype(int)
    data["days_since_patch"] = rng.gamma(2.0, np.where(attacked, 22, 12), k)

    # hacking-risk target = attack severity x intensity + security posture + noise
    weight = {"Normal": 0, "DDoS Attack": 55, "Port Scan": 35, "SQL Injection": 60, "XSS Attack": 45, "Brute Force": 50}
    lab = data[TARGET_CLASS]
    inten = np.zeros(k)
    inten = np.where(lab == "DDoS Attack", np.clip(data["packets_per_sec"] / 30000, .4, 1), inten)
    inten = np.where(lab == "Port Scan", np.clip(data["unique_dst_ports"] / 500, .4, 1), inten)
    inten = np.where(lab == "SQL Injection", np.clip(data["payload_sql_tokens"] / 12, .4, 1), inten)
    inten = np.where(lab == "XSS Attack", np.clip(data["payload_script_tokens"] / 12, .4, 1), inten)
    inten = np.where(lab == "Brute Force", np.clip(data["failed_logins"] / 150, .4, 1), inten)
    risk = lab.map(weight).to_numpy() * inten + _posture_vec(data) * np.where(attacked, 1.0, 0.6)
    data[TARGET_RISK] = np.clip(risk + rng.normal(0, 2.5, k), 0, 100)

    flip = rng.random(k) < label_noise  # annotation noise: keeps accuracy realistic (< 100%)
    data.loc[flip, TARGET_CLASS] = rng.choice(CLASSES, flip.sum())

    # tidy types
    for c in ["src_bytes", "dst_bytes", "packet_count", "unique_dst_ports", "unique_dst_ips", "conn_count",
              "failed_logins", "login_attempts", "payload_sql_tokens", "payload_script_tokens", "open_ports",
              "src_port", "dst_port", "days_since_patch"]:
        data[c] = np.maximum(np.round(data[c]), 0).astype(int)
    data["unique_dst_ports"] = np.maximum(data["unique_dst_ports"], 1)
    data["unique_dst_ips"] = np.maximum(data["unique_dst_ips"], 1)
    for c in ["duration", "packets_per_sec"]:
        data[c] = data[c].round(3)
    for c in ["syn_error_rate", "error_rate"]:
        data[c] = data[c].clip(0, 1).round(4)
    data[TARGET_RISK] = data[TARGET_RISK].round(2)
    data = data.sample(frac=1, random_state=seed).reset_index(drop=True)
    return data[FEATURES + [TARGET_CLASS, TARGET_RISK]]


# --------------------------------------------------------------------------
# 2) Feature engineering
# --------------------------------------------------------------------------
def engineer(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    for c in RAW_BOOL:
        d[c] = d[c].astype(int)
    for c in RAW_NUM:
        d[c] = pd.to_numeric(d[c], errors="coerce").fillna(DEFAULTS[c]).clip(lower=0)
    d["log_duration"] = np.log1p(d["duration"])
    d["log_src_bytes"] = np.log1p(d["src_bytes"])
    d["log_dst_bytes"] = np.log1p(d["dst_bytes"])
    d["log_packets"] = np.log1p(d["packet_count"])
    d["log_pps"] = np.log1p(d["packets_per_sec"])
    d["log_conn"] = np.log1p(d["conn_count"])
    d["log_ports"] = np.log1p(d["unique_dst_ports"])
    d["log_ips"] = np.log1p(d["unique_dst_ips"])
    d["log_failed"] = np.log1p(d["failed_logins"])
    d["log_attempts"] = np.log1p(d["login_attempts"])
    d["log_sql"] = np.log1p(d["payload_sql_tokens"])
    d["log_script"] = np.log1p(d["payload_script_tokens"])
    d["bytes_ratio"] = d["log_src_bytes"] - d["log_dst_bytes"]
    d["bytes_per_packet"] = np.log1p((d["src_bytes"] + d["dst_bytes"]) / (d["packet_count"] + 1))
    d["port_conn_ratio"] = d["unique_dst_ports"] / (d["conn_count"] + 1)
    d["login_fail_ratio"] = d["failed_logins"] / (d["login_attempts"] + 1)
    d["log_payload_tokens"] = np.log1p(d["payload_sql_tokens"] + d["payload_script_tokens"])
    d["token_diff"] = np.sign(d["payload_sql_tokens"] - d["payload_script_tokens"]) * np.log1p(
        (d["payload_sql_tokens"] - d["payload_script_tokens"]).abs())
    d["well_known_dst"] = (d["dst_port"] < 1024).astype(int)
    d["src_port_norm"] = d["src_port"] / 65535.0
    d["dst_port_norm"] = d["dst_port"] / 65535.0
    d["posture_risk"] = _posture_vec(d)
    return d


def make_preprocessor(numeric_cols) -> ColumnTransformer:
    return ColumnTransformer([
        ("num", StandardScaler(), list(numeric_cols)),
        ("cat", OneHotEncoder(handle_unknown="ignore", sparse_output=False), RAW_CAT),
    ])


def add_anomaly_score(d: pd.DataFrame, iso: Pipeline) -> pd.DataFrame:
    d = d.copy()
    d["anomaly_score"] = -iso.score_samples(d[MODEL_NUM_BASE])  # higher = more abnormal
    return d


# --------------------------------------------------------------------------
# 3) Training & evaluation
# --------------------------------------------------------------------------
def _base_learners(fast: bool):
    n_trees = 60 if fast else 150
    return [
        ("random_forest", RandomForestClassifier(n_estimators=n_trees, class_weight="balanced", n_jobs=-1, random_state=SEED)),
        ("extra_trees", ExtraTreesClassifier(n_estimators=n_trees, class_weight="balanced", n_jobs=-1, random_state=SEED)),
        ("gradient_boosting", HistGradientBoostingClassifier(max_iter=80 if fast else 150, learning_rate=0.08, random_state=SEED)),
        ("svm_rbf", SVC(C=3.0, gamma="scale", class_weight="balanced", random_state=SEED)),
        ("logistic_regression", LogisticRegression(C=2.0, max_iter=1500, class_weight="balanced")),
        ("neural_net", MLPClassifier(hidden_layer_sizes=(64, 32), early_stopping=True, max_iter=300, random_state=SEED)),
    ]


def train_and_save(dataset_path: Path = DATASET_PATH, fast: bool = False, tune: bool = False,
                   cv_folds: int = 3, verbose: bool = True) -> dict:
    log = print if verbose else (lambda *a, **k: None)
    t_start = time.time()

    if not Path(dataset_path).exists():
        log(f"[data] {dataset_path} not found -> generating synthetic dataset")
        generate_dataset().to_csv(dataset_path, index=False)
    data = pd.read_csv(dataset_path)
    log(f"[data] {len(data):,} rows | classes:\n{data[TARGET_CLASS].value_counts().to_string()}")

    train_df, test_df = train_test_split(data, test_size=0.20, stratify=data[TARGET_CLASS], random_state=SEED)
    y_tr = train_df[TARGET_CLASS].map(CLASS_TO_ID).to_numpy()
    y_te = test_df[TARGET_CLASS].map(CLASS_TO_ID).to_numpy()
    r_tr, r_te = train_df[TARGET_RISK].to_numpy(), test_df[TARGET_RISK].to_numpy()

    # --- feature engineering + unsupervised anomaly detector (fit on NORMAL traffic only)
    X_tr, X_te = engineer(train_df[FEATURES]), engineer(test_df[FEATURES])
    iso = Pipeline([("sc", StandardScaler()),
                    ("iso", IsolationForest(n_estimators=100 if fast else 200, random_state=SEED, n_jobs=-1))])
    iso.fit(X_tr.loc[y_tr == CLASS_TO_ID["Normal"], MODEL_NUM_BASE])
    X_tr, X_te = add_anomaly_score(X_tr, iso), add_anomaly_score(X_te, iso)
    log("[unsupervised] Isolation Forest fitted on normal traffic -> anomaly_score feature added")

    skf = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=SEED)
    learners = _base_learners(fast)

    # --- optional hyper-parameter tuning of the boosting learner
    if tune:
        log("[tuning] RandomizedSearchCV on gradient boosting ...")
        pipe = Pipeline([("prep", make_preprocessor(MODEL_NUM_FINAL)),
                         ("clf", HistGradientBoostingClassifier(random_state=SEED))])
        search = RandomizedSearchCV(
            pipe, {"clf__learning_rate": [0.03, 0.05, 0.08, 0.12], "clf__max_depth": [None, 4, 6, 8],
                   "clf__max_iter": [100, 150, 250], "clf__l2_regularization": [0.0, 0.1, 1.0],
                   "clf__min_samples_leaf": [10, 20, 40]},
            n_iter=8, cv=3, scoring="f1_macro", random_state=SEED, n_jobs=1)
        search.fit(X_tr, y_tr)
        best = {k.replace("clf__", ""): v for k, v in search.best_params_.items()}
        learners = [(n, HistGradientBoostingClassifier(random_state=SEED, **best)) if n == "gradient_boosting" else (n, e)
                    for n, e in learners]
        log(f"[tuning] best params: {best} (cv f1_macro={search.best_score_:.4f})")

    # --- compare every base learner (hold-out + k-fold cross-validation)
    comparison, rf_importance = [], None
    for name, est in learners:
        t0 = time.time()
        pipe = Pipeline([("prep", make_preprocessor(MODEL_NUM_FINAL)), ("clf", clone(est))])
        cv_f1 = None if fast else float(np.mean(cross_val_score(pipe, X_tr, y_tr, cv=skf, scoring="f1_macro")))
        pipe.fit(X_tr, y_tr)
        pred = pipe.predict(X_te)
        comparison.append({"model": name, "accuracy": accuracy_score(y_te, pred),
                           "f1_macro": f1_score(y_te, pred, average="macro"),
                           "cv_f1_macro": cv_f1, "seconds": round(time.time() - t0, 1)})
        if name == "random_forest":
            names = pipe["prep"].get_feature_names_out()
            rf_importance = sorted(zip(names, pipe["clf"].feature_importances_), key=lambda x: -x[1])[:12]
        log(f"[base] {name:20s} acc={comparison[-1]['accuracy']:.4f}  f1={comparison[-1]['f1_macro']:.4f}"
            + ("" if cv_f1 is None else f"  cv_f1={cv_f1:.4f}"))

    # --- final model: feature selection + stacking ensemble
    stack = StackingClassifier(
        estimators=[(n, clone(e)) for n, e in learners],
        final_estimator=LogisticRegression(max_iter=2000, C=1.0),
        cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=SEED),
        stack_method="auto", n_jobs=1)
    selector = SelectFromModel(ExtraTreesClassifier(n_estimators=100, random_state=SEED, n_jobs=-1), threshold="0.25*mean")
    classifier = Pipeline([("prep", make_preprocessor(MODEL_NUM_FINAL)), ("select", selector), ("stack", stack)])
    log("[stacking] training stacked ensemble (6 base learners + logistic meta-learner) ...")
    classifier.fit(X_tr, y_tr)
    n_sel = int(classifier["select"].get_support().sum())
    n_all = len(classifier["select"].get_support())
    log(f"[feature-selection] kept {n_sel}/{n_all} encoded features")

    class_ids = list(classifier.classes_)
    classes = [CLASSES[i] for i in class_ids]
    proba = classifier.predict_proba(X_te)
    pred = classifier.classes_[proba.argmax(1)]
    metrics = {
        "accuracy": accuracy_score(y_te, pred),
        "precision_macro": precision_score(y_te, pred, average="macro"),
        "recall_macro": recall_score(y_te, pred, average="macro"),
        "f1_macro": f1_score(y_te, pred, average="macro"),
        "roc_auc_ovr": roc_auc_score(y_te, proba, multi_class="ovr", average="macro", labels=class_ids),
        "log_loss": log_loss(y_te, proba, labels=class_ids),
        "train_accuracy": accuracy_score(y_tr, classifier.predict(X_tr)),
    }
    cm = confusion_matrix(y_te, pred, labels=class_ids)
    log("\n[final] STACKING ENSEMBLE - hold-out test set")
    log(classification_report(y_te, pred, labels=class_ids, target_names=classes, digits=4))
    log("  " + "  ".join(f"{k}={v:.4f}" for k, v in metrics.items()))

    # --- regression model for the hacking-risk score (voting ensemble)
    regressor = Pipeline([("prep", make_preprocessor(MODEL_NUM_FINAL)), ("reg", VotingRegressor([
        ("hgb", HistGradientBoostingRegressor(max_iter=80 if fast else 200, learning_rate=0.08, random_state=SEED)),
        ("rf", RandomForestRegressor(n_estimators=60 if fast else 150, n_jobs=-1, random_state=SEED)),
        ("ridge", Ridge(alpha=1.0))], weights=[2, 2, 1]))])
    regressor.fit(X_tr, r_tr)
    r_pred = np.clip(regressor.predict(X_te), 0, 100)
    reg_metrics = {"mae": mean_absolute_error(r_te, r_pred), "r2": r2_score(r_te, r_pred)}
    log(f"[risk-regressor] MAE={reg_metrics['mae']:.2f} points   R2={reg_metrics['r2']:.4f}")

    # --- save artefacts
    report = {
        "trained_at": datetime.now().isoformat(timespec="seconds"), "rows": int(len(data)),
        "mode": "fast" if fast else "full", "tuned": tune, "classes": classes,
        "classification_metrics": metrics, "risk_regression_metrics": reg_metrics,
        "base_learner_comparison": comparison,
        "top_features_random_forest": [(str(n), float(v)) for n, v in (rf_importance or [])],
        "confusion_matrix": {"labels": classes, "matrix": cm.tolist()},
        "per_class": classification_report(y_te, pred, labels=class_ids, target_names=classes, output_dict=True),
    }
    report["sklearn_version"] = sklearn.__version__
    artefact = {"classifier": classifier, "regressor": regressor, "iso": iso, "classes": classes, "report": report,
                "sklearn_version": sklearn.__version__}
    try:  # saving is best-effort: a read-only host (e.g. some cloud platforms) must not break training
        joblib.dump(artefact, MODEL_PATH, compress=3)
        REPORT_PATH.write_text(json.dumps(report, indent=2, default=float))
        _save_confusion_matrix(cm, classes)
        log(f"\n[saved] {MODEL_PATH.name}, {REPORT_PATH.name}  (total {time.time() - t_start:.0f}s)")
    except Exception as exc:
        log(f"\n[warning] could not write model files ({exc}); the model is kept in memory only")
    return artefact


def _save_confusion_matrix(cm, classes):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt

        fig, ax = plt.subplots(figsize=(7, 6))
        ax.imshow(cm, cmap="Blues")
        ax.set_xticks(range(len(classes)), classes, rotation=35, ha="right")
        ax.set_yticks(range(len(classes)), classes)
        for i in range(len(classes)):
            for j in range(len(classes)):
                ax.text(j, i, cm[i, j], ha="center", va="center",
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
        ax.set_xlabel("Predicted")
        ax.set_ylabel("Actual")
        ax.set_title("Confusion Matrix - Stacking Ensemble")
        fig.tight_layout()
        fig.savefig(CM_PATH, dpi=140)
        plt.close(fig)
    except Exception:
        pass


# --------------------------------------------------------------------------
# 4) Prediction API (used by app.py)
# --------------------------------------------------------------------------
_ARTEFACT = None
_LOCK = threading.Lock()
_STATUS = {"state": "not loaded", "detail": ""}


def _try_load_saved():
    """Load the saved model, but only if it was made by the SAME scikit-learn version.

    A model file is a pickle: it breaks (for example "No module named '_loss'") when the library that
    loads it differs from the one that created it. That is exactly what happens when a model trained on a
    laptop is deployed to a cloud service with another scikit-learn version. In that case we return None
    and the caller retrains inside the current environment.
    """
    if not MODEL_PATH.exists():
        return None, "no saved model"
    try:
        art = joblib.load(MODEL_PATH)
    except Exception as exc:
        return None, f"saved model could not be loaded ({type(exc).__name__}: {exc})"
    saved = art.get("sklearn_version") if isinstance(art, dict) else None
    if saved != sklearn.__version__:
        return None, f"saved model was built with scikit-learn {saved}, this environment has {sklearn.__version__}"
    return art, "loaded saved model"


def _load():
    """Return the model, training a fresh one automatically when the saved file is missing or incompatible."""
    global _ARTEFACT
    if _ARTEFACT is not None:
        return _ARTEFACT
    with _LOCK:
        if _ARTEFACT is not None:
            return _ARTEFACT
        art, why = _try_load_saved()
        if art is None:
            print(f"[Model] {why} -> training a fresh model for this environment (about 1-3 minutes) ...")
            art = train_and_save(fast=True, verbose=False)
            _STATUS.update(state="retrained", detail=why)
        else:
            _STATUS.update(state="loaded", detail=why)
        _ARTEFACT = art
    return _ARTEFACT


def model_status() -> dict:
    """Small helper for the interface: how the model was obtained."""
    return {**_STATUS, "sklearn_version": sklearn.__version__}


def predict_attack(features: dict) -> dict:
    """Predict attack class, confidence, class probabilities and 0-100 hacking-risk score."""
    art = _load()
    row = {**DEFAULTS, **{k: v for k, v in features.items() if k in FEATURES}}
    X = add_anomaly_score(engineer(pd.DataFrame([row])[FEATURES]), art["iso"])
    proba = art["classifier"].predict_proba(X)[0]
    probs = {c: float(p) for c, p in zip(art["classes"], proba)}
    top = max(probs, key=probs.get)
    risk = float(np.clip(art["regressor"].predict(X)[0], 0, 100))
    return {
        "attack_type": top, "confidence": probs[top], "probabilities": probs, "risk_score": risk,
        "anomaly_score": float(X["anomaly_score"].iloc[0]),
    }


# --------------------------------------------------------------------------
# 5) Command line
# --------------------------------------------------------------------------
def _demo():
    samples = {
        "Normal web browsing": dict(protocol_type="tcp", service="https", flag="SF", duration=2.5, src_bytes=1500,
                                    dst_bytes=24000, packet_count=40, packets_per_sec=18, dst_port=443, conn_count=12),
        "SYN flood": dict(protocol_type="tcp", service="http", flag="S0", duration=0.4, src_bytes=60, dst_bytes=0,
                          packet_count=9000, packets_per_sec=22000, conn_count=1800, syn_error_rate=0.85,
                          error_rate=0.4, dst_port=80),
        "Port sweep": dict(protocol_type="tcp", service="other", flag="REJ", duration=4, src_bytes=44, dst_bytes=20,
                           packet_count=300, packets_per_sec=75, unique_dst_ports=420, conn_count=210,
                           syn_error_rate=0.55, error_rate=0.4),
        "SQL injection": dict(protocol_type="tcp", service="http", flag="SF", src_bytes=3200, dst_bytes=9000,
                              payload_sql_tokens=9, dst_port=80),
        "XSS payload": dict(protocol_type="tcp", service="https", flag="SF", src_bytes=2800, dst_bytes=7000,
                            payload_script_tokens=8, dst_port=443),
        "SSH password guessing": dict(protocol_type="tcp", service="ssh", flag="RSTO", duration=9, src_bytes=700,
                                      dst_bytes=600, packet_count=200, failed_logins=80, login_attempts=140, dst_port=22),
    }
    for name, f in samples.items():
        r = predict_attack(f)
        print(f"{name:24s} -> {r['attack_type']:14s} conf={r['confidence']:.2f}  risk={r['risk_score']:5.1f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="Train / test the cyber-attack detection model")
    ap.add_argument("--fast", action="store_true", help="smaller models, no k-fold comparison")
    ap.add_argument("--tune", action="store_true", help="hyper-parameter search for the boosting learner")
    ap.add_argument("--regenerate", action="store_true", help="rebuild the dataset before training")
    ap.add_argument("--generate-only", action="store_true", help="only write the dataset CSV")
    ap.add_argument("--samples", type=int, default=15000, help="rows when generating the dataset")
    ap.add_argument("--cv", type=int, default=3, help="folds for base-learner cross-validation")
    ap.add_argument("--demo", action="store_true", help="show sample predictions with the saved model")
    args = ap.parse_args()

    if args.regenerate or args.generate_only or not DATASET_PATH.exists():
        generate_dataset(args.samples).to_csv(DATASET_PATH, index=False)
        print(f"[data] wrote {DATASET_PATH.name} ({args.samples:,} rows)")
    if args.demo:
        _demo()
    elif not args.generate_only:
        train_and_save(fast=args.fast, tune=args.tune, cv_folds=args.cv)
        print("\nSample predictions:")
        _demo()
