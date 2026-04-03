import os
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, GridSearchCV, cross_val_score
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    average_precision_score, classification_report, confusion_matrix,
    precision_recall_curve
)
from sklearn.neural_network import MLPClassifier

import xgboost as xgb
from xgboost import plot_importance

warnings.filterwarnings("ignore")
np.random.seed(42)

OUTPUT_DIR = "output"
os.makedirs(OUTPUT_DIR, exist_ok=True)



df = pd.read_csv("data/bank-full.csv", sep=";")

cat_cols = df.select_dtypes(include="object").columns.tolist()
cat_cols.remove("y")  # target
num_cols = df.select_dtypes(include=["int64", "float64"]).columns.tolist()

print(f"\nCategorical features: {cat_cols}")
print(f"Numerical features: {num_cols}")


df.drop("duration", axis=1, inplace=True)
num_cols.remove("duration")
print("\nDropped 'duration': only available after the call ends, not usable at prediction time.")


df["previously_contacted"] = (df["pdays"] != -1).astype(int)
df["pdays_clean"] = df["pdays"].replace(-1, 0)
df.drop("pdays", axis=1, inplace=True)
num_cols.remove("pdays")
num_cols += ["previously_contacted", "pdays_clean"]

# Feature Engineering ---
df["balance_per_age"] = df["balance"] / (df["age"] + 1)
df["campaign_per_pdays"] = df["campaign"] / (df["pdays_clean"] + 1)


df["age_group"] = pd.cut(df["age"], bins=[0, 30, 40, 50, 60, 100],
                         labels=[0, 1, 2, 3, 4]).astype(int)

num_cols += ["balance_per_age", "campaign_per_pdays", "age_group"]


df["y"] = (df["y"] == "yes").astype(int)


X = df.drop("y", axis=1)
y = df["y"]

X_train_full, X_test, y_train_full, y_test = train_test_split(
    X, y, test_size=0.15, random_state=42, stratify=y
)
X_train, X_val, y_train, y_val = train_test_split(
    X_train_full, y_train_full,
    test_size=0.15 / 0.85,  # ~17.6% of remaining → 15% of total
    random_state=42, stratify=y_train_full
)

print(f"\nTrain: {X_train.shape[0]}, Val: {X_val.shape[0]}, Test: {X_test.shape[0]}")


label_encoders = {}
for col in cat_cols:
    le = LabelEncoder()
    le.fit(X_train[col])
    for split in [X_train, X_val, X_test]:
        split[col] = split[col].map(
            lambda x, le=le: le.transform([x])[0] if x in le.classes_ else -1
        )
    label_encoders[col] = le


scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_val_scaled = scaler.transform(X_val)
X_test_scaled = scaler.transform(X_test)


xgb_model = xgb.XGBClassifier(
    n_estimators=500,
    learning_rate=0.1,
    max_depth=5,
    subsample=0.8,
    reg_alpha=0.1,
    reg_lambda=1.0,
    use_label_encoder=False,
    eval_metric="logloss",
    random_state=42,
    early_stopping_rounds=30,
)

xgb_start = time.time()
xgb_model.fit(
    X_train, y_train,
    eval_set=[(X_train, y_train), (X_val, y_val)],
    verbose=False,
)
xgb_train_time = time.time() - xgb_start
print(f"XGBoost training time: {xgb_train_time:.2f}s")
print(f"Best iteration: {xgb_model.best_iteration}")


results = xgb_model.evals_result()
epochs = len(results["validation_0"]["logloss"])

fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(range(epochs), results["validation_0"]["logloss"], label="Train Loss")
ax.plot(range(epochs), results["validation_1"]["logloss"], label="Validation Loss")
ax.axvline(xgb_model.best_iteration, color="gray", linestyle="--", label="Early Stopping")
ax.set_xlabel("Boosting Round")
ax.set_ylabel("Log Loss")
ax.set_title("XGBoost: Training vs Validation Loss")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/xgb_loss_curve.png", dpi=150)
plt.close()
print("Saved: xgb_loss_curve.png")


fig, ax = plt.subplots(figsize=(10, 8))
plot_importance(xgb_model, ax=ax, max_num_features=15, importance_type="gain")
ax.set_title("XGBoost Feature Importance (Gain)")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/xgb_feature_importance.png", dpi=150)
plt.close()
print("Saved: xgb_feature_importance.png")

# Effect of learning rate
learning_rates = [0.01, 0.1, 0.3]
lr_results = {}

fig, ax = plt.subplots(figsize=(8, 5))
for lr in learning_rates:
    model_lr = xgb.XGBClassifier(
        n_estimators=500, learning_rate=lr, max_depth=5,
        subsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, early_stopping_rounds=30,
    )
    model_lr.fit(X_train, y_train,
                 eval_set=[(X_val, y_val)], verbose=False)
    res = model_lr.evals_result()
    val_loss = res["validation_0"]["logloss"]
    ax.plot(range(len(val_loss)), val_loss, label=f"lr={lr}")

    y_pred_lr = model_lr.predict(X_val)
    lr_results[lr] = {
        "best_iter": model_lr.best_iteration,
        "val_accuracy": accuracy_score(y_val, y_pred_lr),
        "val_f1": f1_score(y_val, y_pred_lr),
    }

ax.set_xlabel("Boosting Round")
ax.set_ylabel("Validation Log Loss")
ax.set_title("XGBoost: Effect of Learning Rate")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/xgb_learning_rate_comparison.png", dpi=150)
plt.close()
print("Saved: xgb_learning_rate_comparison.png")

print("\nLearning rate comparison:")
for lr, metrics in lr_results.items():
    print(f"  lr={lr}: best_iter={metrics['best_iter']}, "
          f"val_acc={metrics['val_accuracy']:.4f}, val_f1={metrics['val_f1']:.4f}")

# Effect of n_estimators
n_estimators_list = [50, 100, 200, 400]
ne_results = {}

fig, ax = plt.subplots(figsize=(8, 5))
for ne in n_estimators_list:
    model_ne = xgb.XGBClassifier(
        n_estimators=ne, learning_rate=0.1, max_depth=5,
        subsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42,
    )
    model_ne.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    res = model_ne.evals_result()
    train_loss = res["validation_0"]["logloss"]
    val_loss = res["validation_1"]["logloss"]
    ax.plot(range(len(val_loss)), val_loss, label=f"n_est={ne}")

    y_pred_ne = model_ne.predict(X_val)
    ne_results[ne] = {
        "train_loss": train_loss[-1],
        "val_loss": val_loss[-1],
        "val_accuracy": accuracy_score(y_val, y_pred_ne),
        "val_f1": f1_score(y_val, y_pred_ne),
    }

ax.set_xlabel("Boosting Round")
ax.set_ylabel("Validation Log Loss")
ax.set_title("XGBoost: Effect of n_estimators")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/xgb_n_estimators_comparison.png", dpi=150)
plt.close()
print("Saved: xgb_n_estimators_comparison.png")

print("\nn_estimators comparison:")
for ne, metrics in ne_results.items():
    print(f"  n_est={ne}: train_loss={metrics['train_loss']:.4f}, "
          f"val_loss={metrics['val_loss']:.4f}, val_f1={metrics['val_f1']:.4f}")

# Effect of max_depth
max_depths = [2, 3, 5, 7, 10]
md_results = {}

fig, ax = plt.subplots(figsize=(8, 5))
for md in max_depths:
    model_md = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=md,
        subsample=0.8, reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, early_stopping_rounds=30,
    )
    model_md.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    res = model_md.evals_result()
    train_loss = res["validation_0"]["logloss"]
    val_loss = res["validation_1"]["logloss"]
    ax.plot(range(len(val_loss)), val_loss, label=f"depth={md}")

    y_pred_md = model_md.predict(X_val)
    md_results[md] = {
        "train_loss": train_loss[-1],
        "val_loss": val_loss[-1],
        "val_accuracy": accuracy_score(y_val, y_pred_md),
        "val_f1": f1_score(y_val, y_pred_md),
    }

ax.set_xlabel("Boosting Round")
ax.set_ylabel("Validation Log Loss")
ax.set_title("XGBoost: Effect of max_depth")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/xgb_max_depth_comparison.png", dpi=150)
plt.close()
print("Saved: xgb_max_depth_comparison.png")

print("\nmax_depth comparison:")
for md, metrics in md_results.items():
    print(f"  depth={md}: train_loss={metrics['train_loss']:.4f}, "
          f"val_loss={metrics['val_loss']:.4f}, val_f1={metrics['val_f1']:.4f}")

# Effect of subsample
subsamples = [0.5, 0.7, 0.8, 1.0]
ss_results = {}

fig, ax = plt.subplots(figsize=(8, 5))
for ss in subsamples:
    model_ss = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=5,
        subsample=ss, reg_alpha=0.1, reg_lambda=1.0,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, early_stopping_rounds=30,
    )
    model_ss.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    res = model_ss.evals_result()
    train_loss = res["validation_0"]["logloss"]
    val_loss = res["validation_1"]["logloss"]
    ax.plot(range(len(val_loss)), val_loss, label=f"subsample={ss}")

    y_pred_ss = model_ss.predict(X_val)
    ss_results[ss] = {
        "train_loss": train_loss[-1],
        "val_loss": val_loss[-1],
        "val_accuracy": accuracy_score(y_val, y_pred_ss),
        "val_f1": f1_score(y_val, y_pred_ss),
    }

ax.set_xlabel("Boosting Round")
ax.set_ylabel("Validation Log Loss")
ax.set_title("XGBoost: Effect of subsample")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/xgb_subsample_comparison.png", dpi=150)
plt.close()
print("Saved: xgb_subsample_comparison.png")

print("\nsubsample comparison:")
for ss, metrics in ss_results.items():
    print(f"  subsample={ss}: train_loss={metrics['train_loss']:.4f}, "
          f"val_loss={metrics['val_loss']:.4f}, val_f1={metrics['val_f1']:.4f}")

# Effect of reg_alpha and reg_lambda
reg_values = [0, 0.1, 1.0, 10.0]
reg_results = {}

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# reg_alpha
for ra in reg_values:
    model_ra = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=5,
        subsample=0.8, reg_alpha=ra, reg_lambda=1.0,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, early_stopping_rounds=30,
    )
    model_ra.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    res = model_ra.evals_result()
    train_loss = res["validation_0"]["logloss"]
    val_loss = res["validation_1"]["logloss"]
    axes[0].plot(range(len(val_loss)), val_loss, label=f"alpha={ra}")

    y_pred_ra = model_ra.predict(X_val)
    reg_results[f"alpha={ra}"] = {
        "train_loss": train_loss[-1],
        "val_loss": val_loss[-1],
        "val_accuracy": accuracy_score(y_val, y_pred_ra),
        "val_f1": f1_score(y_val, y_pred_ra),
    }

axes[0].set_xlabel("Boosting Round")
axes[0].set_ylabel("Validation Log Loss")
axes[0].set_title("Effect of reg_alpha (L1)")
axes[0].legend()

# reg_lambda
for rl in reg_values:
    model_rl = xgb.XGBClassifier(
        n_estimators=500, learning_rate=0.1, max_depth=5,
        subsample=0.8, reg_alpha=0.1, reg_lambda=rl,
        use_label_encoder=False, eval_metric="logloss",
        random_state=42, early_stopping_rounds=30,
    )
    model_rl.fit(X_train, y_train,
                 eval_set=[(X_train, y_train), (X_val, y_val)], verbose=False)
    res = model_rl.evals_result()
    train_loss = res["validation_0"]["logloss"]
    val_loss = res["validation_1"]["logloss"]
    axes[1].plot(range(len(val_loss)), val_loss, label=f"lambda={rl}")

    y_pred_rl = model_rl.predict(X_val)
    reg_results[f"lambda={rl}"] = {
        "train_loss": train_loss[-1],
        "val_loss": val_loss[-1],
        "val_accuracy": accuracy_score(y_val, y_pred_rl),
        "val_f1": f1_score(y_val, y_pred_rl),
    }

axes[1].set_xlabel("Boosting Round")
axes[1].set_ylabel("Validation Log Loss")
axes[1].set_title("Effect of reg_lambda (L2)")
axes[1].legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/xgb_regularization_comparison.png", dpi=150)
plt.close()
print("Saved: xgb_regularization_comparison.png")

print("\nRegularization comparison:")
for name, metrics in reg_results.items():
    print(f"  {name}: train_loss={metrics['train_loss']:.4f}, "
          f"val_loss={metrics['val_loss']:.4f}, val_f1={metrics['val_f1']:.4f}")

# tuning with GridSearchCV
param_grid_xgb = {
    "max_depth": [3, 5, 7],
    "subsample": [0.7, 0.8, 1.0],
    "reg_alpha": [0, 0.1, 1.0],
}

xgb_grid = GridSearchCV(
    xgb.XGBClassifier(
        n_estimators=300, learning_rate=0.1,
        reg_lambda=1.0, use_label_encoder=False,
        eval_metric="logloss", random_state=42,
    ),
    param_grid_xgb,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=0,
)
xgb_grid.fit(X_train, y_train)
print(f"\nBest XGBoost params: {xgb_grid.best_params_}")
print(f"Best CV F1: {xgb_grid.best_score_:.4f}")

# Retrain best model with early stopping
best_xgb = xgb.XGBClassifier(
    n_estimators=500, learning_rate=0.1,
    **xgb_grid.best_params_,
    reg_lambda=1.0, use_label_encoder=False,
    eval_metric="logloss", random_state=42,
    early_stopping_rounds=30,
)
best_xgb.fit(X_train, y_train,
             eval_set=[(X_val, y_val)], verbose=False)


mlp_model = MLPClassifier(
    hidden_layer_sizes=(128, 64),
    activation="relu",
    learning_rate_init=0.001,
    max_iter=300,
    random_state=42,
    early_stopping=True,
    validation_fraction=0.15,
    n_iter_no_change=20,
)

mlp_start = time.time()
mlp_model.fit(X_train_scaled, y_train)
mlp_train_time = time.time() - mlp_start
print(f"MLP training time: {mlp_train_time:.2f}s")
print(f"MLP iterations: {mlp_model.n_iter_}")

# Training loss curve
fig, ax = plt.subplots(figsize=(8, 5))
ax.plot(mlp_model.loss_curve_, label="Training Loss")
if hasattr(mlp_model, "validation_scores_"):
    ax2 = ax.twinx()
    ax2.plot(mlp_model.validation_scores_, color="orange", label="Validation Score")
    ax2.set_ylabel("Validation Score")
    ax2.legend(loc="center right")
ax.set_xlabel("Iteration")
ax.set_ylabel("Training Loss")
ax.set_title("MLP: Training Loss Curve")
ax.legend(loc="upper right")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mlp_loss_curve.png", dpi=150)
plt.close()
print("Saved: mlp_loss_curve.png")

# Effect of network depth/width
architectures = {
    "(64,)": (64,),
    "(128, 64)": (128, 64),
    "(256, 128, 64)": (256, 128, 64),
    "(64, 32, 16)": (64, 32, 16),
}

arch_results = {}
fig, ax = plt.subplots(figsize=(8, 5))
for name, layers in architectures.items():
    mlp_arch = MLPClassifier(
        hidden_layer_sizes=layers,
        activation="relu",
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    mlp_arch.fit(X_train_scaled, y_train)
    ax.plot(mlp_arch.loss_curve_, label=name)

    y_pred_arch = mlp_arch.predict(X_val_scaled)
    arch_results[name] = {
        "val_accuracy": accuracy_score(y_val, y_pred_arch),
        "val_f1": f1_score(y_val, y_pred_arch),
    }

ax.set_xlabel("Iteration")
ax.set_ylabel("Training Loss")
ax.set_title("MLP: Effect of Network Architecture on Training Loss")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mlp_architecture_comparison.png", dpi=150)
plt.close()
print("Saved: mlp_architecture_comparison.png")

print("\nArchitecture comparison:")
for name, metrics in arch_results.items():
    print(f"  {name}: val_acc={metrics['val_accuracy']:.4f}, val_f1={metrics['val_f1']:.4f}")


fig, ax = plt.subplots(figsize=(8, 5))
arch_names = list(arch_results.keys())
val_f1s = [arch_results[n]["val_f1"] for n in arch_names]
val_accs = [arch_results[n]["val_accuracy"] for n in arch_names]
x = np.arange(len(arch_names))
width = 0.35
ax.bar(x - width / 2, val_accs, width, label="Val Accuracy")
ax.bar(x + width / 2, val_f1s, width, label="Val F1")
ax.set_xticks(x)
ax.set_xticklabels(arch_names)
ax.set_xlabel("Architecture")
ax.set_ylabel("Score")
ax.set_title("MLP: Validation Performance by Architecture")
ax.legend()
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mlp_architecture_val_performance.png", dpi=150)
plt.close()
print("Saved: mlp_architecture_val_performance.png")


activations = ["relu", "tanh"]
act_results = {}

fig, ax = plt.subplots(figsize=(8, 5))
for act in activations:
    mlp_act = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation=act,
        learning_rate_init=0.001,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    mlp_act.fit(X_train_scaled, y_train)
    ax.plot(mlp_act.loss_curve_, label=f"{act}")

    y_pred_act = mlp_act.predict(X_val_scaled)
    act_results[act] = {
        "val_accuracy": accuracy_score(y_val, y_pred_act),
        "val_f1": f1_score(y_val, y_pred_act),
    }

ax.set_xlabel("Iteration")
ax.set_ylabel("Training Loss")
ax.set_title("MLP: Effect of Activation Function on Training Loss")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mlp_activation_comparison.png", dpi=150)
plt.close()
print("Saved: mlp_activation_comparison.png")

print("\nActivation function comparison:")
for act, metrics in act_results.items():
    print(f"  {act}: val_acc={metrics['val_accuracy']:.4f}, val_f1={metrics['val_f1']:.4f}")


lr_inits = [0.001, 0.01, 0.1]
lr_mlp_results = {}
fig, ax = plt.subplots(figsize=(8, 5))
for lr in lr_inits:
    mlp_lr = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        learning_rate_init=lr,
        max_iter=300,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.15,
        n_iter_no_change=20,
    )
    mlp_lr.fit(X_train_scaled, y_train)
    ax.plot(mlp_lr.loss_curve_, label=f"lr={lr}")

    y_pred_lr = mlp_lr.predict(X_val_scaled)
    lr_mlp_results[lr] = {
        "val_accuracy": accuracy_score(y_val, y_pred_lr),
        "val_f1": f1_score(y_val, y_pred_lr),
    }

ax.set_xlabel("Iteration")
ax.set_ylabel("Training Loss")
ax.set_title("MLP: Effect of Learning Rate on Training Loss")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mlp_learning_rate_comparison.png", dpi=150)
plt.close()
print("Saved: mlp_learning_rate_comparison.png")

print("\nMLP Learning rate comparison:")
for lr, metrics in lr_mlp_results.items():
    print(f"  lr={lr}: val_acc={metrics['val_accuracy']:.4f}, val_f1={metrics['val_f1']:.4f}")


max_iters = [50, 100, 200, 500]
mi_results = {}

for mi in max_iters:
    mlp_mi = MLPClassifier(
        hidden_layer_sizes=(128, 64),
        activation="relu",
        learning_rate_init=0.001,
        max_iter=mi,
        random_state=42,
        early_stopping=False,  # disable to see full effect of max_iter
    )
    mlp_mi.fit(X_train_scaled, y_train)

    y_pred_mi = mlp_mi.predict(X_val_scaled)
    mi_results[mi] = {
        "actual_iters": mlp_mi.n_iter_,
        "val_accuracy": accuracy_score(y_val, y_pred_mi),
        "val_f1": f1_score(y_val, y_pred_mi),
    }

print("\nmax_iter comparison:")
for mi, metrics in mi_results.items():
    print(f"  max_iter={mi}: actual_iters={metrics['actual_iters']}, "
          f"val_acc={metrics['val_accuracy']:.4f}, val_f1={metrics['val_f1']:.4f}")


fig, ax = plt.subplots(figsize=(8, 5))
mi_names = [str(mi) for mi in max_iters]
val_f1s_mi = [mi_results[mi]["val_f1"] for mi in max_iters]
val_accs_mi = [mi_results[mi]["val_accuracy"] for mi in max_iters]
x = np.arange(len(mi_names))
width = 0.35
ax.bar(x - width / 2, val_accs_mi, width, label="Val Accuracy")
ax.bar(x + width / 2, val_f1s_mi, width, label="Val F1")
ax.set_xticks(x)
ax.set_xticklabels(mi_names)
ax.set_xlabel("max_iter")
ax.set_ylabel("Score")
ax.set_title("MLP: Effect of max_iter on Validation Performance")
ax.legend()
ax.set_ylim(0, 1.0)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/mlp_max_iter_comparison.png", dpi=150)
plt.close()
print("Saved: mlp_max_iter_comparison.png")


param_grid_mlp = {
    "hidden_layer_sizes": [(64,), (128, 64), (256, 128, 64)],
    "activation": ["relu", "tanh"],
    "learning_rate_init": [0.001, 0.01],
}

mlp_grid = GridSearchCV(
    MLPClassifier(max_iter=300, random_state=42, early_stopping=True,
                  validation_fraction=0.15, n_iter_no_change=20),
    param_grid_mlp,
    cv=3,
    scoring="f1",
    n_jobs=-1,
    verbose=0,
)
mlp_grid.fit(X_train_scaled, y_train)
print(f"\nBest MLP params: {mlp_grid.best_params_}")
print(f"Best CV F1: {mlp_grid.best_score_:.4f}")

best_mlp = mlp_grid.best_estimator_



y_pred_xgb = best_xgb.predict(X_test)
y_prob_xgb = best_xgb.predict_proba(X_test)[:, 1]

y_pred_mlp = best_mlp.predict(X_test_scaled)
y_prob_mlp = best_mlp.predict_proba(X_test_scaled)[:, 1]

def compute_metrics(y_true, y_pred, y_prob):
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "Precision": precision_score(y_true, y_pred),
        "Recall": recall_score(y_true, y_pred),
        "F1-Score": f1_score(y_true, y_pred),
        "AUC-PR": average_precision_score(y_true, y_prob),
    }

xgb_metrics = compute_metrics(y_test, y_pred_xgb, y_prob_xgb)
mlp_metrics = compute_metrics(y_test, y_pred_mlp, y_prob_mlp)

comparison_df = pd.DataFrame({
    "Metric": list(xgb_metrics.keys()) + ["Training Time (s)"],
    "XGBoost (GBDT)": list(xgb_metrics.values()) + [round(xgb_train_time, 2)],
    "MLP": list(mlp_metrics.values()) + [round(mlp_train_time, 2)],
})
comparison_df[["XGBoost (GBDT)", "MLP"]] = comparison_df[["XGBoost (GBDT)", "MLP"]].round(4)
print("\n--- Model Comparison Table ---")
print(comparison_df.to_string(index=False))

fig, ax = plt.subplots(figsize=(8, 3))
ax.axis("off")
table = ax.table(
    cellText=comparison_df.values,
    colLabels=comparison_df.columns,
    cellLoc="center",
    loc="center",
)
table.auto_set_font_size(False)
table.set_fontsize(11)
table.scale(1.2, 1.5)
ax.set_title("GBDT vs MLP: Performance Comparison", fontsize=14, pad=20)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/comparison_table.png", dpi=150, bbox_inches="tight")
plt.close()
print("Saved: comparison_table.png")

fig, ax = plt.subplots(figsize=(8, 5))
for name, y_prob in [("XGBoost", y_prob_xgb), ("MLP", y_prob_mlp)]:
    prec, rec, _ = precision_recall_curve(y_test, y_prob)
    ax.plot(rec, prec, label=f"{name} (AUC-PR={average_precision_score(y_test, y_prob):.3f})")
ax.set_xlabel("Recall")
ax.set_ylabel("Precision")
ax.set_title("Precision-Recall Curve: GBDT vs MLP")
ax.legend()
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/pr_curve_comparison.png", dpi=150)
plt.close()
print("Saved: pr_curve_comparison.png")


fig, axes = plt.subplots(1, 2, figsize=(12, 5))
for ax, y_pred, title in zip(axes, [y_pred_xgb, y_pred_mlp], ["XGBoost", "MLP"]):
    cm = confusion_matrix(y_test, y_pred)
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", ax=ax,
                xticklabels=["No", "Yes"], yticklabels=["No", "Yes"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    ax.set_title(f"{title} Confusion Matrix")
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/confusion_matrices.png", dpi=150)
plt.close()
print("Saved: confusion_matrices.png")


print("\n--- XGBoost Classification Report ---")
print(classification_report(y_test, y_pred_xgb, target_names=["No", "Yes"]))

print("--- MLP Classification Report ---")
print(classification_report(y_test, y_pred_mlp, target_names=["No", "Yes"]))
