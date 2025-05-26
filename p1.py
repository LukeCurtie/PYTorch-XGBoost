import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupKFold, GroupShuffleSplit
from sklearn.metrics import ndcg_score
from xgboost import XGBRanker
import optuna

# --- Helpers ------------------------------------------------
def get_group_sizes(idx, grp_labels):
    """
    Given an array of row‐indices `idx` and the full-group labels `grp_labels`,
    return a list of group sizes in the order the groups first appear.
    """
    sub = grp_labels[idx]
    uniq, cnts = np.unique(sub, return_counts=True)
    first_pos = {g: np.where(sub == g)[0][0] for g in uniq}
    ordered = sorted(uniq, key=lambda g: first_pos[g])
    return [int(cnts[np.where(uniq == g)]) for g in ordered]

def precision_at_k(groups, true_rel, pred_scores, k):
    dfm = pd.DataFrame({'grp': groups, 'rel': true_rel, 'score': pred_scores})
    precs = []
    for _, grp_df in dfm.groupby('grp'):
        topk = grp_df.nlargest(k, 'score')
        precs.append((topk['rel'] > 0).sum() / k)
    return np.mean(precs)


# --- 1) Load & feature engineering -------------------------
df = pd.read_csv('./data.csv')  # adjust path if needed
df['Date']        = pd.to_datetime(df['Date_of_Purchase'])
df['Purch_DOW']   = df['Date'].dt.weekday
df['Purch_Month'] = df['Date'].dt.month

df['Cust_Label'] = LabelEncoder().fit_transform(df['Customer_ID'])
for col in ['Gender', 'Age_Group', 'Region']:
    df[col] = LabelEncoder().fit_transform(df[col])

FEATURES = ['Annual_Income', 'Purch_DOW', 'Purch_Month', 'Gender', 'Age_Group', 'Region']
X      = df[FEATURES].values
y      = df['Frequency'].values
groups = df['Cust_Label'].values


# --- 2) Hold‐out split by customer (10%) --------------------
gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
train_val_idx, test_idx = next(gss.split(X, y, groups))

X90, y90, grp90 = X[train_val_idx], y[train_val_idx], groups[train_val_idx]


# --- 3) Optuna objective (no early‐stopping) ---------------
def objective(trial):
    params = {
        'learning_rate':    trial.suggest_float('learning_rate',    0.01, 0.3, log=True),
        'max_depth':        trial.suggest_int('max_depth',          3,    10),
        'n_estimators':     trial.suggest_int('n_estimators',     50,   300),
        'subsample':        trial.suggest_float('subsample',        0.5,  1.0),
        'colsample_bytree': trial.suggest_float('colsample_bytree', 0.5,  1.0),
        'gamma':            trial.suggest_float('gamma',            0.0,  5.0),
        'min_child_weight': trial.suggest_int('min_child_weight',   1,    10),
        'objective':        'rank:pairwise',
        'eval_metric':      'ndcg@3',
        'random_state':     42,
        'verbosity':        0,
    }

    cv = GroupKFold(n_splits=5)
    ndcgs = []

    # we use X90, y90, grp90 for CV
    for tr, val in cv.split(X90, y90, grp90):
        X_tr, X_val = X90[tr],   X90[val]
        y_tr, y_val = y90[tr],   y90[val]
        grp_tr      = get_group_sizes(train_val_idx[tr], groups)
        grp_val     = get_group_sizes(train_val_idx[val], groups)

        model = XGBRanker(**params)
        # NOTE: no early_stopping_rounds or callbacks here
        model.fit(X_tr, y_tr, group=grp_tr, verbose=False)

        preds = model.predict(X_val)
        ndcgs.append(ndcg_score([y_val], [preds], k=3))

    return -np.mean(ndcgs)  # minimize negative NDCG


# --- 4) Run hyperparameter search --------------------------
study = optuna.create_study(direction='minimize')
study.optimize(objective, n_trials=30)

print("Best params:", study.best_params)
print("Best CV NDCG@3:", -study.best_value)


# --- 5) Train final model on 90% ---------------------------
best = study.best_params.copy()
best.update({
    'objective':    'rank:pairwise',
    'eval_metric':  'ndcg@3',
    'random_state': 42,
    'verbosity':    0,
})
final_model = XGBRanker(**best)
final_model.fit(
    X90, y90,
    group=get_group_sizes(train_val_idx, groups),
    verbose=False
)


# --- 6) Evaluate on hold‐out 10% --------------------------
preds_test = final_model.predict(X[test_idx])
print(f"\nTest NDCG@3: {ndcg_score([y[test_idx]], [preds_test], k=3):.4f}")
for k in [1, 3, 5]:
    print(f"Test Precision@{k}: {precision_at_k(groups[test_idx], y[test_idx], preds_test, k):.3f}")


# --- 7) Sample top‐3 recommendations per customer ---------
df['Score'] = final_model.predict(X)
top3 = (
    df
    .sort_values(['Cust_Label','Score'], ascending=[True, False])
    .groupby('Cust_Label')
    .head(3)
    [['Customer_ID','Product','Score']]
)
print("\nSample top‐3 recommendations:")
print(top3.head(10).to_string(index=False))
