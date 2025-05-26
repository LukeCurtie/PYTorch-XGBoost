import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
from sklearn.metrics import ndcg_score
import matplotlib.pyplot as plt

# 1) Load dataset & feature engineering
df = pd.read_csv('data.csv')  # adjust path if needed
df['Date']        = pd.to_datetime(df['Date_of_Purchase'])
df['Purch_DOW']   = df['Date'].dt.weekday
df['Purch_Month'] = df['Date'].dt.month

# 2) Encode customer & other categoricals
df['Cust_Label'] = LabelEncoder().fit_transform(df['Customer_ID'])
for col in ['Gender', 'Age_Group', 'Region']:
    df[col] = LabelEncoder().fit_transform(df[col])

# 3) Prepare features, labels & groups
FEATURES = ['Annual_Income', 'Purch_DOW', 'Purch_Month', 'Gender', 'Age_Group', 'Region']
X       = df[FEATURES].values
y       = df['Frequency'].values
groups  = df['Cust_Label'].values

def get_group_sizes(idx, grp_labels):
    sub = grp_labels[idx]
    unique, counts = np.unique(sub, return_counts=True)
    first_idx = {g: np.where(sub == g)[0][0] for g in unique}
    ordered   = sorted(unique, key=lambda g: first_idx[g])
    return [int(counts[np.where(unique == g)]) for g in ordered]

# 4) Split out 10% of customers for final test
gss = GroupShuffleSplit(n_splits=1, test_size=0.1, random_state=42)
train_val_idx, test_idx = next(gss.split(X, y, groups))

# 5) Build DMatrix objects
dtrain = xgb.DMatrix(X[train_val_idx], label=y[train_val_idx], feature_names=FEATURES)
dtrain.set_group(get_group_sizes(train_val_idx, groups))

dtest  = xgb.DMatrix(X[test_idx], label=y[test_idx], feature_names=FEATURES)
dtest.set_group(get_group_sizes(test_idx, groups))

# 6) Train with early stopping
params = {
    'learning_rate':    0.05904156033397954,
    'max_depth':        9,
    'subsample':        0.7381373340225315,
    'colsample_bytree': 0.8015406020825335,
    'gamma':            2.6080887090749587,
    'min_child_weight': 1,
    'objective':        'rank:pairwise',
    'eval_metric':      'ndcg@3',
    'verbosity':        0,
    'seed':             42,
}
num_round = 249

evals_result = {}
bst = xgb.train(
    params,
    dtrain,
    num_boost_round=num_round,
    evals=[(dtrain, 'train'), (dtest, 'test')],
    early_stopping_rounds=10,
    evals_result=evals_result,
    verbose_eval=False
)

# 7) Final evaluation on held-out test customers
preds_test = bst.predict(dtest)
print(f"Test NDCG@3: {ndcg_score([y[test_idx]], [preds_test], k=3):.4f}")

def precision_at_k(groups, true_rel, pred_scores, k):
    dfm = pd.DataFrame({'grp': groups, 'rel': true_rel, 'score': pred_scores})
    prec = []
    for _, g in dfm.groupby('grp'):
        topk = g.nlargest(k, 'score')
        prec.append((topk['rel'] > 0).sum() / k)
    return np.mean(prec)

for k in [1, 3, 5]:
    print(f"Precision@{k}: {precision_at_k(groups[test_idx], y[test_idx], preds_test, k):.3f}")

# 8) Safe feature-importance plotting
gain_imp   = bst.get_score(importance_type='gain')
weight_imp = bst.get_score(importance_type='weight')

if gain_imp:
    fig, ax = plt.subplots()
    xgb.plot_importance(bst, ax=ax, importance_type='gain', title="Feature Importance (gain)")
    plt.tight_layout()
    plt.show()

elif weight_imp:
    imp_df = (
        pd.DataFrame({
            'feature': list(weight_imp.keys()),
            'weight':  list(weight_imp.values()),
        })
        .sort_values('weight', ascending=True)
    )
    fig, ax = plt.subplots()
    imp_df.plot.barh(x='feature', y='weight', legend=False, ax=ax)
    ax.set_title("Feature Importance (split count)")
    plt.tight_layout()
    plt.show()

else:
    print("Warning: no feature importances available (gain and weight are both empty).")

# 9) Plot learning curves
plt.figure()
plt.plot(evals_result['train']['ndcg@3'], label='train')
plt.plot(evals_result['test']['ndcg@3'],  label='test')
plt.xlabel('Boosting Round')
plt.ylabel('NDCG@3')
plt.title('Learning Curves')
plt.legend()
plt.tight_layout()
plt.show()
