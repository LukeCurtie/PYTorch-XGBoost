import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb

# --- Helper: build a DMatrix with proper group‐wise row shuffling ----------
def make_dmatrix(df_, features):
    grp_labels = df_['Cust_Label'].values
    # determine each group’s first appearance in df_
    first_pos = {g: np.where(grp_labels == g)[0][0] for g in np.unique(grp_labels)}
    ordered_uids = sorted(first_pos, key=lambda g: first_pos[g])
    # collect & shuffle rows *within* each group
    group_indices = [np.where(grp_labels == uid)[0].tolist() for uid in ordered_uids]
    for idxs in group_indices:
        np.random.shuffle(idxs)
    # flatten so rows are contiguous by group (but internally shuffled)
    flat_idx = [i for idxs in group_indices for i in idxs]
    X = df_[features].values[flat_idx]
    y = df_['Rel'].values[flat_idx]
    dmat = xgb.DMatrix(X, label=y)
    dmat.set_group([len(idxs) for idxs in group_indices])
    return dmat

# --- 1) Load & feature engineering ----------------------------------------
df = pd.read_csv('./data.csv', parse_dates=['Date_of_Purchase'])
df['Purch_DOW']   = df['Date_of_Purchase'].dt.weekday
df['Purch_Month'] = df['Date_of_Purchase'].dt.month

# keep original IDs for later
df['Customer_ID_str'] = df['Customer_ID']
df['Product_str']     = df['Product']

# --- 2) Encode categoricals & build inverse maps --------------------------
df['Cust_Label'] = LabelEncoder().fit_transform(df['Customer_ID_str'])
df['Prod_Label'] = LabelEncoder().fit_transform(df['Product_str'])
for col in ['Gender','Age_Group','Region']:
    df[col] = LabelEncoder().fit_transform(df[col])

cust_map = dict(zip(df['Cust_Label'], df['Customer_ID_str']))
prod_map = dict(zip(df['Prod_Label'], df['Product_str']))
all_prods = df['Prod_Label'].unique()

# --- 3) Sample negatives for every user (3× positives) -----------------------
def sample_negatives(pos_df, neg_mult):
    negs = []
    for uid, grp in pos_df.groupby('Cust_Label'):
        purchased = set(grp['Prod_Label'])
        candidates = np.setdiff1d(all_prods, list(purchased))
        picks = np.random.choice(candidates, size=len(grp)*neg_mult, replace=False)
        for p in picks:
            negs.append({
                'Cust_Label':    uid,
                'Customer_ID':   cust_map[uid],
                'Prod_Label':    p,
                'Product':       prod_map[p],
                'Annual_Income': grp['Annual_Income'].iloc[0],
                'Gender':        grp['Gender'].iloc[0],
                'Age_Group':     grp['Age_Group'].iloc[0],
                'Region':        grp['Region'].iloc[0],
                'Purch_DOW':     int(grp['Purch_DOW'].mean()),
                'Purch_Month':   int(grp['Purch_Month'].mean()),
                'Frequency':     0
            })
    return pd.DataFrame(negs)

df_neg    = sample_negatives(df, neg_mult=3)
df_all    = pd.concat([df, df_neg], ignore_index=True)
df_all['Rel'] = (df_all['Frequency'] > 0).astype(int)

# --- 4) 90/10 customer‐level split -----------------------------------------
FEATURES = [
    'Annual_Income','Purch_DOW','Purch_Month',
    'Gender','Age_Group','Region','Prod_Label'
]
X      = df_all[FEATURES].values
y      = df_all['Rel'].values
groups = df_all['Cust_Label'].values

gss = GroupShuffleSplit(n_splits=1, test_size=0.10, random_state=42)
train_idx, valid_idx = next(gss.split(X, y, groups))

df_train = df_all.iloc[train_idx].reset_index(drop=True)
df_valid = df_all.iloc[valid_idx].reset_index(drop=True)

# --- 5) Build DMatrix’s (with per‐group shuffling) --------------------------
dtrain = make_dmatrix(df_train, FEATURES)
dvalid = make_dmatrix(df_valid, FEATURES)

# --- 6) Train with early stopping -----------------------------------------
params = {
    'objective':   'rank:pairwise',
    'eval_metric': 'ndcg@5',
    'eta':          0.1,
    'max_depth':    6,
    'seed':        42,
    'verbosity':    1,
}

bst = xgb.train(
    params,
    dtrain,
    num_boost_round=500,
    evals=[(dtrain, 'train'), (dvalid, 'valid')],
    early_stopping_rounds=20,
    verbose_eval=True,
)

# --- 7) Inference: top‐5 for a held‐out user -------------------------------
# pick one customer from the validation fold
held_uid = df_valid['Cust_Label'].iloc[0]
user_info = df_all[df_all['Cust_Label'] == held_uid].iloc[0]

candidates = []
for p in all_prods:
    candidates.append({
        'Cust_Label':    held_uid,
        'Customer_ID':   cust_map[held_uid],
        'Prod_Label':    p,
        'Product':       prod_map[p],
        'Annual_Income': user_info['Annual_Income'],
        'Gender':        user_info['Gender'],
        'Age_Group':     user_info['Age_Group'],
        'Region':        user_info['Region'],
        'Purch_DOW':     int(user_info['Purch_DOW']),
        'Purch_Month':   int(user_info['Purch_Month']),
    })

cand_df = pd.DataFrame(candidates)
dcand   = xgb.DMatrix(cand_df[FEATURES].values)
cand_df['Score'] = bst.predict(dcand)

top5 = cand_df.sort_values('Score', ascending=False).head(5)
print("\nTop-5 recommendations for held‐out user:\n", 
      top5[['Customer_ID','Product','Score']])
