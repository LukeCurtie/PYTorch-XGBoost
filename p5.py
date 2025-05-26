"""
p6_refined.py: Modular synthetic data ranking pipeline with enhanced feature engineering,
negative sampling, hyperparameter tuning, and evaluation.
"""
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import GroupShuffleSplit
import xgboost as xgb
from sklearn.metrics import ndcg_score

def generate_synthetic_data(n_customers=50,
                             n_products=15,
                             n_interactions=500,
                             seed=42):
    """
    Create a synthetic purchase log with realistic distributions:
    - Zipf-like popularity for products
    - Category-conditional price distributions
    - Timestamp uniformly over one year
    """
    np.random.seed(seed)
    # IDs
    customers = [f"CUST_{i:03d}" for i in range(n_customers)]
    products = [f"PROD_{i:03d}" for i in range(n_products)]

    # Zipf for product popularity
    ranks = np.arange(1, n_products+1)
    weights = 1.0 / ranks
    prod_probs = weights / weights.sum()

    categories = ['Electronics','Clothing','Home','Sports']
    brands     = ['BrandA','BrandB','BrandC','BrandD']

    rows = []
    start = pd.to_datetime('2024-01-01')
    for _ in range(n_interactions):
        cid = np.random.choice(customers)
        pid = np.random.choice(products, p=prod_probs)
        date = start + pd.to_timedelta(np.random.randint(0, 365), unit='d')
        category = np.random.choice(categories)
        price = np.random.normal(loc={'Electronics':500,'Clothing':100,'Home':200,'Sports':150}[category], scale=50)
        price = max(5, round(price,2))
        rows.append({
            'Customer_ID': cid,
            'Product': pid,
            'Date_of_Purchase': date,
            'Frequency': np.random.randint(1, 6),
            'Annual_Income': np.random.choice([30000,50000,70000,90000]),
            'Gender': np.random.choice(['Male','Female']),
            'Age_Group': np.random.choice(['18-25','26-35','36-45','46-55']),
            'Region': np.random.choice(['North','South','East','West']),
            'Price': price,
            'Category': category,
            'Brand': np.random.choice(brands)
        })
    df = pd.DataFrame(rows)
    today = pd.to_datetime('2025-05-24')
    df['Days_Since_Last_Purchase'] = (today - df['Date_of_Purchase']).dt.days
    return df


def encode_labels(df, fields):
    """Label-encode specified categorical fields in-place."""
    for col in fields:
        lbl = LabelEncoder()
        df[f"{col}_Label"] = lbl.fit_transform(df[col])
    return df


def compute_user_rfm(df):
    """
    Compute per-user Recency, Frequency, Monetary aggregate features.
    """
    df['Monetary'] = df['Price'] * df['Frequency']
    user = df.groupby('Customer_ID').agg(
        Recency=('Days_Since_Last_Purchase','min'),
        Frequency=('Frequency','sum'),
        Monetary=('Monetary','sum')
    ).reset_index()
    return df.merge(user, on='Customer_ID', how='left')


def sample_negatives(df_pos, all_prod_labels, neg_mult=3, seed=42):
    """
    Stratified negative sampling by user with popularity weighting.
    """
    np.random.seed(seed)
    negs = []
    pop = df_pos['Product_Label'].value_counts(normalize=True)
    for uid, grp in df_pos.groupby('Customer_ID_Label'):
        purchased = set(grp['Product_Label'])
        candidates = [p for p in all_prod_labels if p not in purchased]
        weights = np.array([pop.get(p, 0) for p in candidates]) + 1e-6
        weights /= weights.sum()
        n = min(len(candidates), len(grp)*neg_mult)
        picks = np.random.choice(candidates, size=n, replace=False, p=weights)
        for p in picks:
            base = grp.iloc[0]
            negs.append({
                'Customer_ID_Label': uid,
                'Customer_ID': base['Customer_ID'],
                'Product_Label': p,
                'Frequency': 0,
                'Annual_Income': base['Annual_Income'],
                'Gender_Label': base['Gender_Label'],
                'Age_Group_Label': base['Age_Group_Label'],
                'Region_Label': base['Region_Label'],
                'Price': df_pos[df_pos['Product_Label']==p]['Price'].mean(),
                'Category_Label': df_pos[df_pos['Product_Label']==p]['Category_Label'].mode()[0],
                'Brand_Label': df_pos[df_pos['Product_Label']==p]['Brand_Label'].mode()[0],
                'Days_Since_Last_Purchase': df_pos[df_pos['Product_Label']==p]['Days_Since_Last_Purchase'].mean()
            })
    return pd.DataFrame(negs)


def make_group_dmatrix(df_, features):
    """
    Build XGBoost Group DMatrix for ranking.
    """
    grp = df_['Customer_ID_Label'].values
    order = np.argsort(grp)
    X = df_.iloc[order][features]
    y = df_.iloc[order]['Rel']
    groups = [len(group) for _, group in df_.groupby('Customer_ID_Label')]
    dmat = xgb.DMatrix(X, label=y)
    dmat.set_group(groups)
    return dmat


def train_ranker(dtrain, dvalid, params=None, num_round=500):
    """
    Train XGBoost pairwise ranker with early stopping.
    """
    params = params or {
        'objective':'rank:ndcg',
        'eval_metric':'ndcg@5',
        'eta':0.1,
        'max_depth':6,
        'seed':42,
        'verbosity':1
    }
    evals = [(dtrain,'train'), (dvalid,'valid')]
    return xgb.train(params, dtrain, num_boost_round=num_round,
                     evals=evals, early_stopping_rounds=20)


def evaluate_model(bst, df_va, FEATURES, k=5):
    """
    Compute NDCG@k and Precision@k on held-out set per user.
    """
    scores, precs = [], []
    for uid, grp in df_va.groupby('Customer_ID_Label'):
        Xg = grp[FEATURES]
        y_true = grp['Rel'].values
        y_pred = bst.predict(xgb.DMatrix(Xg))
        idx = np.argsort(y_pred)[::-1][:k]
        scores.append(ndcg_score([y_true],[y_pred], k=k))
        precs.append((y_true[idx] > 0).sum() / k)
    print(f"Mean NDCG@{k}:", np.mean(scores))
    print(f"Mean Precision@{k}:", np.mean(precs))


def inference_for_user(bst, df_all, uid, FEATURES):
    """
    Score unseen products for user ID.
    """
    user = df_all[df_all['Customer_ID_Label']==uid].iloc[0]
    seen = set(df_all[df_all['Customer_ID_Label']==uid]['Product_Label'])
    candidates = [p for p in df_all['Product_Label'].unique() if p not in seen]
    rows = []
    for p in candidates:
        feat = {}
        for f in FEATURES:
            feat[f] = user[f] if f in user else df_all[df_all['Product_Label']==p][f].mean()
        rows.append(feat)
    df_cand = pd.DataFrame(rows)
    df_cand['Score'] = bst.predict(xgb.DMatrix(df_cand[FEATURES]))
    return df_cand.sort_values('Score', ascending=False).head(5)

if __name__ == '__main__':
    # Generate & prepare data
    df = generate_synthetic_data()
    df = encode_labels(df, ['Customer_ID','Product','Gender','Age_Group','Region','Category','Brand'])
    df = compute_user_rfm(df)
    all_prods = df['Product_Label'].unique()

    # Positives & negatives
    df_pos = df.copy()
    df_neg = sample_negatives(df_pos, all_prods)
    df_all = pd.concat([df_pos.assign(Rel=1), df_neg.assign(Rel=0)], ignore_index=True)

    # Split
    gss = GroupShuffleSplit(test_size=0.1, random_state=42)
    tr_idx, va_idx = next(gss.split(df_all, df_all['Rel'], df_all['Customer_ID_Label']))
    df_tr, df_va = df_all.iloc[tr_idx], df_all.iloc[va_idx]

    FEATURES = ['Annual_Income','Gender_Label','Age_Group_Label','Region_Label',
                'Price','Category_Label','Brand_Label',
                'Days_Since_Last_Purchase','Recency','Frequency_y','Monetary_y']

    # Build DMatrices
    dtrain = make_group_dmatrix(df_tr, FEATURES)
    dvalid = make_group_dmatrix(df_va, FEATURES)

    # Train & eval
    bst = train_ranker(dtrain, dvalid)
    evaluate_model(bst, df_va, FEATURES)

    # Inference
    uid = df_va['Customer_ID_Label'].iloc[0]
    print("\nTop-5 recommendations:")
    print(inference_for_user(bst, df_all, uid, FEATURES))
