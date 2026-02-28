import os
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import MiniBatchKMeans
from config import DATA_PROCESSED
from customer_features import build_customer_features

RANDOM_STATE = 42


def cluster_customers(n_clusters: int = 4, use_price: bool = True):
    """
    Build customer features and run MiniBatchKMeans clustering.

    Output:
      data/processed/customer_segments.csv
    """
    print("[clustering] Building customer features...")
    features = build_customer_features(use_price=use_price)

    X = features.values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    print("[clustering] Running MiniBatchKMeans...")
    model = MiniBatchKMeans(
        n_clusters=n_clusters,
        batch_size=2048,
        random_state=RANDOM_STATE
    )
    labels = model.fit_predict(X_scaled)

    features_clustered = features.copy()
    features_clustered["cluster"] = labels

    out_path = os.path.join(DATA_PROCESSED, "customer_segments.csv")
    features_clustered.to_csv(out_path, index=True)  # index=user_id
    print(f"[clustering] customer_segments saved → {out_path}")
    print("[clustering] Shape:", features_clustered.shape)

    return features_clustered, model, scaler