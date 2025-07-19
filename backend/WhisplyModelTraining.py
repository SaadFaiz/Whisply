import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier

MODEL_PATH = "./pretrained_whisply_models/Whisply_model.pkl"

# ─── training data ─────────────────────────────
data = [
    {"avg_logprob": -0.206, "has_timestamp": 1,  "speech_prob": 0.9,  "human_label": "Y"},
    {"avg_logprob": -0.353, "has_timestamp": 1,  "speech_prob": 0.9,  "human_label": "Y"},
    {"avg_logprob": -0.393, "has_timestamp": 1,  "speech_prob": 0.1,  "human_label": "Y"},
    {"avg_logprob": -0.393, "has_timestamp": 0, "speech_prob": 0.05, "human_label": "Y"},
    {"avg_logprob": -0.487, "has_timestamp": 0, "speech_prob": 0.1,  "human_label": "Y"},
    {"avg_logprob": -0.250, "has_timestamp": 1,  "speech_prob": 0.7,  "human_label": "Y"},
    {"avg_logprob": -0.390, "has_timestamp": 1,  "speech_prob": 0.05, "human_label": "Y"},
    {"avg_logprob": -0.447, "has_timestamp": 1,  "speech_prob": 0.3,  "human_label": "Y"},
    {"avg_logprob": -0.916, "has_timestamp": 1,  "speech_prob": 0.4, "human_label": "Y"},
    {"avg_logprob": -0.479, "has_timestamp": 0, "speech_prob": 0.05, "human_label": "Y"},
    {"avg_logprob": -0.510, "has_timestamp": 0, "speech_prob": 0.05, "human_label": "Y"},
    {"avg_logprob": -0.305, "has_timestamp": 1,  "speech_prob": 0.9,  "human_label": "Y"},
    {"avg_logprob": -0.315, "has_timestamp": 1,  "speech_prob": 0.9,  "human_label": "Y"},
    {"avg_logprob": -0.236, "has_timestamp": 1,  "speech_prob": 0.9,  "human_label": "Y"},
    {"avg_logprob": -0.193, "has_timestamp": 1,  "speech_prob": 0.9,  "human_label": "Y"},
    {"avg_logprob": -0.174, "has_timestamp": 1,  "speech_prob": 0.9,  "human_label": "Y"},
    {"avg_logprob": -0.920, "has_timestamp": 0, "speech_prob": 0.05, "human_label": "N"},
    {"avg_logprob": -0.248, "has_timestamp": 1,  "speech_prob": 0.9,  "human_label": "Y"},
    {"avg_logprob": -0.303, "has_timestamp": 1,  "speech_prob": 0.9,  "human_label": "Y"},
    {"avg_logprob": -1.628, "has_timestamp": 0, "speech_prob": 0.4,  "human_label": "N"},
    {"avg_logprob": -0.115, "has_timestamp": 0, "speech_prob": 0.1,  "human_label": "N"},
    {"avg_logprob": -0.8,   "has_timestamp": 0, "speech_prob": 0.05, "human_label": "N"},
    {"avg_logprob": -0.755, "has_timestamp": 0, "speech_prob": 0.2,  "human_label": "N"},
    {"avg_logprob": -0.8,   "has_timestamp": 0, "speech_prob": 0.05, "human_label": "N"},
    {"avg_logprob": -1.675, "has_timestamp": 0, "speech_prob": 0.05, "human_label": "N"},
    {"avg_logprob": -1.075, "has_timestamp": 0, "speech_prob": 0.05, "human_label": "N"},
    {"avg_logprob": -1.189, "has_timestamp": 0, "speech_prob": 0.05, "human_label": "N"},
    {"avg_logprob": -0.983, "has_timestamp": 0, "speech_prob": 0.05, "human_label": "N"},
    {"avg_logprob": -0.948, "has_timestamp": 0, "speech_prob": 0.05, "human_label": "N"},
    {"avg_logprob": -0.493, "has_timestamp": 0, "speech_prob": 0.05, "human_label": "N"},
    {"avg_logprob": -0.374, "has_timestamp": 0, "speech_prob": 0.05, "human_label": "N"},
]

df = pd.DataFrame(data)

# Ensure correct dtypes (optional)
df['speech_prob']   = df['speech_prob'].astype(float)  # float
df['has_timestamp'] = df['has_timestamp'].astype(int)  # 0 or 1

# Feature engineering:
df['logprob_speech_interaction'] = df['avg_logprob'] * df['speech_prob']
df['no_timestamp_low_speech']    = (
    (df['has_timestamp'] == 0) &
    (df['speech_prob'] <= 0.1)
).astype(int)
df['confidence_score']           = (
    df['avg_logprob'] *
    df['speech_prob'] *
    (1 + df['has_timestamp'])
)

features = [
    'avg_logprob',
    'has_timestamp',
    'speech_prob',
    'logprob_speech_interaction',
    'no_timestamp_low_speech',
    'confidence_score'
]
X_train = df[features]
y_train = df['human_label']

# ─── Train the RandomForest model ────────────────────────────────────

model = RandomForestClassifier(
    random_state=42,
    n_estimators=200,
    max_depth=5,
    min_samples_split=5,
    class_weight='balanced',
    max_features='sqrt'
)
model.fit(X_train, y_train)
joblib.dump(model, MODEL_PATH)