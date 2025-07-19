import joblib
import pandas as pd

model = joblib.load("./pretrained_whisply_models/Whisply_model_V1.pkl")

# ─── Prediction function for new samples ─────────────────────────────────────
def speech_exist(sample_dict):
    """
    sample_dict expects numeric fields:
      - 'avg_logprob': float
      - 'has_timestamp': int (0 or 1)
      - 'speech_prob': float
    Returns 'Y' or 'N'.
    """
    df = pd.DataFrame([sample_dict])

    # Ensure numeric dtypes
    df['speech_prob']   = df['speech_prob'].astype(float)
    df['has_timestamp'] = df['has_timestamp'].astype(int)

    # Recompute features:
    df['logprob_speech_interaction'] = df['avg_logprob'] * df['speech_prob']
    df['no_timestamp_low_speech']    = (
        (df['has_timestamp'] == 0) &
        (df['speech_prob'] <= 0.1)
    ).astype(int)
    df['confidence_score'] = (
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

    X = df[features]
    return model.predict(X)[0]

