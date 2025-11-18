import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

# 1. 起動時の読み込み
print("--- TensorFlowモデルを読み込んでいます... ---")
try:
    # モデル
    model = tf.keras.models.load_model('baseball_model.keras')
    # スケーラー
    scaler = joblib.load('scaler.joblib')
    # 列定義
    model_columns = joblib.load('model_columns.joblib')
    print("--- 読み込み完了 ---")
except Exception as e:
    print(f"エラー: ファイルが見つかりません。train.pyを実行してください。詳細: {e}")
    model = None

# リクエスト型定義
class PitchData(BaseModel):
    pitch_speed: float
    pitch_type_name: str
    ball_x: float
    ball_y: float
    ball_z: float

app = FastAPI()

# ゾーン判定ロジック (簡易版)
def calculate_zone(x, y, z):
    if 0.2 < x < 0.5: return "1"
    if -0.2 < x <= 0.2: return "5"
    return "13"

@app.post("/predict")
async def predict_catch(data: PitchData):
    if model is None:
        return {"error": "サーバー準備中: モデルがありません"}, 500

    try:
        # 1. データフレーム作成
        zone = calculate_zone(data.ball_x, data.ball_y, data.ball_z)
        
        input_df = pd.DataFrame([{
            'pitch_speed': data.pitch_speed,
            'pitch_type_name': data.pitch_type_name,
            'pitch_zone': zone
        }])

        # 2. 前処理 (train.pyと同じ手順)
        input_df_encoded = pd.get_dummies(input_df)
        input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        input_features = scaler.transform(input_df_encoded)

        # 3. 予測 (TensorFlow)
        prediction_prob = model.predict(input_features)[0][0]
        
        is_hit = prediction_prob > 0.5 
        result_text = "安打" if is_hit else "凡打"
        
        # 捕球成功 = 凡打
        is_caught = not is_hit

        return {
            "prediction": result_text,
            "is_caught": is_caught, 
            "hit_probability": float(prediction_prob), 
            "used_zone": zone
        }

    except Exception as e:
        return {"error": f"予測エラー: {str(e)}"}, 500