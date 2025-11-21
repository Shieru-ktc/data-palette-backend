import pandas as pd

import joblib
import tensorflow as tf
import math
from fastapi import FastAPI, HTTPException
from schemas import BattedBallData 

print("--- モデル読み込み中... ---")
model = tf.keras.models.load_model('baseball_model.keras')
scaler = joblib.load('scaler.joblib')
model_columns = joblib.load('model_columns.joblib')
print("--- 完了 ---")

app = FastAPI()

@app.post("/predict")
async def predict_catch(data: BattedBallData): 
    if model is None:
        raise HTTPException(status_code=500, detail="モデルがありません")

    try:
        # 前提: バッターボックスが (0,0) であること
        
        # 1. 飛距離 
        distance = math.sqrt(data.ball_x**2 + data.ball_z**2)
        
        # 2. 方向 
        direction = math.degrees(math.atan2(data.ball_x, data.ball_z))

        # 3. 打球タイプ (Unityから送ってもらうのがいいかも、なければ高さから簡易判定する)
        if distance > 0 and (data.ball_y / distance) > 0.5:
            ball_type = "フライ"  
        else:
            ball_type = "ライナー"

        input_df = pd.DataFrame([{
            'batted_ball_distance': distance,
            'batted_ball_direction': direction,
            'batted_ball_type': ball_type
        }])

        # 前処理
        input_df_encoded = pd.get_dummies(input_df)
        input_df_encoded = input_df_encoded.reindex(columns=model_columns, fill_value=0)
        input_features = scaler.transform(input_df_encoded)

        # 予測
        prediction_prob = model.predict(input_features)[0][0]
        
        # 0.5以上なら安打(捕れない)、未満なら凡打(捕れる)
        is_hit = prediction_prob > 0.5 
        
        return {
            "prediction": "安打" if is_hit else "凡打",
            "is_caught": not is_hit, # Unityはこれを見る
            "hit_probability": float(prediction_prob),
            "calc_distance": distance
        }

    except Exception as e:
        raise HTTPException(status_code=500, detail=f"予測エラー: {str(e)}")