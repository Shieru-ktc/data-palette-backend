from fastapi import FastAPI
from pydantic import BaseModel
import pandas as pd
import numpy as np
import os
from sklearn.linear_model import LinearRegression

app = FastAPI()
CSV_FILE_PATH = "result.csv"

class TimeToDistanceRequest(BaseModel):
    time_elapsed: float 

class TimeToDistanceResponse(BaseModel):
    status: str
    predicted_distance: float 
    message: str


def train_model_and_predict(target_time):
    if not os.path.exists(CSV_FILE_PATH):
        return None

    df = pd.read_csv(CSV_FILE_PATH, dtype={'catch_result': 'boolean'})
    
    # 1. 捕れたデータだけにする
    caught_df = df[df['catch_result']]
    if caught_df.empty:
        return None

    # 2. データを「速さ」で評価して、上位データを抽出する
    caught_df['speed'] = caught_df['distance_from_start'] / caught_df['land_time_sec']
    
    # 上位20%のスピードのデータだけ抜き出す
    threshold_speed = caught_df['speed'].quantile(0.80) 
    top_tier_df = caught_df[caught_df['speed'] >= threshold_speed]
    if len(top_tier_df) < 2:
        top_tier_df = caught_df

    # 3. AI (線形回帰) に学習させる
    X = top_tier_df[['land_time_sec']] # 入力
    y = top_tier_df['distance_from_start'] # 正解

    model = LinearRegression()
    model.fit(X, y) 
    prediction = model.predict([[target_time]])
    return prediction[0]


@app.post("/predict_distance", response_model=TimeToDistanceResponse)
async def predict_distance(req: TimeToDistanceRequest):
    
    predicted_dist = train_model_and_predict(req.time_elapsed)

    if predicted_dist is None:
        return TimeToDistanceResponse(
            status="error", predicted_distance=0.0, message="データ不足"
        )
    
    # マイナスになったら0にする
    final_distance = max(0.0, predicted_dist)

    return TimeToDistanceResponse(
        status="success",
        predicted_distance=round(final_distance, 2),
        message=f"AI予測: {req.time_elapsed}秒あれば {round(final_distance, 2)}m 移動できます"
    )