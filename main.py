import numpy as np
import joblib
import tensorflow as tf
from fastapi import FastAPI
from pydantic import BaseModel

# 1. AIモデルとスケーラーを起動時に読み込む
print("--- AIモデルとスケーラーを読み込んでいます... ---")
try:
    model = tf.keras.models.load_model('baseball_model.keras')
    scaler = joblib.load('scaler.joblib')
    print("--- 読み込み完了 ---")
except IOError:
    print("エラー: 'baseball_model.keras' または 'scaler.joblib' が見つかりません。")
    print("先に `train.py` を実行して、モデルファイルを作成してください。")
    model = None
    scaler = None

# 2. Pydanticモデル（リクエストの型定義）
class PlayData(BaseModel):
    # Unity側で「この座標に落ちた」と判断した座標
    ball_x: float
    ball_z: float
    
    # Unity側で「この選手が捕球対象」と判断した選手の座標
    player_x: float
    player_z: float
    
    # Unity側で計算した打球の滞空時間
    time_in_air: float


app = FastAPI()

@app.get("/")
def read_root():
    return {"Hello": "Data Palette Backend"}


# 予測APIエンドポイント
@app.post("/predict")
async def predict_catch(data: PlayData):
    """
    Unityからの打球・選手データを受け取り、捕球確率を予測するAPI
    """
    if model is None or scaler is None:
        return {"error": "モデルが読み込まれていません。"}, 500

    try:
        # Unityからのデータを使って「特徴量」を計算
        distance_to_ball = np.sqrt(
            (data.player_x - data.ball_x)**2 +
            (data.player_z - data.ball_z)**2
        )
        
        safe_time_in_air = max(data.time_in_air, 0.0001)
        required_speed = distance_to_ball / safe_time_in_air

        #データをNumpy配列（2次元）に変換
        features = np.array([[distance_to_ball, required_speed]])

        #スケーリング
        features_scaled = scaler.transform(features)

        #AIモデルで予測
        probability = model.predict(features_scaled)[0][0]

        #結果をUnityに返す
        return {
            "catch_probability": float(probability),
            "calculated_distance": float(distance_to_ball),
            "calculated_speed": float(required_speed)
        }
    
    except Exception as e:
        return {"error": f"予測中にエラーが発生しました: {str(e)}"}, 500