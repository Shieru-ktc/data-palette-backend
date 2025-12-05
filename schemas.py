from pydantic import BaseModel

# Unityから送るデータ (リクエスト)
class DefensiveRangeRequest(BaseModel):
    target_time_sec: float # 滞空時間

# Unityに返すデータ (レスポンス)
class DefensiveRangeResponse(BaseModel):
    status: str
    player_max_speed: float # 分析した秒速 (m/s)
    effective_range: float  # 計算された守備範囲 (m)
    message: str

class TimeToDistanceRequest(BaseModel):
    time_elapsed: float  # 経過時間 (秒)

class TimeToDistanceResponse(BaseModel):
    status: str
    predicted_distance: float # AIが予測した距離 (m)
    message: str