from pydantic import BaseModel

# リクエストの型定義
class BattedBallData(BaseModel):
    ball_x: float
    ball_y: float
    ball_z: float