from pydantic import BaseModel

# Unityに返すデータ
class DefensiveRangeResponse(BaseModel):
    status: str          
    effective_range: float # 計算結果 (m)
    message: str         