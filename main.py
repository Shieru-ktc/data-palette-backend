from fastapi import FastAPI
from schemas import DefensiveRangeResponse
import pandas as pd
import numpy as np
import os

app = FastAPI()


CSV_FILE_PATH = "result.csv"

@app.get("/get_defensive_range", response_model=DefensiveRangeResponse)
async def get_defensive_range():
    
    if not os.path.exists(CSV_FILE_PATH):
        return DefensiveRangeResponse(
            status="error", effective_range=0.0, message="CSVファイルが見つかりません"
        )

    df = pd.read_csv(CSV_FILE_PATH, dtype={'catch_result': 'boolean'})

    #catch_resultが True のデータだけを抽出
    caught_df = df[df['catch_result']]

    if caught_df.empty:
        return DefensiveRangeResponse(
            status="success", effective_range=0.0, message="データなし"
        )

    #distance_from_startの列を取り出す
    distances = caught_df['distance_from_start'].values

    distances.sort()
    effective_range = np.percentile(distances, 95)

    return DefensiveRangeResponse(
        status="success",
        effective_range=round(effective_range, 2),
        message="Analysis Complete"
    )