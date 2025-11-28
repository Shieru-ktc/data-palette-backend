from fastapi import FastAPI
import pandas as pd
import numpy as np
import os
from schemas import DefensiveRangeResponse

app = FastAPI()

CSV_FILE_PATH = "dummy_data.csv"

@app.get("/get_defensive_range", response_model=DefensiveRangeResponse)
async def get_defensive_range():
    
    if not os.path.exists(CSV_FILE_PATH):
        return DefensiveRangeResponse(
            status="error",
            effective_range=0.0,
            message="データファイルがありません"
        )

    df = pd.read_csv(CSV_FILE_PATH)
    caught_df = df[df['is_caught'].astype(str).str.lower() == 'true']

    if caught_df.empty:
        return DefensiveRangeResponse(
            status="success",
            effective_range=0.0,
            message="データなし"
        )

    distances = caught_df['distance'].values
    distances.sort()
    effective_range = np.percentile(distances, 95)

    return DefensiveRangeResponse(
        status="success",
        effective_range=round(effective_range, 2),
        message="Analysis Complete"
    )