import pandas as pd
import joblib
import numpy as np 
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def classify_pa_result(result):
    res_str = str(result)
    hit_keywords = ['安', '２', '３', '本', 'H', '2B', '3B', 'HR']
    if any(keyword in res_str for keyword in hit_keywords): return 1 # 安打(捕れない)
    
    out_keywords = ['ゴロ', '飛', '直', '振', '併', '邪', '犠'] 
    if any(keyword in res_str for keyword in out_keywords): return 0 # 凡打(捕れる)
    
    return None

def main():
    # 1. データの読み込み
    try:
        df = pd.read_csv('3_イベントデータ_野球.csv', encoding='utf-8')
    except UnicodeDecodeError:
        df = pd.read_csv('3_イベントデータ_野球.csv', encoding='cp932')


    features = ['batted_ball_distance', 'batted_ball_type', 'batted_ball_direction'] 
    
    df['Target'] = df['pa_result'].apply(classify_pa_result)
    train_data = df.dropna(subset=['Target'])

        # 特徴量に欠損がある行は削除
    train_data = train_data.dropna(subset=features)

    if len(train_data) == 0:
        print("エラー: 学習データ0件")
        return

    print(f"学習データ数: {len(train_data)} 件")

    # 2. 前処理
    X = train_data[features].copy()
    y = train_data['Target'].astype(int)


    # 飛距離や方向が空なら平均値で埋める
    X['batted_ball_distance'] = X['batted_ball_distance'].fillna(X['batted_ball_distance'].mean())
    X['batted_ball_direction'] = X['batted_ball_direction'].fillna(X['batted_ball_direction'].mean())
    X['batted_ball_type'] = X['batted_ball_type'].fillna('Unknown')

    # One-Hot Encoding (打球タイプ: ゴロ、フライなどを数値化)
    X_encoded = pd.get_dummies(X, drop_first=True)

    # スケーリング
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # 3. 分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 4. モデル構築
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(16, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    # 5. 学習
    model.fit(X_train, y_train, epochs=50, batch_size=8, verbose=1)

    # 6. 保存
    model.save('baseball_model.keras')
    joblib.dump(scaler, 'scaler.joblib')
    model_columns = list(X_encoded.columns)
    joblib.dump(model_columns, 'model_columns.joblib')
    
    print("保存完了！")

if __name__ == "__main__":
    main()