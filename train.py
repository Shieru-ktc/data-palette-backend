import pandas as pd
import numpy as np
import joblib
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

def classify_pa_result(result):
    res_str = str(result)
    hit_keywords = ['安', '２', '３', '本', 'H', '2B', '3B', 'HR']
    if any(keyword in res_str for keyword in hit_keywords): return 1 # 安打=1
    if '犠飛' in res_str or '犠打' in res_str: return None
    if any(keyword in res_str for keyword in ['四球', '死球', '失', '妨', '振逃']): return None
    out_keywords = ['ゴロ', '飛', '直', '振', '併', '邪']
    if any(keyword in res_str for keyword in out_keywords): return 0 # 凡打=0
    return None

def main():
    print("TensorFlow Version:", tf.__version__)

    # 1. データの読み込み
    try:
        df = pd.read_csv('3_イベントデータ_野球.csv', encoding='utf-8')
    except UnicodeDecodeError:
        try:
            df = pd.read_csv('3_イベントデータ_野球.csv', encoding='cp932')
        except Exception:
            return

    features = ['pitch_zone', 'pitch_type_name', 'pitch_speed'] 
    
    df['Target'] = df['pa_result'].apply(classify_pa_result)
    train_data = df.dropna(subset=['Target'])

    if len(train_data) == 0:
        print("エラー: 学習データ0件")
        return

    print(f"学習データ数: {len(train_data)} 件")

    # 2. 前処理
    X = train_data[features].copy()
    y = train_data['Target'].astype(int) 

    # 欠損値埋め
    X['pitch_speed'] = X['pitch_speed'].fillna(X['pitch_speed'].median())
    X['pitch_type_name'] = X['pitch_type_name'].fillna('Unknown')
    X['pitch_zone'] = X['pitch_zone'].astype(str).fillna('Unknown')

    # One-Hot Encoding
    X_encoded = pd.get_dummies(X, drop_first=True)

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X_encoded)

    # 3. 分割
    X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

    # 4. TensorFlow モデル構築
    model = tf.keras.Sequential([
        tf.keras.layers.Dense(16, activation='relu', input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(8, activation='relu'),
        tf.keras.layers.Dense(1, activation='sigmoid')
    ])

    # モデルのコンパイル
    model.compile(optimizer='adam',
                loss='binary_crossentropy',
                metrics=['accuracy'])

    # 5. 学習実行
    print("--- 学習開始 ---")
    model.fit(X_train, y_train, epochs=50, batch_size=4, verbose=1)

    # 評価
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"モデルの精度: {accuracy:.2f}")

    # 6. 保存 
    print("ファイルを保存しています...")
    
    # (1) モデル本体 (.keras)
    model.save('baseball_model.keras')
    
    # (2) スケーラー (数値の正規化用)
    joblib.dump(scaler, 'scaler.joblib')
    
    # (3) 列名のリスト (One-Hot Encoding再現用)
    model_columns = list(X_encoded.columns)
    joblib.dump(model_columns, 'model_columns.joblib')
    
    print("保存完了: baseball_model.keras, scaler.joblib, model_columns.joblib")

if __name__ == "__main__":
    main()