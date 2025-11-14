import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

def classify_pa_result(result):
    """
    'pa_result' 列の具体的なプレー結果（文字列）を受け取り、
    '安打', '凡打', None (除外) に分類する関数。
    """
    res_str = str(result)

    # --- カテゴリ1: 安打 (Hit) ---
    hit_keywords = ['安', '２', '３', '本', 'H', '2B', '3B', 'HR']
    if any(keyword in res_str for keyword in hit_keywords):
        return '安打'

    # --- カテゴリ3: その他（学習データから除外） ---
    if '犠飛' in res_str:
        return None
    if '犠打' in res_str:
        return None
    if any(keyword in res_str for keyword in ['四球', '死球']):
        return None
    if '失' in res_str:
        return None
    if any(keyword in res_str for keyword in ['妨', '振逃']):
        return None

    # --- カテゴリ2: 凡打 (Out) ---
    out_keywords = ['ゴロ', '飛', '直', '振', '併', '邪']
    if any(keyword in res_str for keyword in out_keywords):
        return '凡打'

    # --- 上記のいずれにも分類されなかった場合 ---
    return None

# --- メイン処理 ---
def main():
    # 1. データの読み込み
    try:
        # UTF-8 でまず試す
        df = pd.read_csv('3_イベントデータ_野球.csv', encoding='utf-8')
    except UnicodeDecodeError:
        # 失敗したら cp932 (Shift_JIS) で再試行
        print("UTF-8での読み込みに失敗。cp932で再試行します...")
        try:
            df = pd.read_csv('3_イベントデータ_野球.csv', encoding='cp932')
        except Exception as e:
            print(f"cp932でも読み込みに失敗しました: {e}")
            return
            

    # 2. 特徴量の準備 (使用する列を定義)
    features = ['pitch_zone', 'pitch_type_name', 'pitch_speed'] 
    
    # 3. 目的変数の作成 
    df['Target'] = df['pa_result'].apply(classify_pa_result)

    # 4. 学習データの絞り込み
    train_data = df.dropna(subset=['Target'])


    if len(train_data) == 0:
        print("エラー: 学習データが0件です。")
        print("pa_result列の分類ロジック（classify_pa_result関数）が、")
        print("実際のデータ（your_data.csv）の内容と一致しているか確認してください。")

        print("\n--- pa_result列のユニークな値 (上位50件) ---")
        print(df['pa_result'].unique()[:50])
        print("-----------------------------------------")
        return

    print(f"読み込み成功。学習データ {len(train_data)} 件を抽出しました。")
    print(train_data['Target'].value_counts()) 

    # --- 5. 特徴量エンジニアリング (★ここから追加・修正★) ---
    # モデルが計算できるように、テキストや欠損値を処理する

    # 5a. 特徴量 X と 目的変数 y を取り出す
    # .copy() をつけて、後で値を変更しても警告が出ないようにする
    X = train_data[features].copy()
    y = train_data['Target'] 

    # 5b. 欠損値(NaN)の処理
    # 'pitch_speed' (数値) が欠損していたら、「中央値」で埋める
    X['pitch_speed'] = X['pitch_speed'].fillna(X['pitch_speed'].median())
    
    # 'pitch_type_name' (テキスト) が欠損していたら、'Unknown' という文字列で埋める
    X['pitch_type_name'] = X['pitch_type_name'].fillna('Unknown')
    
    # 'pitch_zone' (テキスト/カテゴリ) が欠損していたら、'Unknown' で埋める
    # (pitch_zoneが数値の場合も考慮し、一度文字列に変換してから埋める)
    X['pitch_zone'] = X['pitch_zone'].astype(str).fillna('Unknown')

    # 5c. テキストデータを数値に変換 (One-Hot Encoding)
    # pd.get_dummies() が X の中のテキスト列を自動で数値列に変換する
    X_encoded = pd.get_dummies(X, drop_first=True)
    # --- (★追加・修正はここまで★) ---


    # 6. 訓練データとテストデータに分割
    # ★修正: X ではなく、エンコード済みの X_encoded を使う
    X_train, X_test, y_train, y_test = train_test_split(X_encoded, y, test_size=0.2, random_state=42)

    # 7. モデルの初期化と学習
    model = RandomForestClassifier(random_state=42)
    model.fit(X_train, y_train) # ← これで ValueError が解消するはず
    
    # 8. 予測と評価
    y_pred = model.predict(X_test)
    print(f"モデルの精度: {accuracy_score(y_test, y_pred)}")


if __name__ == "__main__":
    main()