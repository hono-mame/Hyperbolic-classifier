import sqlite3
import pandas as pd
from sklearn.model_selection import train_test_split 
import argparse
import os # ディレクトリ操作のためにosモジュールを追加

# --- 1. コマンドライン引数の設定 ---
def parse_arguments():
    """
    コマンドライン引数を解析します。
    Usage: dataGenerationRandom.py <n_lines> <filter_nouns> <output_dir>
    """
    parser = argparse.ArgumentParser(
        description="WordNetJpnからハイパーニム・ハイポニム関係を抽出し、学習・評価用に分割して保存します。"
    )
    # n_lines (n_head): 抽出する行数 (整数)
    parser.add_argument(
        "n_lines",
        type=int,
        help="抽出するデータセットの行数。全行を使用する場合は0を指定します。",
    )
    # filter_nouns: 名詞フィルタリングの有無 (ブール値, 'True'/'False')
    parser.add_argument(
        "filter_nouns",
        type=lambda x: x.lower() == 'true', # 文字列'true'/'True'をTrueに変換
        help="名詞のみに絞るか (True/False)。",
    )
    # output_dir (base_path): 出力先のディレクトリパス
    parser.add_argument(
        "output_dir",
        type=str,
        help="データファイルを保存するベースディレクトリのパス。",
    )

    args = parser.parse_args()
    return args

# --- 2. メイン処理 ---
def main():
    args = parse_arguments()

    # コマンドライン引数を変数に設定
    n_head = args.n_lines
    filter_nouns = args.filter_nouns
    base_path = args.output_dir
    
    # パスがディレクトリであることを確認し、存在しない場合は作成
    if not os.path.isdir(base_path):
        os.makedirs(base_path, exist_ok=True)
        print(f"ディレクトリ: {base_path} を作成しました。")
        
    # データベースへの接続パス (ここは固定と仮定)
    # 必要に応じて、このパスも引数で受け取るように拡張可能です。
    db_path = "/Users/honokakobayashi/dev/Univ/Research/data/wnjpn.db" 
    try:
        conn = sqlite3.connect(db_path)
    except sqlite3.Error as e:
        print(f"データベース接続エラー: {e}")
        print(f"データベースファイルが見つかりません: {db_path}")
        return

    print(f"設定: n_head={n_head}, filter_nouns={filter_nouns}, output_dir={base_path}")

    # SQLクエリの定義
    query = """
    SELECT 
        w1.lemma AS hyper,
        w2.lemma AS hypo
    FROM synlink AS sl
    INNER JOIN synset AS sy1 ON sy1.synset = sl.synset1
    INNER JOIN synset AS sy2 ON sy2.synset = sl.synset2
    INNER JOIN sense AS se1 ON se1.synset = sy1.synset
    INNER JOIN sense AS se2 ON se2.synset = sy2.synset
    INNER JOIN word AS w1 ON w1.wordid = se1.wordid
    INNER JOIN word AS w2 ON w2.wordid = se2.wordid
    WHERE sl.link = 'hypo'
      AND se1.lang = 'jpn' AND se2.lang = 'jpn'
      AND w1.lang = 'jpn' AND w2.lang = 'jpn'
    """

    # 名詞フィルタリングの追加
    if filter_nouns:
        query += " AND sy1.pos = 'n' AND sy2.pos = 'n'"
        print("名詞のみに絞ります。")
    else:
        print("全品詞を使用します。")

    # データの抽出
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # ファイル名サフィックスの設定
    file_suffix = "_nouns" if filter_nouns else ""
    
    # n_headが指定されている場合はサフィックスを追加
    if n_head > 0 and n_head < len(df):
        df_process = df.head(n_head).copy()
        n_head_suffix = f"_head_{n_head}"
        print(f"データベースから {len(df)} 行抽出。先頭 {n_head} 行を使用します。")
    else:
        df_process = df.copy()
        n_head_suffix = ""
        print(f"データベースから {len(df)} 行抽出。全行を使用します。")

    # 全データの保存 (オプション: n_headで絞る前の全抽出データを保存)
    # output_file = os.path.join(base_path, f"hypernym_relations_jpn{file_suffix}_full.csv")
    # df.to_csv(output_file, index=False, encoding="utf-8")
    # print(f"全抽出データ: {len(df)} 行を {output_file} に保存しました。")

    # --- データの分割 ---
    
    # データが空でないことを確認
    if len(df_process) == 0:
        print("エラー: 処理対象のデータが空です。")
        return
    
    # train_test_splitを使ってランダムに80%を訓練、20%をテストに分割
    # 行数が1行の場合は分割エラーになるため、その場合は訓練データに全て割り当て
    if len(df_process) > 1:
        df_train, df_test = train_test_split(df_process, test_size=0.2, random_state=42)
        print(f"訓練/テストに80%/20% (random_state=42) で分割しました。")
    else:
        df_train = df_process.copy()
        df_test = pd.DataFrame() # テストデータは空にするか、最小限にする
        print("行数が1行のため、訓練データに全て割り当て、テストデータは空としました。")

    # --- データの保存 ---

    # 訓練データ (80%)
    train_file = os.path.join(base_path, f"train.csv")
    df_train.to_csv(train_file, index=False, encoding="utf-8")
    print(f"訓練データ: {len(df_train)} 行を {train_file} に保存しました。")

    # テストデータ (20%)
    test_file = os.path.join(base_path, f"eval.csv")
    df_test.to_csv(test_file, index=False, encoding="utf-8")
    print(f"テストデータ: {len(df_test)} 行を {test_file} に保存しました。")

# スクリプトとして実行されたときにmain関数を呼び出す
if __name__ == "__main__":
    main()