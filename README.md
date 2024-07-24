# MLSA Neural Vocoder

[この記事](https://dmw.nico/ja/articles/intern_ashida)のMLSAニューラルボコーダーの学習コードです。

## 使い方

Python 3.10以上が必要です。
あらかじめ環境にあったPyTorch 2を導入してください。

```bash
pip install -r requirements.txt
```

### 設定ファイル

`config`ディレクトリにサンプルの設定ファイルがあります。
適宜`data_path`や`preprocessed_path`、`log_dir`などのパラメータを変更することで前処理・学習に使用できます。

- `data_path`: 学習に使用したいwavファイルが入ったディレクトリを指定してください。
- `preprocessed_path`: 前処理データを格納するディレクトリを指定してください。
- `log_dir`: 学習ログ(tensorboardのデータ)とチェックポイントを保存するディレクトリを指定してください。

### 前処理

```bash
python preprocessor.py <config file>
```

長い音声(歌声データなど)を使用する場合は`-s`もしくは`--split`オプションを使ってください。
```bash
python preprocessor.py <config file> -s
```

### 学習

```bash
python train.py <config file>
```

サンプル音声がTensorboard上に出力されます。

## ライセンス

[MIT ライセンス](./LICENSE)です。
