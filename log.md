# 変更履歴

## 2025-01-XX（最新）

### 学習パラメータの調整
- 方法3の学習設定を調整：
  - `EPOCHS`を50から**300**に増加
  - `EARLY_STOP_PATIENCE`を10から**30**に増加
  - より長い学習を可能にし、精度向上を期待
- 現在のモデル：**方法3（Method3_FiveClassSeg）**で学習中（5クラス領域セグメンテーション）
- 出力解像度：640×640

### データ拡張の実装（get_augmentation関数）
- **Horizontal Flip**（50%の確率）: 水平反転で左右対称性を考慮
- **Rotate ±7°**（30%の確率）: 小角度回転で回転不変性を向上
  - 画像は双3次補間（order=1）、マスクは最近傍補間（order=0）
- **弱いColorJitter**（20%の確率）: Brightness（0.9-1.1）とContrast（0.9-1.1）で色の変化に対応
- **軽いCutout**（10%の確率）: 10×10ピクセル領域を平均値で塗りつぶし、局所的な堅牢性を向上
- マスクも画像と同期して変換されるため、対応関係が保たれる
- **修正**: `np.fliplr()`後の`.copy()`を追加（負のストライド配列の問題を解消）

### U-Net型モデルへの置き換え
- **FPN版モデルを削除**し、U-Net版のみに統一
- **VGG16ベースのU-Netエンコーダ**を使用
- **U-Netデコーダ**で640×640解像度に段階的にアップサンプリング
- **方法1（UNetMethod1）**: 眼瞼セグ + 楕円回帰（640解像度で推論）
- **方法2（UNetMethod2）**: 縁セグ（640解像度で推論）
- **方法3（UNetMethod3）**: 5クラス領域セグ（640解像度で推論）
- エイリアスで既存のコードとの互換性を維持（`Method1_EyelidEllipse = UNetMethod1` など）

### 解像度の一貫性
- **512×512で統一**（入力・ラベル・モデル出力）
- ラベルを512×512で生成することで、学習と推論で一貫性を保証
- 画像サイズは512×512、モデルは512×512で推論

## 2025-01-XX（以前）

### train.ipynb実装完了
- ResNetBackboneクラス追加（マルチスケール特徴抽出）
- SimpleFPNのforward実装完了
- 3手法のモデル（Method1/2/3）のforward実装完了
- LossFunction3（Multi-class Dice Loss）の実装完了
- 学習ループ（train_epoch, validate）の実装完了
- 3手法のモデル初期化、Optimizer、Scaler設定
- 学習実行コード（方法3の簡易版）追加
- 推論と可視化機能を追加：
  - 学習済みモデルのロード
  - 検証データでの推論
  - 予測結果と正解の可視化（画像保存）
  - クラス別Dice係数の計算と表示
  - 検証セット全体でのDice係数評価
- 必要インポート（cv2, pandas）を追加
- modelディレクトリ自動作成
- pretrainedパラメータをweights='DEFAULT'に更新（非推奨警告の解消）
- エラーチェック完了：データローダー、モデル初期化、forward処理を確認
- NameError対策：
  - モデルロード処理を条件付きに変更（ファイル存在チェック）
  - 推論セクションに実行前チェックとガイドを追加
  - **学習セクションに依存関係チェックを追加（Cell 21）**：model3, train_loader, criterion3, optimizer3, scaler3, fold_idxが定義されているかチェックし、不足している場合は明確なエラーを表示
  - セル実行順序を明記するガイドを追加：
    - Cell 20（学習実行）: 「すべて実行」対応のガイド追加
    - Cell 21（学習実行コード）: 必要な変数が揃っていない場合、自動的にスキップ（エラーを出さない）
    - Cell 23（モデル初期化）: 「1番目に実行」注記
    - Cell 25（最適化設定）: 「2番目に実行」注記
    - 重複セル22を削除
  - **セルの順序を再構成**：「すべて実行」で止まらないように改善
    - Cell 21を空にして、学習実行をCell 29-30に移動
    - 新しい実行順序：Cell 25-26（モデル初期化）→ Cell 27-28（最適化設定）→ Cell 29-30（学習実行）
    - すべてのガイドセル（Cell 20, 25, 27）を更新
  - **画像サイズの統一**: RuntimeError（異なるサイズの画像）を修正
    - EyeSegmentationDatasetクラスに画像とマスクのリサイズ処理を追加
    - **画像サイズを640×640に統一**（実験計画とprocess_data.ipynbに合わせる）
    - IMAGE_HEIGHT = 640, IMAGE_WIDTH = 640を設定に追加
    - マスクはINTER_NEAREST、画像はINTER_LINEARでリサイズ
    - BATCH_SIZEを16→8に削減（640×640用のメモリ対策）
    - **確認**: process_data.ipynbで生成されたラベルは640×640であることを確認
  - **RuntimeError: 画像テンソルの形状修正**
    - 画像をHWC（Height, Width, Channels）からCHW（Channels, Height, Width）に変換する処理を追加
    - `.permute(2, 0, 1)`でRGB画像を正しい形式に変換
    - transform関数とdatasetクラスの両方に適用
  - **AttributeError: F.interpolate修正**
    - SimpleFPNクラス内で`F.interpolate`を`torch.nn.functional.interpolate`に変更
    - torchvision.transforms.functionalにはinterpolateが存在しないため修正
- 次: crossvalidation.ipynb（5-fold交差検証）

### ライブラリ追加
- pandas 2.3.3をインストール
- scikit-learn 1.7.2をインストール
- opencv-python 4.12.0.88をインストール
- scipy 1.16.2をインストール（scikit-learnの依存）
- joblib 1.5.2をインストール（scikit-learnの依存）
- numpyを2.3.4→2.2.6にダウングレード（依存関係の調整）
- requirements.txtを更新

### 学習ノートブックの実装開始
- train.ipynbの基本構造完成
- GPU確認、設定、fold_indices読み込み
- データセットクラス（EyeSegmentationDataset）の実装
- データローダー作成（fold別）
- 3手法のモデル骨組み：
  - 方法1: Method1_EyelidEllipse（眼瞼セグ + 楕円回帰）
  - 方法2: Method2_EdgeSeg（縁セグ）
  - 方法3: Method3_FiveClassSeg（5クラス領域セグ）
- 損失関数の骨組み（Dice係数、BCE+Dice）
- 学習ループの骨組み
- 次: 各手法の詳細実装と学習実行

### 膨張処理の削除
- ピクセル単位で厳密にアノテーションしているため、mask_lid_dil（膨張版）を削除
- vis/occの計算でmask_lidを直接使用
- experiment_plan.mdから膨張に関する記述を削除
- improvement.mdに記録

### vis/occの定義修正（重要！）
- **問題**: 眼瞼マスクの定義が逆だった
- **原因**: 眼瞼マスク = 「眼瞼縁に囲まれた部分（眼球の露出部分）」と理解していなかった
- **修正**: 
  - process_data.ipynbのコードを修正（vis/occを入れ替え）
  - experiment_plan.mdの定義を更新
  - improvement.mdに詳細を記録
- **教訓**: アノテーションの定義を最初に確認することが重要

### improvement.mdを作成
- エラー対処や工夫点を記録するファイルを作成
- experiment_plan.mdに参照を追加
- 実装の進捗状況を可視化（Phase別の完了状況）

### データ前処理実装完了
- `process_data.ipynb`の実装完了
- CVAT XMLパース機能を実装
- 患者ID抽出機能を実装
- ラスタライズ関数（polygon, ellipse）を実装
- 全データ処理とラベル生成機能を実装
- GroupKFold分割（患者IDベース）を実装
- ラベル画像保存機能を実装
- メタデータ保存機能を実装
- 保存場所の説明をmarkdownに追記

### 環境構築（第2回）
- PyTorchをGPU版（2.6.0+cu124）に更新
- train.ipynbにGPUチェックコードを追加（冒頭セル）
- requirements.txtをGPU版対応に更新
- GPU確認: NVIDIA GeForce RTX 3080 Ti Laptop GPU（CUDA 12.4）
- matplotlib 3.10.7を追加
- requirements.txtを更新（pip freeze実行）

### 環境構築（第1回）
- Python仮想環境（venv）の作成
- PyTorch 2.9.0（CPU版）のインストール
- Jupyter、NumPy、その他関連パッケージのインストール
- requirements.txt の作成
- README.md の作成
- log.md の作成（本ファイル）

### インストールパッケージ
- torch==2.6.0+cu124（GPU版）
- torchvision==0.21.0+cu124（GPU版）
- numpy==2.3.4
- jupyter==1.1.1
- その他関連パッケージ（詳細は requirements.txt を参照）

### 次のステップ
- [ ] データ前処理の実装（process_data.ipynb）
- [ ] 学習スクリプトの実装（train.ipynb）
- [ ] 交差検証スクリプトの実装（crossvalidation.ipynb）

