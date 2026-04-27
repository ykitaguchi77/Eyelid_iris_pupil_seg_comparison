# Eyelid_Iris_pupil_seg_comparison

眼部画像セグメンテーションの3つの手法（Method1/2/3）を比較するプロジェクトです。
CVAT XMLアノテーションから学習用ラベルを生成し、U-Netベースの3つの異なるアプローチでセグメンテーションを実行します。

## 📋 目次

- [プロジェクト概要](#プロジェクト概要)
- [セットアップ](#セットアップ)
- [使い方](#使い方)
- [データ前処理（process_data.ipynb）](#データ前処理process_dataipynb)
- [モデル学習（train.ipynb）](#モデル学習trainipynb)
- [5-Fold Cross-Validation（crossvalidation.ipynb）](#5-fold-cross-validationcrossvalidationipynb)
- [Ablation Study（ablation_study.ipynb）](#ablation-studyablation_studyipynb)
- [ディレクトリ構造](#ディレクトリ構造)
- [評価指標](#評価指標)

---

## 🎯 プロジェクト概要

このプロジェクトは、眼部画像における**眼瞼（Eyelid）**、**虹彩（Iris）**、**瞳孔（Pupil）**のセグメンテーションを行います。

### 3つのアプローチ

| 手法 | 説明 | アーキテクチャ |
|------|------|----------------|
| **Method1** | 眼瞼セグメンテーション + 虹彩・瞳孔の楕円パラメータ回帰 | U-Net + 回帰ヘッド |
| **Method2** | エッジセグメンテーション（3チャネル: 眼瞼縁、虹彩縁、瞳孔縁） | U-Net + エッジ検出 |
| **Method3** | 6クラス領域セグメンテーション（背景、結膜、可視虹彩、遮蔽虹彩、可視瞳孔、遮蔽瞳孔） | U-Net + マルチクラス分類 |

### 6クラス定義

```
0: background   = lid外 ∩ iris外 ∩ pupil外
1: conj         = lid内 ∩ iris外 ∩ pupil外（結膜露出部）
2: iris_vis     = lid内 ∩ iris内 ∩ pupil外（可視虹彩）
3: iris_occ     = lid外 ∩ iris内 ∩ pupil外（遮蔽虹彩）
4: pupil_vis    = lid内 ∩ iris内 ∩ pupil内（可視瞳孔）
5: pupil_occ    = lid外 ∩ iris内 ∩ pupil内（遮蔽瞳孔）
```

---

## ⚙️ セットアップ

### 必要な環境

- Python 3.8+
- CUDA対応GPU（推奨: RTX 3080以上）
- PyTorch 2.0+（CUDA版）

### インストール

```bash
# 仮想環境の作成
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# 必要なライブラリのインストール
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install opencv-python numpy pandas scikit-learn scikit-image matplotlib pillow tqdm
pip install transformers  # SegFormer使用時（ablation_study.ipynb用）
```

---

## 🚀 使い方

### ステップ1: データ前処理

まず、`process_data.ipynb`を実行してアノテーションからラベルを生成します。

```bash
# Jupyter Notebookを起動
jupyter notebook process_data.ipynb
```

**重要**: このノートブックは**上から順番に実行**してください。「Run All」を推奨します。

### ステップ2: モデル学習

次に、`train.ipynb`で3つの手法を学習します。

```bash
jupyter notebook train.ipynb
```

学習したい手法を選択するには、**カラム7**のセルで以下のフラグを設定します：

```python
# 学習フラグ（必要なものだけTrueでOK）
TRAIN_METHOD1 = True   # 眼瞼セグメンテーション + 楕円パラメータ回帰
TRAIN_METHOD2 = True   # エッジセグメンテーション
TRAIN_METHOD3 = True   # 6クラス領域セグメンテーション
```

---

## 📊 データ前処理（process_data.ipynb）

### 処理の流れ

1. **CVAT XMLファイルの読み込み**
   - `Images/eyelid_caruncle_seg_0-2000.xml`（眼瞼・涙丘）
   - `Images/obb_iris_pupil_1-3000.xml`（虹彩・瞳孔）

2. **患者IDの抽出**
   - ファイル名から患者IDを抽出（例: `1-20141126-38-091804_...jpg` → 患者ID=1）

3. **画像・ラベルの処理**
   - 512×512へのリサイズ
   - ラベル生成（眼瞼マスク、虹彩マスク、瞳孔マスク、6クラスラベル）

4. **楕円の回転対応**
   - CVATのellipse回転属性（`rotation`）に対応
   - 傾いた虹彩・瞳孔も正確にラスタライズ

5. **GroupKFold分割**
   - 患者IDベースの5-fold分割
   - 同一患者の画像がTrain/Valに跨らないように分割

### 生成されるファイル

#### ラベル画像

**`Images/labels_seg/`**
- `*_mask_lid.png` - 眼瞼マスク（結膜露出部）
- `*_iris_vis.png` - 可視虹彩
- `*_iris_occ.png` - 遮蔽虹彩
- `*_pupil_vis.png` - 可視瞳孔
- `*_pupil_occ.png` - 遮蔽瞳孔
- `*_sixcls.png` - **6クラス統合ラベル（カラー画像）**

**`Images/labels_obb/`**
- `*_mask_iris.png` - 虹彩マスク（完全楕円、回転対応✅）
- `*_mask_pupil.png` - 瞳孔マスク（完全楕円、回転対応✅）

#### メタデータ（プロジェクトルート）

- `fold_indices.json` - 5-fold GroupKFold分割情報
- `image_metadata.csv` - 画像メタデータ（image_id, filename, patient_id, original_size）
- `patient_list.json` - 患者IDリスト

### 実行時の注意点

⚠️ **必ず「Run All」または上から順番に実行してください**

途中のセルで既存ラベルの削除処理があります。個別実行する場合は依存関係に注意してください。

---

## 🤖 モデル学習（train.ipynb）

### ノートブック構成

train.ipynbは**カラム（セクション）**で構成されています：

| カラム | 内容 |
|--------|------|
| **カラム1** | 環境確認・基本設定（GPU確認、パス設定、ハイパーパラメータ） |
| **カラム2** | データセット & データローダ（sixcls.pngの読み込み、データ拡張） |
| **カラム3** | U-Net（Method1/2/3）定義（VGG16エンコーダ + デコーダ） |
| **カラム4** | 損失関数・レンダリング関数（Dice Loss、楕円レンダリング） |
| **カラム5** | Method2用ラベル生成・評価補助（エッジ抽出、楕円フィッティング） |
| **カラム6** | Method2の教師エッジ生成（オンザフライ） |
| **カラム7** | 学習ループ（Optimizer、Early Stopping） |
| **カラム8** | 学習実行（学習ループの実行） |
| **カラム9** | 評価ロジック（Eyelid/Iris/Pupil Dice計算） |
| **カラム10** | 推論・可視化（GT/Method1/2/3の並列比較） |
| **カラム11** | 全クラス可視化（Method3の6クラス予測） |

### ハイパーパラメータ

```python
IMAGE_HEIGHT = 512
IMAGE_WIDTH  = 512
BATCH_SIZE   = 16
NUM_EPOCHS   = 50
LEARNING_RATE = 1e-3
WEIGHT_DECAY  = 1e-4
EARLY_STOP_PATIENCE = 30
```

### Method1: 眼瞼セグメンテーション + 楕円パラメータ回帰

**アーキテクチャ:**
```
入力画像 (3, 512, 512)
    ↓
VGG16エンコーダ
    ↓
U-Netデコーダ (64ch)
    ├─→ 眼瞼セグメンテーションヘッド (1ch logits)
    ├─→ 虹彩楕円回帰ヘッド (5パラメータ: cx,cy,a,b,θ)
    └─→ 瞳孔楕円回帰ヘッド (5パラメータ)
```

**損失関数:**
- 眼瞼: BCEWithLogitsLoss + Dice Loss
- 虹彩/瞳孔: 楕円パラメータ → レンダリング → BCEWithLogitsLoss

### Method2: エッジセグメンテーション

**アーキテクチャ:**
```
入力画像 (3, 512, 512)
    ↓
VGG16エンコーダ
    ↓
U-Netデコーダ (64ch)
    ↓
エッジセグメンテーションヘッド (3ch logits)
    ├─→ ch0: 眼瞼エッジ
    ├─→ ch1: 虹彩エッジ（眼瞼内のみ）
    └─→ ch2: 瞳孔エッジ（眼瞼内のみ）
```

**損失関数:**
- EdgeBCELossWithNHWC（pos_weight=3.0でエッジピクセルに重み）

**特徴:**
- 教師ラベルはオンザフライで生成（thickness=3の太いエッジ）
- エッジから領域への変換: モルフォロジカルクロージング（25×25カーネル、6iterations）
- 楕円フィッティング: cv2.fitEllipse（エッジが綺麗なので高速・高精度）

### Method3: 6クラス領域セグメンテーション

**アーキテクチャ:**
```
入力画像 (3, 512, 512)
    ↓
VGG16エンコーダ
    ↓
U-Netデコーダ (64ch)
    ↓
6クラスセグメンテーションヘッド (6ch logits)
```

**損失関数:**
- Multi-class Dice Loss（各クラスのDiceを平均）

**後処理（評価時）:**
1. クラス2,3（虹彩領域）からマスク合成
2. エッジ抽出（thickness=3）
3. RANSAC + 最小二乗法で楕円フィッティング（部分エッジにも対応）
4. 楕円マスク化してDice計算

### 学習の実行

**カラム7**で学習フラグを設定してからセルを実行：

```python
TRAIN_METHOD1 = True   # Method1を学習
TRAIN_METHOD2 = True   # Method2を学習
TRAIN_METHOD3 = True   # Method3を学習
```

学習済みモデルは`model/`ディレクトリに保存されます：
- `method1_fold0_best.pth`
- `method2_fold0_best.pth`
- `method3_fold0_best.pth`

### 評価の実行

**カラム9**で評価フラグを設定：

```python
LOAD_PRETRAINED = True  # 学習済みモデルをロード
EVALUATE_METHOD1 = True
EVALUATE_METHOD2 = True
EVALUATE_METHOD3 = True
```

セルを実行すると、Dice係数（Eyelid/Iris/Pupil）が計算されます。

**評価指標の定義（統一）:**

| 対象 | GroundTruth | Method1 | Method2 | Method3 |
|------|-------------|---------|---------|---------|
| **Eyelid** | mask_lid | sigmoid(logits) | エッジ→塗りつぶし | クラス1∪2∪4 |
| **Iris** | mask_iris | 楕円パラメータ | エッジ→楕円 | クラス2∪3→エッジ→RANSAC |
| **Pupil** | mask_pupil | 楕円パラメータ | エッジ→楕円 | クラス4∪5→エッジ→RANSAC |

### 可視化

**カラム10**: GT/Method1/2/3の並列比較（3行×5列レイアウト）

```python
# ランダムに3サンプルを可視化
visualize_compare(val_ds[i], device)
```

**カラム11**: Method3の6クラス予測可視化

```python
# 4枚表示: Original | GroundTruth | Prediction | Ellipse Fitting
visualize_method3_all_classes(sample, device, show_stats=True)
```

---

## 🔬 5-Fold Cross-Validation（crossvalidation.ipynb）

### 概要

3つの手法すべてを5-fold cross-validationで公平に比較し、統計的に有意な性能評価を行います。

**特徴:**
- ✅ **3手法すべてを自動学習・評価**（Method1/2/3）
- 🔄 **Resume機能**: 中断しても続きから再開可能
- ⚡ **高速化機能**: 楕円キャッシュ、並列ロード、sixcls直接読込
- 📊 **詳細な比較レポート**: CSV形式で保存

### 設定

```python
NUM_EPOCHS = 300           # 300エポック（train.ipynbは50エポック）
EARLY_STOP_PATIENCE = 30   # 30エポックでearly stopping
BATCH_SIZE = 16
NUM_FOLDS = 5              # 5-fold cross-validation
NUM_WORKERS = 4            # 並列データロード（20-30%高速化）
```

### 実行方法

#### 初回実行

```bash
jupyter notebook crossvalidation.ipynb
```

1. **セル1-3**: 環境設定
2. **セル4**: 楕円キャッシュ生成（Method1高速化用、1-2分）
3. **セル5以降**: Run All

#### 中断後の再開

- そのまま **Run All** を再実行
- 完了済みタスクは自動スキップ
- 進捗は `cache/cv_progress.json` に保存

#### 最初からやり直す

1. **セル8**で `RESET_PROGRESS = True` に変更
2. セル8を実行（進捗リセット）
3. Run All

### 高速化機能

| 機能 | 対象 | 効果 | 設定 |
|------|------|------|------|
| **楕円キャッシュ** | Method1 | 25%高速化 | セル4実行 |
| **並列ロード** | 全メソッド | 20-30%高速化 | 自動適用 |
| **sixcls直接読込** | Method3 | 15-20%高速化 | 自動適用 |

**総合効果:**
- Method1: 40%高速化（500分 → 300分）
- Method2: 25-30%高速化（500分 → 350-375分）
- Method3: 30-35%高速化（500分 → 325-337分）

### 出力ファイル

**モデル:**
```
model/cv_300ep/
  ├── method1_fold{0-4}_best.pth  # Method1 × 5 folds
  ├── method2_fold{0-4}_best.pth  # Method2 × 5 folds
  └── method3_fold{0-4}_best.pth  # Method3 × 5 folds
```

**評価結果:**
```
results/
  ├── cv_train_method{1,2,3}_*.csv     # 学習結果（fold別）
  ├── cv_eval_method{1,2,3}_*.csv      # 評価結果（fold別）
  └── cv_comparison_*.csv              # 3手法比較
```

**進捗管理:**
```
cache/
  ├── cv_progress.json                 # 進捗状態（Resume用）
  └── ellipse_params/
      └── ellipse_params.npz          # 楕円パラメータキャッシュ
```

### Resume機能の動作例

**初回実行:**
```
🆕 新規実行: 5-Fold Cross-Validation 開始
学習メソッド: [1, 2, 3]
総タスク数: 15 (Folds × Methods)

Fold 0 / 4
  --- Method 1 学習開始 ---
  [100 epochs 完了] ← ここで中断
```

**再開時:**
```
🔄 Resume: 前回の続きから実行
   完了済み: 7 / 15 タスク

Fold 0 / 4
  ✅ Method 1 - Fold 0: 完了済みスキップ
  ✅ Method 2 - Fold 0: 完了済みスキップ
  --- Method 3 学習開始 ---  ← ここから再開！
```

### 評価レポート例

```
【3手法比較】
   Method  Mean Dice    Std
  Method1     0.9389  0.0123
  Method2     0.9435  0.0098
  Method3     0.9647  0.0087  ← Best!
```

---

## 🔬 Ablation Study（ablation_study.ipynb）

### 概要

Method3（U-Net）と**SegFormer**（Vision Transformerベースのセマンティックセグメンテーションモデル）を比較するためのAblation Studyです。

**目的:**
- Method3（U-Net + VGG16）の性能とSegFormerの性能を比較
- 最新のTransformerベースのセグメンテーションモデルの適用可能性を検証

### SegFormerについて

- **開発元**: NVIDIA Research
- **アーキテクチャ**: Vision Transformer (ViT) ベース
- **特徴**: 軽量かつ高精度なセマンティックセグメンテーション
- **モデルサイズ**: SegFormer-B2（27.4M params、推奨サイズ）
- **参考**: https://github.com/NVlabs/SegFormer

### 設定

```python
IMAGE_HEIGHT = 512
IMAGE_WIDTH  = 512
BATCH_SIZE   = 8   # SegFormerはメモリ使用量が多いため少し小さく
NUM_EPOCHS   = 300
LEARNING_RATE = 6e-5  # SegFormer推奨学習率（Transformerは低め）
WEIGHT_DECAY  = 1e-4
EARLY_STOP_PATIENCE = 30
NUM_FOLDS = 5
NUM_CLASSES = 6  # 6クラス分類（Method3と同じ）
NUM_WORKERS = 0  # Windows対応
```

**モデル選択:**
- デフォルト: `nvidia/segformer-b2-finetuned-ade-512-512`（バランス型、推奨）
- その他の選択肢:
  - `nvidia/segformer-b0-finetuned-ade-512-512`（軽量版、3.7M params）
  - `nvidia/segformer-b5-finetuned-ade-512-512`（最高精度、84.7M params）

### 実行方法

#### 初回実行

```bash
jupyter notebook ablation_study.ipynb
```

1. **セル1-2**: GPUチェック・基本設定
2. **セル3**: データセット定義（SegFormer用ImageProcessorを使用）
3. **セル4**: SegFormerモデル定義（6クラス用にカスタマイズ）
4. **セル5**: 損失関数定義（Cross Entropy + Dice Loss）
5. **セル6**: 学習ループ定義
6. **セル7**: 評価関数定義（Method3と同じ評価ロジック）
7. **セル8**: 進捗確認（オプション）
8. **セル9**: 5-Fold CV実行（Resume機能対応）
9. **セル10以降**: 結果集計・Method3との比較・可視化

#### Resume機能

`crossvalidation.ipynb`と同様に、中断しても続きから再開可能です：

- 進捗は `cache/ablation_progress.json` に保存
- 完了済みFoldは自動スキップ
- 中断されたFoldは最初から再学習

#### 進捗リセット

1. **セル8**で `RESET_PROGRESS = True` に変更
2. セル8を実行（進捗リセット）
3. Run All

### 特徴

- ✅ **Resume機能**: 中断しても続きから再開可能
- 📊 **Method3との自動比較**: 結果集計時にMethod3の最新結果と自動比較
- 🎨 **可視化機能**: Method3とSegFormerの予測を並列比較
- 📈 **詳細なログ**: JSON形式で実験条件・結果を保存

### 出力ファイル

**モデル:**
```
model/ablation_study/
  └── segformer_fold{0-4}_best.pth  # SegFormer × 5 folds
```

**評価結果:**
```
results/
  ├── ablation_segformer_train_*.csv      # 学習結果（fold別）
  ├── ablation_segformer_eval_*.csv       # 評価結果（fold別）
  ├── ablation_segformer_summary_*.csv    # 5-Fold CVサマリー
  └── ablation_experiment_log_*.json      # 実験ログ（条件・結果）
```

**進捗管理:**
```
cache/
  └── ablation_progress.json              # 進捗状態（Resume用）
```

### Method3との比較例

```
【SegFormer vs Method3比較】
           Method  Eyelid    Iris   Pupil   Mean
        SegFormer  0.9863  0.8692  0.9095  0.9217
         Method3   0.9854  0.9705  0.9576  0.9712  ← Best!

結論: Method3の方が高精度（特にIris/Pupil）
```

### 実行時間の目安

- **1 Fold**: 約4-6時間（GPU: RTX 3080 Ti、300 epochs、early stopping有効時）
- **5-Fold CV**: 約20-30時間（全Fold完了まで）

### 必要なライブラリ

SegFormerを使用するため、以下の追加ライブラリが必要です：

```bash
pip install transformers
```

---

## 📁 ディレクトリ構造

```
Eyelid_Iris_pupil_seg_comparison/
├── Images/
│   ├── images/                          # 元画像（*.jpg）
│   ├── labels_seg/                      # 眼瞼系ラベル
│   │   ├── *_mask_lid.png              # 眼瞼マスク
│   │   ├── *_iris_vis.png              # 可視虹彩
│   │   ├── *_iris_occ.png              # 遮蔽虹彩
│   │   ├── *_pupil_vis.png             # 可視瞳孔
│   │   ├── *_pupil_occ.png             # 遮蔽瞳孔
│   │   └── *_sixcls.png                # 6クラス統合ラベル（カラー）
│   ├── labels_obb/                      # 虹彩・瞳孔ラベル
│   │   ├── *_mask_iris.png             # 虹彩マスク（完全楕円）
│   │   └── *_mask_pupil.png            # 瞳孔マスク（完全楕円）
│   ├── eyelid_caruncle_seg_0-2000.xml  # CVAT XML（眼瞼・涙丘）
│   └── obb_iris_pupil_1-3000.xml       # CVAT XML（虹彩・瞳孔）
├── model/                               # 学習済みモデル
│   ├── method1_fold0_best.pth          # train.ipynb用（50 epochs）
│   ├── method2_fold0_best.pth
│   ├── method3_fold0_best.pth
│   ├── cv_300ep/                       # crossvalidation.ipynb用（300 epochs）
│   │   ├── method1_fold{0-4}_best.pth
│   │   ├── method2_fold{0-4}_best.pth
│   │   └── method3_fold{0-4}_best.pth
│   └── ablation_study/                 # ablation_study.ipynb用（300 epochs）
│       └── segformer_fold{0-4}_best.pth
├── cache/                               # キャッシュ・進捗管理
│   ├── cv_progress.json                # CV進捗（Resume用）
│   ├── ablation_progress.json          # Ablation Study進捗（Resume用）
│   └── ellipse_params/
│       └── ellipse_params.npz          # 楕円パラメータキャッシュ
├── results/                             # 評価結果（CSV）
│   ├── cv_train_method{1,2,3}_*.csv
│   ├── cv_eval_method{1,2,3}_*.csv
│   ├── cv_comparison_*.csv
│   ├── ablation_segformer_train_*.csv
│   ├── ablation_segformer_eval_*.csv
│   ├── ablation_segformer_summary_*.csv
│   └── ablation_experiment_log_*.json
├── process_data.ipynb                   # データ前処理スクリプト
├── train.ipynb                          # 学習・評価スクリプト（50 epochs）
├── crossvalidation.ipynb                # 5-Fold CV（300 epochs, Resume対応）
├── ablation_study.ipynb                 # Ablation Study: SegFormer vs Method3
├── fold_indices.json                    # 5-fold分割情報
├── image_metadata.csv                   # 画像メタデータ
├── patient_list.json                    # 患者IDリスト
└── README.md                            # このファイル
```

---

## 📈 評価指標

### Dice係数（Sørensen–Dice coefficient）

セグメンテーション精度の評価には**Dice係数**を使用します：

```
Dice = (2 × |Prediction ∩ GroundTruth|) / (|Prediction| + |GroundTruth|)
```

- 範囲: 0（一致なし）～ 1（完全一致）
- 各手法について、Eyelid/Iris/Pupilの3つのDice係数を計算

### 評価結果の例

```
=== Validation Dice (mean over samples) ===
       Method1 Method2 Method3
Eyelid  0.9893  0.9638  0.9854
Iris    0.9245  0.9472  0.9705
Pupil   0.8063  0.9241  0.9576
```

### 解釈

- **Eyelid**: Method1が最も高精度（直接セグメンテーション）
- **Iris**: Method3が最も高精度（領域ベース + RANSAC楕円フィッティング）
- **Pupil**: Method3が最も高精度（小領域でも安定）

---

## 🔧 トラブルシューティング

### GPUメモリ不足

バッチサイズを減らしてください：

```python
BATCH_SIZE = 8  # デフォルト: 16
```

### CUDA out of memory

カラム26のメモリクリアセルを実行：

```python
torch.cuda.empty_cache()
gc.collect()
```

### モデルが見つからない

`LOAD_PRETRAINED = True`の場合、`model/`ディレクトリに`.pth`ファイルが必要です。
学習していない場合は、先に学習フラグ（`TRAIN_METHOD*`）をTrueにしてカラム8を実行してください。

### XMLファイルが見つからない

`Images/`ディレクトリに以下のファイルがあることを確認：
- `eyelid_caruncle_seg_0-2000.xml`
- `obb_iris_pupil_1-3000.xml`

---

## 📝 ライセンス

このプロジェクトは研究目的で使用されます。

## 🤝 貢献

バグ報告や機能リクエストは、GitHubのIssueでお願いします。

---

## 🎓 推奨ワークフロー

### 開発・実験時
```bash
# 1. データ前処理（初回のみ）
jupyter notebook process_data.ipynb  # Run All

# 2. 単一Foldで動作確認（50 epochs）
jupyter notebook train.ipynb  # カラム7でフラグ設定 → Run

# 3. 本格的な5-Fold CV（300 epochs）
jupyter notebook crossvalidation.ipynb  # Run All
```

### 本番評価時
```bash
# crossvalidation.ipynb のみ実行
jupyter notebook crossvalidation.ipynb  # Run All
# → 3手法×5-fold = 15モデルを学習・評価
# → 統計的に有意な性能比較
```

### Ablation Study実行時
```bash
# SegFormerとMethod3を比較
jupyter notebook ablation_study.ipynb  # Run All
# → SegFormer × 5-fold = 5モデルを学習・評価
# → Method3との性能比較
```

---

**最終更新:** 2025年11月3日
