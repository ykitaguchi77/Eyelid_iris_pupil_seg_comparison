# 実験ノート：眼部画像セグメンテーション 3手法比較

## 📋 実験概要

### 目的
眼部画像における**眼瞼（Eyelid）**、**虹彩（Iris）**、**瞳孔（Pupil）**のセグメンテーションを、3つの異なるアプローチで実装し、5-fold cross-validationにより公平に比較評価する。

### 実験期間
- **開始**: 2025年11月
- **最新結果**: 2026年01月01日（SegFormer/YOLO ablation 追記）

### データセット
- **総画像数**: 1,992枚
- **被験者（患者）数**: 122名（ファイル名先頭IDでクラスタリング、GroupKFoldで患者単位分割）
- **解像度**: 512×512ピクセル
- **アノテーション形式**: CVAT XML

---

## 🔬 3つの手法（Method1/2/3）

### Method1: Eyelid Segmentation + Ellipse Regression

**アプローチ:**
- 眼瞼はセグメンテーション（バイナリマスク）
- 虹彩・瞳孔は楕円パラメータ回帰（5パラメータ: cx, cy, a, b, θ）

**アーキテクチャ:**
```
入力画像 (3, 512, 512)
    ↓
VGG16-BNエンコーダ
    ↓
U-Netデコーダ (64ch)
    ├─→ 眼瞼セグメンテーションヘッド (1ch logits)
    ├─→ 虹彩楕円回帰ヘッド (5パラメータ)
    └─→ 瞳孔楕円回帰ヘッド (5パラメータ)
```

**損失関数:**
- 眼瞼: BCEWithLogitsLoss + Dice Loss
- 虹彩/瞳孔: 楕円パラメータ → レンダリング → BCEWithLogitsLoss

**特徴:**
- ✅ 眼瞼の精度が非常に高い（Dice: 0.9840）
- ⚠️ 瞳孔の精度が低い（Dice: 0.7181）かつ不安定
- ⚠️ 楕円近似による誤差が課題

**コード位置:**
- モデル定義: `crossvalidation.ipynb` セル3（`UNetMethod1`）
- 損失関数: `crossvalidation.ipynb` セル4（`LossFunction1`）
- 楕円レンダリング: `crossvalidation.ipynb` セル4（`render_ellipse_logits`）

---

### Method2: Edge Segmentation

**アプローチ:**
- 3チャネルのエッジセグメンテーション
  - ch0: 眼瞼エッジ
  - ch1: 虹彩エッジ（眼瞼内のみ）
  - ch2: 瞳孔エッジ（眼瞼内のみ）

**アーキテクチャ:**
```
入力画像 (3, 512, 512)
    ↓
VGG16-BNエンコーダ
    ↓
U-Netデコーダ (64ch)
    ↓
エッジセグメンテーションヘッド (3ch logits)
```

**損失関数:**
- EdgeBCELossWithNHWC（pos_weight=3.0でエッジピクセルに重み）

**後処理（推論時）:**
1. **Eyelid**: エッジ → モルフォロジカルクロージング（25×25カーネル、6回）→ 塗りつぶし
2. **Iris/Pupil**: エッジ → `cv2.fitEllipse()` → 楕円マスク化

**特徴:**
- ✅ バランスが良い（全クラスで0.85以上の精度）
- ✅ 瞳孔の精度がMethod1より大幅に向上（0.8901 vs 0.7181）
- ✅ エッジが綺麗なので楕円フィッティングが高速・高精度

**コード位置:**
- モデル定義: `crossvalidation.ipynb` セル3（`UNetMethod2`）
- 損失関数: `crossvalidation.ipynb` セル4（`LossFunction2`）
- エッジ生成: `crossvalidation.ipynb` セル4（`mask_to_edge`, `build_method2_targets`）

---

### Method3: 6-Class Region Segmentation

**アプローチ:**
- 6クラスの領域セグメンテーション
- 各ピクセルを6クラスのいずれかに分類

**6クラス定義:**
```
0: background   = lid外 ∩ iris外 ∩ pupil外
1: conj         = lid内 ∩ iris外 ∩ pupil外（結膜露出部）
2: iris_vis     = lid内 ∩ iris内 ∩ pupil外（可視虹彩）
3: iris_occ     = lid外 ∩ iris内 ∩ pupil外（遮蔽虹彩）
4: pupil_vis    = lid内 ∩ iris内 ∩ pupil内（可視瞳孔）
5: pupil_occ    = lid外 ∩ iris内 ∩ pupil内（遮蔽瞳孔）
```

**アーキテクチャ:**
```
入力画像 (3, 512, 512)
    ↓
VGG16-BNエンコーダ
    ↓
U-Netデコーダ (64ch)
    ↓
6クラスセグメンテーションヘッド (6ch logits)
```

**損失関数:**
- Multi-class Dice Loss（各クラスのDiceを平均）

**後処理（推論時）:**
1. クラス2,3（虹彩領域）からマスク合成
2. エッジ抽出（thickness=3）
3. RANSAC + 最小二乗法で楕円フィッティング（部分エッジにも対応）
4. 楕円マスク化してDice計算

**特徴:**
- 🏆 **最高性能**: Mean Dice = 0.9424
- ✅ **最も安定**: 標準偏差が最小（0.0051）
- ✅ **全クラスで高精度**: Eyelid (0.9807), Iris (0.9424), Pupil (0.9042)
- ✅ **Ellipse補正の効果**: Irisで+0.0649の向上

**コード位置:**
- モデル定義: `crossvalidation.ipynb` セル3（`UNetMethod3`）
- 損失関数: `crossvalidation.ipynb` セル4（`LossFunction3`）
- 6クラスラベル生成: `crossvalidation.ipynb` セル2（`build_sixclass_target`）

---

## 📊 データ前処理

### 処理の流れ

**ファイル**: `process_data.ipynb`

1. **CVAT XMLファイルの読み込みとパース**
   - `eyelid_caruncle_seg_0-2000.xml`（眼瞼・涙丘）
   - `obb_iris_pupil_1-3000.xml`（虹彩・瞳孔）

2. **患者IDの抽出**
   - ファイル名から患者IDを抽出
   - GroupKFoldで患者単位分割（データリーク防止）

3. **画像・ラベルの処理**
   - 512×512へのリサイズ
   - 画像: 双3次補間
   - マスク: 最近傍補間

4. **ラベル生成**
   - `mask_lid.png`: 眼瞼マスク（バイナリ）
   - `mask_iris.png`: 虹彩マスク（完全楕円）
   - `mask_pupil.png`: 瞳孔マスク（完全楕円）
   - `sixcls.png`: 6クラス統合ラベル（カラー画像）

5. **GroupKFold分割**
   - 5-foldで患者単位分割
   - `fold_indices.json`に保存

**出力先:**
```
Images/
├── images/                    # 元画像（*.jpg）
├── labels_seg/               # 眼瞼系ラベル
│   ├── *_mask_lid.png
│   └── *_sixcls.png
└── labels_obb/               # 虹彩・瞳孔ラベル
    ├── *_mask_iris.png
    └── *_mask_pupil.png
```

**メタデータ:**
- `fold_indices.json`: 5-fold GroupKFold分割情報
- `image_metadata.csv`: 画像メタデータ（image_id, filename, patient_id, original_size）
- `patient_list.json`: 患者IDリスト

---

## 🔄 5-Fold Cross-Validation実装

### 実装ファイル

**`crossvalidation.ipynb`**: 完全独立実装（train.ipynbからの引用なし）

### 実験設定

```python
NUM_EPOCHS = 300              # 300エポック（train.ipynbは50エポック）
EARLY_STOP_PATIENCE = 30      # 30エポックでearly stopping
BATCH_SIZE = 16
NUM_FOLDS = 5                 # 5-fold cross-validation
NUM_WORKERS = 4               # 並列データロード
IMAGE_HEIGHT = 512
IMAGE_WIDTH = 512
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
```

### ノートブック構造

| セル | 内容 |
|------|------|
| **1** | 環境設定（GPU確認、再現性設定、パス定義） |
| **2** | データセット定義（`EyeSegmentationDataset`） |
| **3** | モデル定義（`UNetMethod1/2/3`） |
| **4** | 損失関数・ユーティリティ関数 |
| **5** | **楕円キャッシュ生成**（Method1高速化用） |
| **6** | 学習ループ定義（`train_epoch`, `validate_epoch`, `run_train_fold`） |
| **7** | 評価関数定義（`evaluate_method1/2/3`） |
| **8** | 進捗確認・検証（Resume機能） |
| **9** | 進捗リセット（オプション） |
| **10** | **5-Fold CV実行**（メインループ） |
| **11** | 評価実行（各Foldごと） |
| **12** | 結果集計・保存 |
| **13** | 可視化（オプション） |
| **14** | メモリクリア（オプション） |

### 実行手順

#### 初回実行

1. **セル1-3**: 環境設定、データセット、モデル定義
2. **セル4**: 楕円キャッシュ生成（1-2分、Method1高速化用）
3. **セル5以降**: Run All

#### 中断後の再開

- そのまま **Run All** を再実行
- 完了済みタスクは自動スキップ
- 進捗は `cache/cv_progress.json` に保存

#### 最初からやり直す

1. **セル9**で `RESET_PROGRESS = True` に変更
2. セル9を実行（進捗リセット）
3. Run All

---

## ⚡ 高速化の取り組み

### 1. 楕円キャッシュ（Method1）

**問題**: Method1で、マスクから楕円パラメータへの変換が学習中に毎回実行され、ボトルネックになっていた。

**解決策**: 学習前に一括変換してキャッシュに保存

**実装:**
- **セル5**: `generate_ellipse_cache()` 関数
- **キャッシュファイル**: `cache/ellipse_params/ellipse_params.npz`
- **内容**: 各画像の虹彩・瞳孔楕円パラメータ（5パラメータ × 2）

**効果**: Method1の学習時間を**25%短縮**（500分 → 375分）

**コード位置:**
- キャッシュ生成: `crossvalidation.ipynb` セル5
- キャッシュ読み込み: `crossvalidation.ipynb` セル2（`EyeSegmentationDataset`）

---

### 2. 並列データロード（全メソッド）

**実装:**
```python
DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=True,
    num_workers=NUM_WORKERS,  # 4ワーカーで並列ロード
    pin_memory=True,          # GPU転送高速化
    persistent_workers=True   # ワーカー再利用
)
```

**効果**: 全メソッドで**20-30%高速化**

**注意点:**
- Windowsでは `NUM_WORKERS=0` の方が起動が速い場合がある
- Linux/Macでは `NUM_WORKERS=4-8` が推奨

---

### 3. sixcls直接読み込み（Method3）

**問題**: Method3で、6クラスラベルを個別マスクから合成していた（オンザフライ生成）。

**解決策**: 既存の`sixcls.png`を直接読み込み

**実装:**
- `EyeSegmentationDataset`で`sixcls.png`を直接読み込み
- `LossFunction3`で`gt_sixcls`を直接使用

**効果**: Method3の学習時間を**15-20%短縮**

**コード位置:**
- データセット: `crossvalidation.ipynb` セル2（`EyeSegmentationDataset.__getitem__`）
- 損失関数: `crossvalidation.ipynb` セル4（`LossFunction3`）

---

### 総合効果

| Method | 高速化前 | 高速化後 | 改善率 |
|--------|---------|---------|--------|
| Method1 | 500分 | 300分 | **40%** |
| Method2 | 500分 | 350-375分 | **25-30%** |
| Method3 | 500分 | 325-337分 | **30-35%** |

---

## 🔄 Resume機能の実装

### 背景

5-fold CVは長時間実行（15タスク × 数時間 = 数十時間）のため、中断・再開機能が必須。

### 実装内容

#### 進捗管理ファイル

**`cache/cv_progress.json`**:
```json
{
  "started_at": "2025-11-26 08:00:00",
  "last_update": "2025-11-26 14:30:15",
  "completed": {
    "method1_fold0": {
      "best_val_loss": 0.0857,
      "epoch": 151,
      "stopped_reason": "early_stop",
      "completed_at": "2025-11-26 10:15:20"
    },
    ...
  },
  "in_progress": {
    "method2_fold3": {
      "started_at": "2025-11-26 12:00:00",
      "current_epoch": 75,
      "current_best_loss": 0.0145,
      "last_update": "2025-11-26 14:30:15"
    }
  }
}
```

#### 記録タイミング（リアルタイム）

1. **学習開始時**: `in_progress`に即座に記録
2. **ベストモデル更新時**: エポック数と最良lossを更新
3. **学習完了時**: `completed`に移動
4. **途中で中断**: `in_progress`に記録が残る → 次回再学習

#### 検証機能

**3段階の検証**:
1. **ファイル存在チェック**: モデルファイルが存在するか
2. **ファイル整合性チェック**: サイズ、読み込み、キー確認
3. **完了状態チェック**: `stopped_reason`が`early_stop`または`completed`か

**終了理由の種類**:
- `completed`: NUM_EPOCHSまで完了 ✅ 正常
- `early_stop`: Early stoppingで終了 ✅ 正常
- `unknown`: 途中で中断 ❌ 異常 → 再学習

**コード位置:**
- 進捗管理: `crossvalidation.ipynb` セル8（`load_progress`, `save_progress`, `is_completed`, `mark_completed`）
- 検証: `crossvalidation.ipynb` セル8（`verify_model_file`）
- 進捗確認: `crossvalidation.ipynb` セル8（進捗確認セル）

---

## 📁 ファイル構成

### コードファイル

| ファイル | 説明 |
|---------|------|
| `process_data.ipynb` | データ前処理（CVAT XML → ラベル生成） |
| `train.ipynb` | 単一Foldでの学習・評価（50エポック） |
| `crossvalidation.ipynb` | **5-Fold CV実装（300エポック、Resume機能付き）** |

### データファイル

```
Images/
├── images/                    # 元画像（*.jpg）
├── labels_seg/               # 眼瞼系ラベル
│   ├── *_mask_lid.png
│   └── *_sixcls.png
└── labels_obb/               # 虹彩・瞳孔ラベル
    ├── *_mask_iris.png
    └── *_mask_pupil.png
```

### モデルファイル

```
model/
├── method1_fold0_best.pth   # train.ipynb用（50 epochs）
├── method2_fold0_best.pth
├── method3_fold0_best.pth
└── cv_300ep/                # crossvalidation.ipynb用（300 epochs）
    ├── method1_fold{0-4}_best.pth
    ├── method2_fold{0-4}_best.pth
    └── method3_fold{0-4}_best.pth
```

### キャッシュ・進捗管理

```
cache/
├── cv_progress.json          # CV進捗（Resume用）
└── ellipse_params/
    └── ellipse_params.npz    # 楕円パラメータキャッシュ
```

### 結果ファイル

```
results/
├── cv_train_method{1,2,3}_*.csv     # 学習結果（fold別）
├── cv_eval_method{1,2,3}_*.csv     # 評価結果（fold別）
├── cv_eval_summary_*.csv            # 評価サマリー
└── cv_comparison_*.csv             # 3手法比較
```

### メタデータ

- `fold_indices.json`: 5-fold GroupKFold分割情報
- `image_metadata.csv`: 画像メタデータ
- `patient_list.json`: 患者IDリスト

---

## 📊 実験結果

### 最新結果（2025-11-26 08:50:52）

#### 全体比較

| Method | Mean Dice | Std | 順位 |
|--------|-----------|-----|------|
| **Method3** | **0.9424** | **0.0051** | 🥇 1位 |
| **Method2** | **0.9147** | **0.0147** | 🥈 2位 |
| **Method1** | **0.8619** | **0.0129** | 🥉 3位 |

#### 各クラス別の性能

| Method | Eyelid Mean | Eyelid Std | Iris Mean | Iris Std | Pupil Mean | Pupil Std |
|--------|-------------|------------|-----------|----------|------------|-----------|
| Method1 | 0.9840 | 0.0023 | 0.8825 | 0.0055 | 0.7181 | 0.0376 |
| Method2 | 0.9523 | 0.0132 | 0.9018 | 0.0101 | 0.8901 | 0.0239 |
| Method3 | **0.9807** | **0.0023** | **0.9424** | **0.0079** | **0.9042** | **0.0128** |

#### 性能向上率（Method1基準）

| Metric | Method2 | Method3 |
|--------|---------|---------|
| Eyelid | -3.2% | -0.3% |
| Iris | +2.2% | **+6.8%** |
| Pupil | **+24.0%** | **+25.9%** |
| Mean | +6.1% | **+9.3%** |

### 詳細結果

詳細は `Results.md` を参照してください。

---

## 🔍 考察

### Method1の課題

- **Pupilの精度が低い**: Dice係数0.7181と、他のメソッドに比べて大幅に低い
- **Pupilの不安定性**: 標準偏差0.0376と、Fold間でばらつきが大きい
- **原因**: Ellipse回帰による近似誤差が、特にPupilで大きい可能性

### Method2の特徴

- **バランスが良い**: すべてのクラスで0.85以上の精度
- **Pupilの大幅改善**: Method1と比較して+24.0%の向上
- **Fold 4の性能低下**: 原因の調査が必要

### Method3の優位性

- **最高性能**: Mean Dice 0.9424で、Method1より+9.3%、Method2より+3.0%向上
- **最も安定**: 標準偏差0.0051と、Fold間のばらつきが最小
- **全クラスで高精度**: Eyelid, Iris, Pupilすべてで0.90以上
- **Ellipse補正の効果**: Irisで+0.0649の向上（Ellipse補正前後比較）

---

## 🎯 結論

1. **Method3が最優秀**: Mean Dice 0.9424で最高性能かつ最も安定
2. **Method2はバランス型**: すべてのクラスで良好な性能
3. **Method1はEyelid特化**: Eyelidは最高だが、Pupilの精度が課題

**推奨**: 実用には**Method3**を採用することを推奨します。

---

## 📚 参考資料

- **README.md**: プロジェクト全体の説明
- **Results.md**: 実験結果の詳細サマリー
- **improvement.md**: 実装の改善履歴と技術的詳細

---

## 🔧 技術スタック

- **Python**: 3.8+
- **PyTorch**: 2.0+（CUDA版）
- **OpenCV**: 画像処理・楕円フィッティング
- **scikit-image**: 画像処理・モルフォロジー演算
- **scikit-learn**: GroupKFold分割
- **Jupyter Notebook**: 実験環境

---

## 📝 実験ログ

### 2025-11-26
- 最新の5-Fold CV結果を取得
- Method3が最高性能を確認（Mean Dice: 0.9424）
- Results.mdを作成

### 2025-12-xx 〜 2026-01-01（追加：後処理 ablation + YOLO/SegFormer）
- **目的**: 虹彩/瞳孔の楕円化で「露出弧のみ（OuterArc）」と「full mask最大外輪郭（FullMax）」を比較し、遮蔽領域（occ）活用の効果を検証。
- **共通の評価方針**: 512×512、Dice（Eyelid/Iris/Pupil/Mean）、被験者クラスタ（subject_id=filename先頭）で統計（置換検定＋bootstrap CI）。
- **Method3（U-Net）後処理比較（5-fold）**:
  - raw / outerarc / fullmax / ransac(whole) / ransac(arc) を同一定義で比較（`crossvalidation.ipynb`）。
  - 結論: **FullMax が OuterArc を有意に上回る**（被験者クラスタ、Holm補正）。
- **YOLO ablation（YOLO11l-seg）**:
  - custom評価の重大バグを修正（YOLOのmask floatを `uint8` に早期変換していた→閾値処理が壊れてDiceが過小評価）。
  - 推論画像は DataLoader 変換画像ではなく **元画像パスから直接推論**するように統一。
  - 結論: YOLOでも **FullMax > OuterArc** の傾向（被験者クラスタで有意）。
- **SegFormer ablation（SegFormer-B2）**:
  - `ablation_yolo11_clean_final.ipynb` と **完全に同一の評価指標・後処理**（raw/outerarc/fullmax/ransac）に合わせて `ablation_SegFormer.ipynb` を更新。
  - RANSACで退化点群により `RuntimeWarning` が出ることがあるため、破綻パラメータ（NaN/inf等）を弾くガードを追加（評価は継続）。
  - 主要CSV:
    - fold平均: `results/segformer_eval_{mode}_*.csv`
    - per-image: `results/segformer_eval_perimage_{mode}_*.csv`
  - 2026-01-01 時点の集計（被験者=122）:
    - fold平均Mean Dice: raw 0.9384 / outerarc 0.9518 / **fullmax 0.9607** / ransac_whole 0.9588 / ransac_arc 0.9471
    - 被験者平均Mean Dice: raw 0.9340 / outerarc 0.9443 / **fullmax 0.9558** / ransac_whole 0.9540 / ransac_arc 0.9398
    - 直接比較（被験者クラスタ）: **FullMax − OuterArc = +0.0115**（CI95% [+0.0068, +0.0172]、p_perm < 5e-5）

### 2025-11-25
- 複数回の実験で結果の一貫性を確認
- Resume機能の改善（終了理由の検証を追加）

### 2025-11-16
- 初期の5-Fold CV結果を取得
- Method3の優位性を確認

### 2025-11-14
- crossvalidation.ipynbの実装完了
- Resume機能の実装完了
- 高速化機能の実装完了

---

## 🚀 今後の改善案

1. **Method1の改善**
   - Pupilの精度向上（楕円近似の改善）
   - より高度な楕円フィッティング手法の検討

2. **Method2の改善**
   - Fold 4の性能低下の原因調査
   - エッジ生成の最適化

3. **Method3の改善**
   - Ellipse補正の最適化
   - より高度な後処理手法の検討

4. **全体的な改善**
   - データ拡張の強化
   - アンサンブル手法の検討
   - 推論速度の最適化

