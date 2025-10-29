# experiment_plan（CVAT XML／患者IDグループ分割／画像ID 0–1999）

## 0. 目的・仮説
- **目的**：3手法（1️⃣ 眼瞼1クラス＋楕円回帰、2️⃣ 縁セグ［補足］、3️⃣ 5クラス領域セグ）を**同一バックボーン・同一前処理**で公平比較。  
- **主評価**：Dice（クラス別）。副次：推論速度（ms/枚）。  
- **仮説**：3️⃣が総合Diceで最良。

---

## 1. データと分割（CVAT XML 前提）
- **入力画像**：RGB、640×640（中心クロップ/リサイズで統一）。  
- **アノテ形式**：CVAT XML  
  - 眼瞼（YOLO-Seg相当）：`label="eyelid"`（クラス0）、`label="caruncle"`（クラス1）などのpolygon/bitmap。  
  - 虹彩/瞳孔（YOLO-OBB相当）：`label="iris"` / `label="pupil"` の **rotated box (rbox)** または矩形polygon。
- **使用データの限定**：**CVATの image id が 0〜1999** のフレームのみ採用。  
- **患者IDの定義**：ファイル名の **basename 先頭の整数**（例：`1-2014...jpg` → **patient_id=1**）。  
- **分割**：**GroupKFold(n=5, groups=patient_id)**（同一patient_idが train/val を跨がないよう厳守）。  
- **層化の目安**：患者ごとの「虹彩/瞳孔注釈の有無」「遮蔽率」分布を粗く均衡化。  
- **シード**：`global_seed=42`。

---

## 2. ラベル生成（統一前処理）
### 2.1 眼瞼（lid＋lacrimal 統合）
- CVAT→ラスタライズ（640×640）：`mask_lid`, `mask_lacrimal`。  
- **統合**：`M_lid = union(mask_lid, mask_lacrimal)`（= 眼瞼＋涙丘）。  
- ノイズ除去：必要に応じ **opening/closing 1px**。

### 2.2 虹彩・瞳孔（OBB→楕円）
- CVAT rbox/rect から (cx, cy, w, h, θ) を取得。  
- **内接楕円**：a=w/2, b=h/2, angle=θ。  
- ラスタライズ：`E_iris`, `E_pupil`（640×640、バイナリ）。  
- 一貫性：`E_pupil ← E_pupil ∩ E_iris` を強制。

### 2.3 3️⃣用 5クラス教師（露出/隠れ）
- **クラスID**：`{0: lid(涙丘含む), 1: iris_vis, 2: iris_occ, 3: pupil_vis, 4: pupil_occ}`（背景は暗黙）。  

- **眼瞼マスクの定義**（重要）：
  - 眼瞼マスク = **眼瞼縁に囲まれた部分** = **眼球の露出部分**
  - つまり、眼瞼マスクの中 = **見えている部分**、眼瞼マスクの外 = **隠れている部分**

- **定義（数学的表現）**：  
  - `iris_vis = E_iris ∩ M_lid`  （虹彩と眼瞼の**重なる**部分 = **見える部分**）
  - `iris_occ = E_iris \ M_lid`  （虹彩のうち、眼瞼と**重ならない**部分 = **隠れている部分**）
  - `pupil_vis = E_pupil ∩ M_lid` （瞳孔と眼瞼の**重なる**部分 = **見える部分**）
  - `pupil_occ = E_pupil \ M_lid`  （瞳孔のうち、眼瞼と**重ならない**部分 = **隠れている部分**）

**注意**：
- `vis` (visible) = **見えている** = **眼瞼マスクの中**（眼瞼縁に囲まれた領域内）
- `occ` (occluded) = **隠されている** = **眼瞼マスクの外**（眼瞼で覆われている）

### 2.4 2️⃣用 縁教師（補足）
- 各マスクの **細線化（1–2px）** により 眼瞼縁/虹彩縁/瞳孔縁 を生成。

---

## 3. モデル共通仕様
- **バックボーン**：ResNet-50 + FPN（または ConvNeXt-T）を**全手法で共有**。  
- **前処理**：ImageNet mean/std 正規化。  
- **オーグメント（全手法共通）**：Horizontal Flip、Rotate ±7°、弱 ColorJitter、軽 Cutout。  
- **最適化**：AdamW（lr=1e-3, wd=1e-4）、Cosine decay、warmup 200 iters。  
- **学習**：bs=16、AMP有効、max 300 epochs、早期停止（patience=30, metric=val_Dice_macro）。  
- **実装**：PyTorch/Lightning 推奨、`cudnn.benchmark=True`。

---

## 4. 手法別設定・損失
### 4.1 1️⃣ 眼瞼=セグ1クラス／虹彩・瞳孔=楕円回帰（既報準拠・最小）
- **出力**  
  - SegHead：眼瞼（1ch, sigmoid）。  
  - EllipseHead×2：各 \[(cx,cy,w,h,θ)\]（0–1正規化、θは sin/cos 表現可）。
- **損失**  
  - 眼瞼：**BCE + Dice**（1:1）。  
  - 楕円：**微分可能レンダ → BCE**（教師は `E_iris`, `E_pupil`）。  
  - 合計：`L = L_lid + λ·L_ellipse`（λ=1）。  
- **評価**：レンダした虹彩/瞳孔マスクでDice算出（3️⃣と整合）。

### 4.2 2️⃣ 縁セグ（補足）
- **出力**：3クラス縁（眼瞼/虹彩/瞳孔）。  
- **損失**：**CE + Generalized Dice + 距離重みCE**（または Focal + Balanced CE）。  
- **備考**：主表からは外し、付録実験。

### 4.3 3️⃣ 5クラス領域セグ（主比較・Diceのみ）
- **出力**：5ch（lid, iris_vis, iris_occ, pupil_vis, pupil_occ, softmax）。  
- **損失**：**Multi-class Dice のみ**（背景除外平均、重みなしで開始）。

---

## 5. 評価プロトコル
- **分割**：**GroupKFold（患者ID）** 5-fold（train/val で患者は非重複）。  
- **主指標**  
  - 3️⃣：5クラスDice（lid／iris_vis／iris_occ／pupil_vis／pupil_occ）。  
  - 1️⃣：眼瞼Dice、虹彩Dice（レンダ）、瞳孔Dice（レンダ）。  
- **合算（参考）**：虹彩（vis∪occ）Dice、瞳孔（vis∪occ）Dice。  
- **速度**：A100/4090、FP16、batch=1、前向き平均ms/枚。  
- **統計**：5-fold平均±SD。**対応のあるt検定**または**線形混合モデル**で 1️⃣ vs 3️⃣ を比較（主要：眼瞼、虹彩合算、瞳孔合算）。

---

## 6. アブレーション（最小）
- 1️⃣：λ ∈ {0.5, 1.0, 2.0}。  
- Backbone：ResNet-50 vs ConvNeXt-T（付録）。

---

## 7. 品質管理
- **自己整合性**：`E_pupil ⊂ E_iris` を強制。  
- **眼瞼＋涙丘統合**後の穴・重複を opening/closing 1px で軽整形。  
- **過学習検知**：train/val Dice 乖離>0.1 でAugを一段強化。  
- **保存**：fold別 ckpt・学習曲線・評価CSV。

## 8. スクリプトと構成
images\images\ #元画像
Images\eyelid_caruncle_seg_0-2000.xml #セグメンテーションのannotation (cvat)
Images\obb_iris_pupil_1-3000.xml #obbのannotation (cvat)
model
process_data.ipynb
train.ipynb
crossvalidation.ipynb
readme.md
log.md
improvement.md
- コードが散らからないように、最初はipynbでまとめてください。何をしているかわかるようにマークダウンもつけて。
- ユーザが使いやすいようにreadme.mdも随時更新
- 変更履歴をlog.mdに追記して下さい
- エラーが出て対処した点や工夫してうまくいった点をimprovement.mdに記載して次に生かせるようにしてください

## 9. 実行順序

**⚠️ 重要：実行方法**
- 各ノートブック（`process_data.ipynb`、`train.ipynb`）は**「すべて実行（Run All）」で上から順番に実行されることを前提に設計**されています。
- セルを個別に実行する場合は、セル間の依存関係に注意してください。
- 「すべて実行」により、必要な変数や関数が正しい順序で定義され、エラーなく実行できます。

**実行ステップ**：
1. `process_data.ipynb`：cvat形式のxmlからアノーテーションを抜き出してトレーニングに使う形式に成形。basename先頭整数からpatient_idを抽出してGroupKfoldリストを作成
2. `train.ipynb`：3種類のexperimentに関してトレーニングをしてモデル作成、テスト画像のinferenceまで。
3. `crossvalidation.ipynb`：crossvalidationを行いDiceの比較表を作成


---

## 10. 成果物（論文用）
- **主表**：1️⃣ vs 3️⃣ の Dice（眼瞼／虹彩合算／瞳孔合算）＋速度。  
- **付録表**：3️⃣ 5クラス詳細、2️⃣ 縁結果、λ・膨張量・Backbone差。

---

## 11. 実装プラン

### 11.1 技術的決定事項
- **アノテーションファイル範囲**：
  - `eyelid_caruncle_seg_0-2000.xml`（image id 0-2000）→ 使用: 0-1999
  - `obb_iris_pupil_1-3000.xml`（image id 1-3000）→ 使用: 1-1999
- **楕円データ形式**：CVAT ellipse（cx, cy, rx, ry）を使用（回転角は考慮しない）
- **バックボーン**：最初はResNet-50 + FPNで実装
- **ノートブック構成**：`process_data.ipynb`に統一（plan.mdの矛盾修正）

### 11.2 実装Phase

#### Phase 1: データ前処理 (`process_data.ipynb`)
**目的**：CVAT XMLをパースしてトレーニング用ラベルを生成

**実装内容**：
1. **CVAT XMLパース**：
   - `eyelid_caruncle_seg_0-2000.xml`からEyelid, Caruncleのpolygon取得
   - `obb_iris_pupil_1-3000.xml`からIris, Pupilのellipse取得
   - image id 0-1999のみ抽出

2. **患者ID抽出**：
   - ファイル名のbasename先頭整数を抽出（例：`1-2014...jpg` → patient_id=1）
   - 全画像のpatient_idをリスト化

3. **画像・ラベル処理**（640×640）：
   - 画像をリサイズ/中心クロップ（640×640）
   - ラベル生成：
     - `mask_eyelid`, `mask_caruncle`をラスタライズ
     - `M_lid = union(mask_eyelid, mask_caruncle)`（統合眼瞼）
     - Iris ellipse→`E_iris`、Pupil ellipse→`E_pupil`ラスタライズ
     - `E_pupil = E_pupil ∩ E_iris`を強制（一貫性保証）
   - 5クラス教師生成（3️⃣用）：
     - `iris_vis = E_iris ∩ M_lid`
     - `iris_occ = E_iris \ M_lid`
     - `pupil_vis = E_pupil ∩ M_lid`
     - `pupil_occ = E_pupil \ M_lid`

4. **GroupKFold分割**：
   - `GroupKFold(n_splits=5, groups=patient_id)`
   - 同一patient_idがtrain/valを跨がないよう分割
   - 各foldのtrain/valリストを保存

5. **保存形式**：
   - 画像とラベルの対応リスト（CSV or JSON）
   - fold別のtrain/valリスト
   - メタデータ（患者ID、アノテ有無フラグなど）

#### Phase 2: 学習 (`train.ipynb`)
**目的**：3手法のモデル学習と評価

**実装内容**：
1. **モデル実装**：
   - 方法1️⃣：ResNet-50+FPN → SegHead(眼瞼1ch) + EllipseHead×2
   - 方法2️⃣：ResNet-50+FPN → EdgeHead(3クラス縁)
   - 方法3️⃣：ResNet-50+FPN → SegHead(5クラス)

2. **学習ループ**：
   - 共通前処理：ImageNet正規化
   - 共通Aug：Horizontal Flip、Rotate ±7°、弱 ColorJitter、軽 Cutout
   - 各手法の損失関数実装
   - AMP有効、早期停止、fold別ckpt保存

#### Phase 3: 交差検証 (`crossvalidation.ipynb`)
**目的**：5-fold GroupKFold評価と比較表生成

**実装内容**：
1. 各foldで学習済みモデルをロード
2. 検証セットでDice算出（クラス別）
3. 5-fold平均±SD計算
4. 1️⃣ vs 3️⃣ の統計的比較（対応のあるt検定）
5. 比較表出力（CSV/HTML）

### 11.3 次のステップ
- [x] `process_data.ipynb`の実装完了（2025-01-XX）
- [x] CVAT XMLパース機能実装
- [x] ラベル生成とGroupKFold分割実装
- [ ] Phase 2: `train.ipynb`の実装（モデル定義、学習ループ）
- [ ] Phase 3: `crossvalidation.ipynb`の実装（評価・比較表）
- [ ] 実装完了時の検証と評価

### 11.4 実装の進捗
- **Phase 1（データ前処理）**: ✅ 完了
- **Phase 2（学習）**: 🚧 次に実装
- **Phase 3（交差検証）**: ⏳ 未実装

改善点や工夫点は `improvement.md` に記録していきます。