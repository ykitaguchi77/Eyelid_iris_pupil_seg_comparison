# Method3 Ablation Study: モダンなセグメンテーションモデル候補

## 📋 調査目的

Method3（6-Class Region Segmentation）のablation studyとして、他のモダンなセグメンテーションモデルを用いた場合の精度を調査するための候補モデルを選定する。

---

## 🎯 候補モデル一覧

### 1. YOLO11n-seg（既存候補）

**概要:**
- Ultralytics社の最新YOLOモデル（2024年リリース）
- インスタンスセグメンテーション対応
- リアルタイム推論に最適化

**特徴:**
- ✅ **高速推論**: リアルタイム処理に最適
- ✅ **軽量**: モバイル・エッジデバイス対応
- ✅ **実装が容易**: Ultralyticsライブラリで簡単に利用可能
- ⚠️ **精度**: セマンティックセグメンテーション専用モデルよりやや低い可能性

**適用可能性:**
- インスタンスセグメンテーション → セマンティックセグメンテーションへの変換が必要
- 6クラス分類への適応が必要

**リソース:**
- GitHub: https://github.com/ultralytics/ultralytics
- ドキュメント: https://docs.ultralytics.com/

---

### 2. SegFormer（Transformerベース）

**概要:**
- NVIDIA社が開発（2021年）
- Vision Transformer（ViT）ベースのセマンティックセグメンテーションモデル
- 軽量かつ高精度

**特徴:**
- ✅ **高精度**: ADE20KでmIoU 84.0%（SegFormer-B5）
- ✅ **効率的**: CNNベースより少ないパラメータで高精度
- ✅ **マルチスケール特徴**: 階層的なTransformerエンコーダ
- ✅ **PyTorch実装**: Hugging Face Transformersで利用可能

**モデルサイズ:**
- SegFormer-B0: 3.7M params（軽量）
- SegFormer-B1: 13.2M params
- SegFormer-B2: 27.4M params
- SegFormer-B3: 47.3M params
- SegFormer-B4: 64.1M params
- SegFormer-B5: 84.7M params（最高精度）

**適用可能性:**
- ✅ セマンティックセグメンテーション専用
- ✅ 6クラス分類に直接適用可能
- ✅ 医療画像での実績あり

**リソース:**
- Paper: https://arxiv.org/abs/2105.15203
- Hugging Face: https://huggingface.co/docs/transformers/model_doc/segformer
- 実装: `transformers.SegformerForSemanticSegmentation`

---

### 3. Mask2Former（Transformerベース、SOTA）

**概要:**
- Meta AI（Facebook Research）が開発（2022年）
- セマンティック・インスタンス・パノプティックセグメンテーションを統一
- 2022年時点でSOTA達成

**特徴:**
- 🏆 **SOTA性能**: ADE20KでmIoU 57.8%（セマンティック）
- ✅ **統一アーキテクチャ**: セマンティック/インスタンス/パノプティックを1つのモデルで
- ✅ **マスクアテンション**: マスクベースのアテンション機構
- ✅ **高精度**: 細かい境界も正確にセグメンテーション

**モデルサイズ:**
- Mask2Former-S: 44M params
- Mask2Former-B: 61M params
- Mask2Former-L: 195M params

**適用可能性:**
- ✅ セマンティックセグメンテーション専用
- ✅ 6クラス分類に直接適用可能
- ⚠️ 計算コストがやや高い

**リソース:**
- Paper: https://arxiv.org/abs/2112.01527
- GitHub: https://github.com/facebookresearch/Mask2Former
- Detectron2ベースの実装

---

### 4. DeepLabV3+（CNNベース、実績豊富）

**概要:**
- Google Researchが開発（2018年）
- アトラス空間ピラミッドプーリング（ASPP）とデコーダを組み合わせ
- 長年SOTAを維持した実績豊富なモデル

**特徴:**
- ✅ **実績豊富**: 多くのベンチマークで高精度
- ✅ **安定性**: 様々なデータセットで安定した性能
- ✅ **実装が容易**: PyTorch公式実装あり
- ✅ **バックボーン選択可能**: ResNet, Xception, MobileNet等

**バックボーン:**
- DeepLabV3+ (ResNet-50): 39.7M params
- DeepLabV3+ (ResNet-101): 58.2M params
- DeepLabV3+ (Xception-65): 41.1M params

**適用可能性:**
- ✅ セマンティックセグメンテーション専用
- ✅ 6クラス分類に直接適用可能
- ✅ 医療画像での実績あり

**リソース:**
- Paper: https://arxiv.org/abs/1802.02611
- PyTorch: `torchvision.models.segmentation.deeplabv3_resnet50`
- GitHub: https://github.com/tensorflow/models/tree/master/research/deeplab

---

### 5. SegNeXt（CNNベース、最新）

**概要:**
- 2022年発表
- マルチスケール畳み込みアテンション（MSCA）を導入
- CNNベースながらTransformer並みの性能

**特徴:**
- ✅ **高精度**: ADE20KでmIoU 55.2%（SegNeXt-S）
- ✅ **効率的**: Transformerより軽量で高速
- ✅ **CNNベース**: 実装がシンプル
- ✅ **マルチスケール**: 異なるスケールの特徴を効果的に統合

**モデルサイズ:**
- SegNeXt-T: 4.2M params
- SegNeXt-S: 13.9M params
- SegNeXt-B: 27.4M params
- SegNeXt-L: 49.7M params

**適用可能性:**
- ✅ セマンティックセグメンテーション専用
- ✅ 6クラス分類に直接適用可能
- ✅ 医療画像での実績あり

**リソース:**
- Paper: https://arxiv.org/abs/2209.08575
- GitHub: https://github.com/uyzhang/PaddleSeg/tree/develop/configs/segnext

---

### 6. InternImage（ConvNeXtベース、2023年SOTA）

**概要:**
- 2023年発表
- 大規模畳み込みニューラルネットワーク
- ImageNet分類でSOTA達成後、セグメンテーションでも高精度

**特徴:**
- 🏆 **SOTA性能**: ADE20KでmIoU 60.1%（InternImage-XL）
- ✅ **ConvNeXtベース**: CNNの最新技術を統合
- ✅ **スケーラブル**: 小規模から大規模まで対応
- ⚠️ **計算コスト**: 大規模モデルは重い

**モデルサイズ:**
- InternImage-T: 30M params
- InternImage-S: 50M params
- InternImage-B: 112M params
- InternImage-L: 256M params
- InternImage-XL: 368M params

**適用可能性:**
- ✅ セマンティックセグメンテーション対応
- ✅ 6クラス分類に直接適用可能
- ⚠️ 大規模モデルはメモリ・計算リソースが必要

**リソース:**
- Paper: https://arxiv.org/abs/2211.05778
- GitHub: https://github.com/OpenGVLab/InternImage

---

### 7. SAM2（Segment Anything Model 2）

**概要:**
- Meta AIが開発（2024年）
- ゼロショットセグメンテーション
- プロンプト（ポイント、ボックス、マスク）からセグメンテーション

**特徴:**
- ✅ **ゼロショット**: 学習データにない画像でもセグメンテーション可能
- ✅ **汎用性**: 様々なドメインに適用可能
- ⚠️ **セマンティックセグメンテーション**: インスタンスセグメンテーションが主目的
- ⚠️ **6クラス分類**: 適応が必要

**モデルサイズ:**
- SAM2-Tiny: 39M params
- SAM2-Small: 46M params
- SAM2-Base: 93M params
- SAM2-Large: 225M params

**適用可能性:**
- ⚠️ インスタンスセグメンテーション → セマンティックセグメンテーションへの変換が必要
- ⚠️ 6クラス分類への適応が必要
- ✅ ゼロショット性能が高い

**リソース:**
- Paper: https://arxiv.org/abs/2311.15796
- GitHub: https://github.com/facebookresearch/segment-anything-2

---

### 8. RT-DETR（Real-Time DETR）

**概要:**
- Baiduが開発（2023年）
- リアルタイム物体検出・セグメンテーション
- DETRベースだが高速化

**特徴:**
- ✅ **リアルタイム**: 高速推論
- ✅ **高精度**: DETRベースの精度を維持
- ⚠️ **セグメンテーション**: 主に物体検出・インスタンスセグメンテーション
- ⚠️ **6クラス分類**: 適応が必要

**適用可能性:**
- ⚠️ インスタンスセグメンテーション → セマンティックセグメンテーションへの変換が必要
- ⚠️ 6クラス分類への適応が必要
- ✅ リアルタイム性能が高い

**リソース:**
- Paper: https://arxiv.org/abs/2304.08069
- GitHub: https://github.com/lyuwenyu/RT-DETR

---

### 9. RF-DETR（Roboflow Detection Transformer）

**概要:**
- Roboflow社が開発（2024年）
- COCOデータセットで60以上のmAPを達成した初のリアルタイムモデル
- Transformerベースのリアルタイム物体検出

**特徴:**
- 🏆 **高精度**: COCOでmAP 60+（リアルタイムモデル初）
- ✅ **リアルタイム**: 高速推論性能
- ✅ **汎用性**: 様々なドメイン・データセットに適応可能
- ✅ **オープンソース**: Apache 2.0ライセンス（商用利用可能）
- ⚠️ **セグメンテーション**: 主に物体検出・インスタンスセグメンテーション
- ⚠️ **6クラス分類**: 適応が必要

**性能:**
- COCO mAP: 60+（リアルタイムモデル初）
- RF100-VLデータセットでもSOTA性能

**適用可能性:**
- ⚠️ インスタンスセグメンテーション → セマンティックセグメンテーションへの変換が必要
- ⚠️ 6クラス分類への適応が必要
- ✅ リアルタイム性能が高い
- ✅ 高精度（RT-DETRより精度が高い可能性）

**リソース:**
- Blog: https://blog.roboflow.com/rf-detr/
- GitHub: Roboflow公式リポジトリ（要確認）
- ライセンス: Apache 2.0

---

## 📊 モデル比較表

| モデル | アーキテクチャ | セマンティック対応 | 6クラス直接適用 | 精度（ADE20K mIoU） | 推論速度 | 実装容易度 | 推奨度 |
|--------|--------------|------------------|----------------|---------------------|---------|-----------|--------|
| **YOLO11n-seg** | CNN | ⚠️ 要変換 | ⚠️ 要適応 | N/A | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ |
| **SegFormer-B2** | Transformer | ✅ | ✅ | ~51.8% | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **Mask2Former-S** | Transformer | ✅ | ✅ | ~57.8% | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **DeepLabV3+** | CNN | ✅ | ✅ | ~45.5% | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ | ⭐⭐⭐⭐ |
| **SegNeXt-S** | CNN | ✅ | ✅ | ~55.2% | ⭐⭐⭐⭐ | ⭐⭐⭐⭐ | ⭐⭐⭐⭐⭐ |
| **InternImage-S** | CNN | ✅ | ✅ | ~56.1% | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐⭐ |
| **SAM2** | Transformer | ⚠️ 要変換 | ⚠️ 要適応 | N/A | ⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **RT-DETR** | Transformer | ⚠️ 要変換 | ⚠️ 要適応 | N/A | ⭐⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐ |
| **RF-DETR** | Transformer | ⚠️ 要変換 | ⚠️ 要適応 | COCO mAP 60+ | ⭐⭐⭐⭐ | ⭐⭐⭐ | ⭐⭐⭐ |

---

## 🎯 推奨候補（優先順位）

### Tier 1: 最優先候補（セマンティックセグメンテーション専用、高精度）

1. **SegFormer-B2/B3**
   - ✅ セマンティックセグメンテーション専用
   - ✅ 6クラス分類に直接適用可能
   - ✅ Hugging Faceで簡単に利用可能
   - ✅ 医療画像での実績あり
   - ✅ バランスが良い（精度・速度・実装容易度）

2. **SegNeXt-S/B**
   - ✅ セマンティックセグメンテーション専用
   - ✅ 6クラス分類に直接適用可能
   - ✅ CNNベースで実装がシンプル
   - ✅ 高精度（Transformer並み）

3. **Mask2Former-S**
   - ✅ セマンティックセグメンテーション専用
   - ✅ 6クラス分類に直接適用可能
   - ✅ SOTA性能
   - ⚠️ 計算コストがやや高い

### Tier 2: 実績豊富な候補

4. **DeepLabV3+ (ResNet-50/101)**
   - ✅ セマンティックセグメンテーション専用
   - ✅ 6クラス分類に直接適用可能
   - ✅ 実装が非常に容易（PyTorch公式）
   - ✅ 医療画像での実績豊富
   - ⚠️ 最新モデルよりやや精度が低い可能性

### Tier 3: 特殊用途候補

5. **RF-DETR**
   - ✅ リアルタイム推論に最適
   - ✅ 高精度（COCO mAP 60+）
   - ✅ オープンソース（Apache 2.0）
   - ⚠️ インスタンス→セマンティック変換が必要
   - ⚠️ 6クラス分類への適応が必要

6. **YOLO11n-seg**
   - ✅ リアルタイム推論に最適
   - ✅ 実装が容易
   - ⚠️ インスタンス→セマンティック変換が必要
   - ⚠️ 6クラス分類への適応が必要

7. **RT-DETR**
   - ✅ リアルタイム推論に最適
   - ⚠️ RF-DETRより精度がやや低い可能性
   - ⚠️ インスタンス→セマンティック変換が必要
   - ⚠️ 6クラス分類への適応が必要

8. **InternImage-S**
   - ✅ 高精度
   - ✅ セマンティックセグメンテーション対応
   - ⚠️ 計算コストがやや高い

---

## 💡 推奨アプローチ

### Phase 1: セマンティックセグメンテーション専用モデル（優先）

1. **SegFormer-B2** または **SegNeXt-S**
   - 理由: バランスが良く、実装が容易
   - 期待効果: Method3（U-Net）と同等またはそれ以上の精度

2. **Mask2Former-S**
   - 理由: SOTA性能
   - 期待効果: 最高精度の可能性

3. **DeepLabV3+ (ResNet-50)**
   - 理由: ベースライン比較用
   - 期待効果: 実績豊富なモデルとの比較

### Phase 2: リアルタイムモデル（オプション）

4. **RF-DETR**
   - 理由: リアルタイム推論が必要で、かつ高精度が求められる場合
   - 期待効果: RT-DETRやYOLO11n-segより高精度の可能性

5. **YOLO11n-seg**
   - 理由: リアルタイム推論が必要な場合
   - 期待効果: 速度重視の比較

6. **RT-DETR**
   - 理由: リアルタイム推論が必要な場合（RF-DETRとの比較用）
   - 期待効果: RF-DETRとの性能比較

---

## 🔧 実装時の考慮事項

### 1. データ形式の統一

- **入力**: 512×512 RGB画像（既存と同じ）
- **出力**: 512×512 6クラスセグメンテーションマスク
- **評価指標**: Dice係数（Eyelid/Iris/Pupil/Mean）

### 2. 学習設定の統一

- **エポック数**: 300 epochs
- **Early stopping**: 30 epochs
- **Batch size**: 16
- **Learning rate**: 1e-3（モデルに応じて調整）
- **Data augmentation**: 既存と同じ

### 3. バックボーン・事前学習

- **ImageNet事前学習**: 可能な限り使用
- **医療画像事前学習**: 利用可能な場合は検討（MedSAM等）

### 4. 評価方法

- **5-Fold Cross-Validation**: 既存と同じ
- **比較指標**: 
  - Mean Dice係数
  - 各クラス（Eyelid/Iris/Pupil）のDice係数
  - 標準偏差（Fold間の安定性）
  - 推論速度（FPS）

---

## 📚 参考リソース

### 論文・技術資料

1. **SegFormer**: https://arxiv.org/abs/2105.15203
2. **Mask2Former**: https://arxiv.org/abs/2112.01527
3. **DeepLabV3+**: https://arxiv.org/abs/1802.02611
4. **SegNeXt**: https://arxiv.org/abs/2209.08575
5. **InternImage**: https://arxiv.org/abs/2211.05778
6. **SAM2**: https://arxiv.org/abs/2311.15796
7. **RF-DETR**: https://blog.roboflow.com/rf-detr/

### 実装リソース

1. **Hugging Face Transformers**: https://huggingface.co/docs/transformers
2. **PyTorch Vision**: https://pytorch.org/vision/stable/models.html
3. **Detectron2**: https://github.com/facebookresearch/detectron2
4. **Ultralytics YOLO**: https://docs.ultralytics.com/
5. **Roboflow RF-DETR**: https://blog.roboflow.com/rf-detr/

---

## 🎯 次のステップ

1. **候補モデルの選定**: Tier 1から1-2モデルを選定
2. **実装準備**: 選定モデルの実装方法を調査
3. **データローダー適応**: 既存の`EyeSegmentationDataset`との統合
4. **学習ループ実装**: `crossvalidation.ipynb`への統合
5. **評価・比較**: Method3（U-Net）との性能比較

---

## 📝 備考

- **医療画像特化モデル**: MedSAM等の医療画像事前学習モデルも検討可能
- **アンサンブル**: 複数モデルのアンサンブルも検討可能
- **転移学習**: ImageNet事前学習モデルをベースに、医療画像でファインチューニング
- **ハイパーパラメータ調整**: 各モデルに最適な学習率・バッチサイズ等を調整

