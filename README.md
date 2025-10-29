# Eyelid, Iris, Pupil Segmentation Comparison

眼瞼・虹彩・瞳孔のセグメンテーション手法の比較実験プロジェクト

## 環境セットアップ

### 1. 仮想環境の作成とアクティベート

```powershell
# 仮想環境の作成
python -m venv venv

# 仮想環境のアクティベート（PowerShell）
.\venv\Scripts\Activate.ps1

# 仮想環境のアクティベート（Command Prompt）
venv\Scripts\activate.bat
```

### 2. 依存パッケージのインストール

```powershell
# 仮想環境がアクティベートされていることを確認
pip install -r requirements.txt
```

### 3. PyTorchの確認

```powershell
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

**注意**: このプロジェクトは**GPU版のPyTorch**を使用します。CPU版の場合は、以下のコマンドでGPU版に更新してください：
```powershell
.\venv\Scripts\Activate.ps1
pip uninstall torch torchvision -y
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

## プロジェクト構成

```
Eyelid_Iris_pupil_seg_comparison/
├── Images/
│   ├── images/                      # 元画像 (4467枚)
│   ├── eyelid_caruncle_seg_0-2000.xml  # 眼瞼・涙丘セグメンテーションアノテーション (CVAT形式)
│   ├── obb_iris_pupil_1-3000.xml      # 虹彩・瞳孔OBBアノテーション (CVAT形式)
│   ├── labels_obb/                   # OBBラベル
│   └── labels_seg/                    # セグメンテーションラベル
├── model/                            # 学習済みモデルの保存先
├── venv/                             # Python仮想環境
├── experimet_plan.md                 # 実験計画書
├── process_data.ipynb                # データ前処理ノートブック
├── train.ipynb                       # 学習ノートブック
├── crossvalidation.ipynb             # 交差検証ノートブック
├── requirements.txt                  # Python依存パッケージ一覧
└── README.md                         # このファイル
```

## 実験計画概要

3つの手法を比較します：

1. **方法1**: 眼瞼セグメンテーション（1クラス）＋ 虹彩・瞳孔楕円回帰
2. **方法2**: 縁セグメンテーション（補足実験）
3. **方法3**: 5クラス領域セグメンテーション（主要比較手法）

詳細は `experimet_plan.md` を参照してください。

## 実行手順

### 1. データ前処理
```powershell
# Jupyter Notebook を起動
.\venv\Scripts\Activate.ps1
jupyter notebook

# process_data.ipynb を開いて実行
```

### 2. モデル学習
```powershell
# train.ipynb を開いて実行
# 3つの手法について学習を実行
```

### 3. 交差検証と評価
```powershell
# crossvalidation.ipynb を開いて実行
# 5-fold GroupKFoldによる評価と比較表の生成
```

## データ仕様

- **画像サイズ**: 640×640ピクセル（RGB）
- **アノテーション**: CVAT XML形式
  - 眼瞼・涙丘: ポリゴン形式（セグメンテーション）
  - 虹彩・瞳孔: 回転バウンディングボックス（OBB）
- **患者ID抽出**: ファイル名のベースネーム先頭の整数（例: `1-2014...jpg` → patient_id=1）
- **使用データ**: CVAT image id 0-1999 のフレームのみ

## 主な評価指標

- **Dice係数**（クラス別）
- **推論速度**（ms/画像）
- **5-fold交差検証**による統計的比較

## 必要なPythonパッケージ

主要なパッケージ:
- PyTorch 2.9.0
- TorchVision 0.24.0
- NumPy 2.3.4
- Jupyter

詳細は `requirements.txt` を参照してください。

## 注意事項

- **このプロジェクトはGPU必須です**：train.ipynbの冒頭でGPUの確認を行い、利用できない場合はエラーを出力します
- 現在の環境：NVIDIA GeForce RTX 3080 Ti Laptop GPU（CUDA 12.4）
- PyTorch 2.6.0+cu124（GPU版）を使用
- データは `Images/images/` ディレクトリに配置してください
- アノテーションファイルはCVAT形式のXMLです

## ライセンス

（プロジェクト固有のライセンス情報を記載してください）

