# Jupyter Notebook (.ipynb) 効率的な修正方法

## 背景

Jupyter Notebookファイル（.ipynb）はJSON形式で保存されており、直接編集する際にいくつかの課題があります。

## 課題

1. **edit_notebookツールの制約**
   - 文字列の完全一致が必要
   - インデントや改行の微妙な違いでマッチング失敗
   - エスケープ文字の扱いが複雑

2. **search_replaceツールの制約**
   - .ipynbファイルには使用不可（専用ツール使用が必要）

3. **JSONの複雑な構造**
   - セルごとに`source`が配列形式
   - 各行が個別の文字列要素
   - メタデータや出力情報も含まれる

## 推奨される効率的な修正方法

### 方法1: Pythonスクリプトによる直接JSON編集（最も確実）

```python
import json

# ノートブックを読み込み
with open('notebook.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

# セルを検索して修正
for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code':
        source = cell['source']
        
        # 特定の文字列を含む行を検索して修正
        new_source = []
        for line in source:
            if '修正対象の文字列' in line:
                # 行を置換
                new_source.append(line.replace('古い値', '新しい値'))
            else:
                new_source.append(line)
        
        cell['source'] = new_source

# 保存
with open('notebook.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
```

**利点:**
- 確実に修正可能
- 複雑な条件分岐も対応可能
- 複数セルの一括修正も容易
- バックアップが簡単（元ファイルを自動保存可）

**欠点:**
- 一時的なスクリプトファイルが必要
- JSON構造の理解が必要

### 方法2: grepで検索 → セル番号特定 → edit_notebook

```bash
# 1. 該当箇所を含むセルを特定
grep -n "検索文字列" notebook.ipynb

# 2. execution_countを確認
grep -B5 "検索文字列" notebook.ipynb | grep execution_count

# 3. edit_notebookツールで修正（ただし文字列の完全一致が必要）
```

**利点:**
- 既存ツールの組み合わせ
- セル番号が明確

**欠点:**
- 文字列マッチングが厳密
- 複数箇所の修正には不向き

### 方法3: read_file → 全体構造把握 → Pythonスクリプト

今回採用した方法：

1. `grep`で修正箇所を特定
2. `read_file`で該当セル周辺を確認
3. Pythonスクリプトを作成してJSON編集
4. スクリプト実行後、一時ファイル削除

## 実例: train3.ipynbの修正

### 問題
```python
pos_w = torch.tensor([3.0, 3.0, 3.0], device=device)
criterion2 = EdgeBCELossWithNHWC(pos_weight=pos_w)
```
↓
```python
# pos_weightをNoneに設定（次元の問題を回避）
criterion2 = EdgeBCELossWithNHWC(pos_weight=None)
```

### 解決手順

```python
# fix_notebook.py
import json

with open('train3.ipynb', 'r', encoding='utf-8') as f:
    notebook = json.load(f)

for i, cell in enumerate(notebook['cells']):
    if cell['cell_type'] == 'code' and cell.get('execution_count') == 10:
        source = cell['source']
        new_source = []
        skip_next = False
        
        for line in source:
            if 'pos_w = torch.tensor' in line:
                new_source.append("    # pos_weightをNoneに設定（次元の問題を回避）\n")
                new_source.append("    criterion2 = EdgeBCELossWithNHWC(pos_weight=None)\n")
                skip_next = True
            elif skip_next and 'criterion2 = EdgeBCELossWithNHWC' in line:
                skip_next = False
            else:
                new_source.append(line)
        
        cell['source'] = new_source
        print(f"セル{i}を修正しました")
        break

with open('train3.ipynb', 'w', encoding='utf-8') as f:
    json.dump(notebook, f, ensure_ascii=False, indent=1)
```

実行:
```bash
python fix_notebook.py
rm fix_notebook.py  # 修正後は削除
```

## ベストプラクティス

1. **必ず検証**
   - 修正前に`grep`で該当箇所を確認
   - 修正後も`grep`で結果を確認

2. **セル特定の優先順位**
   ```python
   # 優先度1: execution_count（最も確実）
   if cell.get('execution_count') == 10:
   
   # 優先度2: セル内の特徴的な文字列
   if any('特徴的な関数名' in line for line in cell['source']):
   
   # 優先度3: セルのインデックス（変更に弱い）
   if i == 9:  # 0-indexed
   ```

3. **安全な修正**
   ```python
   # バックアップを作成
   import shutil
   shutil.copy('notebook.ipynb', 'notebook.ipynb.bak')
   
   # 修正処理
   # ...
   
   # 検証後にバックアップ削除
   os.remove('notebook.ipynb.bak')
   ```

4. **JSON構造の保持**
   - `indent=1`を使用（可読性と差分管理のバランス）
   - `ensure_ascii=False`で日本語を保持
   - 出力（`outputs`）やメタデータは触らない

## 避けるべき方法

❌ **手動でJSONを編集**
- 構文エラーのリスク
- カンマやブラケットのミス

❌ **正規表現による複雑な置換**
- エスケープが複雑
- 誤マッチのリスク

❌ **全文一致による edit_notebook**
- セル全体をコピペする必要がある
- インデントの違いで失敗しやすい

## まとめ

Jupyterノートブックの修正には：
1. **小規模修正**: `edit_notebook`ツール（文字列完全一致が可能な場合）
2. **複雑な修正**: Pythonスクリプト + JSON直接編集（推奨）
3. **大規模修正**: 複数セルの場合は必ずPythonスクリプト

**今回の教訓:**
- edit_notebookの文字列マッチングは厳密すぎる
- JSON直接編集が最も確実で柔軟
- 一時スクリプトの作成・削除は効率的なワークフロー

## .gitignoreとgit追跡の重要な注意点

### 問題：.gitignoreが効かないケース

.gitignoreに`model/`や`Images/`を追加しても、**既にgit追跡されているファイルには効果がありません**。

```bash
# .gitignoreの内容
model/
Images/
```

しかし`git status`で表示される → **既に追跡されているため**

### 根本的な解決方法

```bash
# 1. git追跡から削除（ファイル自体は残す）
git rm -r --cached model/
git rm -r --cached Images/

# 2. コミット
git commit -m "Remove model/ and Images/ from git tracking"

# 3. プッシュ
git push

# 以降、.gitignoreが正常に機能する
```

### なぜこれが必要か

- `.gitignore`：**新規ファイルの追跡を防ぐ**（予防）
- `git rm --cached`：**既存の追跡を解除する**（治療）

両方が必要です。

### GitHub容量制限の問題

- 単一ファイル上限：100MB
- model/method3_fold0_best.pth：214.47MB → **プッシュ不可**
- 解決策：
  1. `git rm --cached model/` で追跡解除
  2. Git LFS（Large File Storage）導入（別途設定必要）
  3. または model/ を完全に除外

### 推奨対応（今回のケース）

```bash
# 1. 大容量ファイルを追跡から除外
git rm -r --cached model/
git rm -r --cached Images/

# 2. .gitignoreが既に設定済みであることを確認
# （model/とImages/が既に記載されている）

# 3. コミット
git commit -m "Remove large files from git tracking (.gitignore already set)"

# 4. プッシュ（以降は軽量ファイルのみ）
git push
```

### 重要な教訓

1. **プロジェクト開始時に.gitignoreを設定**すること
2. **大容量ファイルは絶対にコミットしない**（後から除外が大変）
3. **既に追跡されているファイルは`git rm --cached`で除外が必要**

## Method2のエッジギャップ分析（2025-11-03調査結果）

### 調査の背景

Method2では`thickness=3px`でエッジを生成して学習していますが、推論時のギャップ発生が懸念されました。1/3/5/7pxでのギャップ発生率を比較調査しました。

### 当初の誤った結果（スケルトン化ベース）

スケルトン化（`skimage.morphology.skeletonize`）を用いた端点検出で以下の結果が出ました：

| 厚み | ギャップ発生 | 割合 |
|------|-------------|------|
| 1px  | 6件         | 0.3% |
| 3px  | 22件        | 1.1% |
| 5px  | 44件        | 2.2% |
| 7px  | 50件        | 2.5% |

**問題点**：厚いエッジほどギャップが増える矛盾した結果

### 根本原因の発見

詳細調査により、以下が判明：

1. **元のGTマスクは99.8%が完全に閉じた輪郭**（問題なし）
2. **`drawContours(thickness=3/5/7)`で生成されたエッジも100%完全につながっている**
3. **しかし`skeletonize()`処理が閉じたリング状エッジに対して意図しない端点を生成**
   - 特に「水平に長く垂直に薄い」形状（眼角部など）で不安定
   - 本来は端点0個（閉じたループ）のはずが、端点2個を誤生成
   - 結果として357px等の「架空のギャップ」を検出

### 正しい判定方法と結果

**スケルトン化を使わず、エッジの連結性を直接チェック**する方法に変更：

```python
def has_gap_proper(edge_bin):
    """エッジピクセルの隣接数を直接カウント。
    端点（neighbor==1）が2個以上あればギャップと判定。
    """
    kernel = np.ones((3, 3), dtype=np.uint8)
    neighbor_count = cv2.filter2D(edge_u8, ...) - edge_u8
    endpoints = (edge_u8 > 0) & (neighbor_count == 1)
    # 画像境界を除外して内部の端点のみカウント
    return np.count_nonzero(interior_endpoints) >= 2
```

**修正後の正しい結果（全1992件）**：

| 厚み | ギャップ発生 | 割合 |
|------|-------------|------|
| **1px**  | **3件**     | **0.2%** |
| **3px**  | **0件**     | **0.0%** ← 完璧！|
| **5px**  | 0件         | 0.0% |
| **7px**  | 0件         | 0.0% |

### 結論

1. **元のGTマスク品質は極めて高い**（99.8%が閉じた輪郭）
2. **thickness=3pxの太線化は完璧に機能**：残り0.2%の真のギャップも完全解消
3. **5px/7pxは不要**：3pxで既に完璧、太すぎると境界精度が低下するだけ
4. **スケルトン化は不適切**：ギャップ判定には使うべきでない
   - 閉じたリング状の形状で誤検出が多発
   - 特に「水平に長く垂直に薄い」領域で不安定

### 推論時の処理

- **Eyelid**: エッジ→`bin_edge_to_filled()`で面化
  - 25×25カーネル×6回のモルフォロジカルクロージングで最大150px程度の欠損まで補完
  - 3pxの太線化とは独立した、より強力なギャップ補正
- **Iris/Pupil**: エッジ→`cv2.fitEllipse()`で楕円フィット
  - エッジが綺麗なのでRANSAC不要、最小二乗法で高速・高精度

### 教訓

- **ギャップ判定にスケルトン化を使用しない**
- **エッジの連結性は直接（隣接ピクセル数）チェック**
- **形状の特性（細長い、くびれがある等）を考慮したアルゴリズム選択が重要**