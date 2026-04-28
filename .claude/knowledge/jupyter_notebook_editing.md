## Jupyter Notebook (.ipynb) の編集コツ（2026-02-20）

### 状況
crossvalidation.ipynb（50k行、4.9MB）にMethod 4を追加する作業

### 問題 / 課題
1. NotebookEdit ツールは大きなノートブックでファイルサイズ超過エラーになる
2. 文字列置換で空白行のインデント（例: 8スペースの空行）が一致しない
3. 古い出力が残っているとノートブックの読み込み・実行が遅くなる

### 解決策 / コツ
1. **大きなnotebookはPythonスクリプトで直接JSON操作**が確実
   ```python
   with open('notebook.ipynb', 'r', encoding='utf-8') as f:
       nb = json.load(f)
   src = ''.join(nb['cells'][cell_idx]['source'])
   # 編集後
   lines = new_src.split('\n')
   nb['cells'][cell_idx]['source'] = [line + '\n' for line in lines]
   nb['cells'][cell_idx]['source'][-1] = nb['cells'][cell_idx]['source'][-1].rstrip('\n')
   ```
2. **置換前に `repr()` で実際の文字列を確認**する（空白行のインデントが見えない）
3. **変更後は必ず全セルの `ast.parse()` で構文チェック**する
4. **編集後に全出力をクリア**してファイルサイズを削減
5. **変数の定義順に注意**: セル内で後方定義の変数を前方参照に変えるとNameErrorになる

### 参考
- `Experimental_record/20260220.md` — 実際の作業記録
