"""
Generate PDF summary for 130th Japanese Ophthalmological Society presentation.
Compares Method 1 (ellipse regression), Method 6 (visible-only seg), Method 4 (amodal seg).
"""
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
from fpdf import FPDF
from pathlib import Path
import os

# ===== Setup =====
OUT_DIR = Path("Experimental_record")
OUT_DIR.mkdir(exist_ok=True)
OUT_PDF = OUT_DIR / "130回日本眼科学会_v2.pdf"

# Japanese font
JP_FONT_PATH = "C:/Windows/Fonts/meiryo.ttc"
JP_FONT_BOLD_PATH = "C:/Windows/Fonts/meiryob.ttc"
fp = fm.FontProperties(fname=JP_FONT_PATH)
plt.rcParams['font.family'] = fp.get_name()
plt.rcParams['axes.unicode_minus'] = False

# ===== Load Data =====
df1_fold = pd.read_csv("results/cv_eval_method1_20260325_093809.csv")
df6_fold = pd.read_csv("results/cv_eval_method6_20260325_093809.csv")
df4_img = pd.read_csv("results/cv_method4_full_vs_exposed_perimage_20260325_093809.csv")
df1_img = pd.read_csv("results/cv_method1_perimage_20260203_022055.csv")
df6_img = pd.read_csv("results/cv_method6_visible_boundary_perimage_20260325_093809.csv")

# Method 4 per-fold stats by mode
def fold_stats(df, mode):
    sub = df[df['mode'] == mode] if 'mode' in df.columns else df
    return sub.groupby('fold')[['eyelid', 'iris', 'pupil', 'mean']].mean()

m4_raw = fold_stats(df4_img, 'raw')
m4_outer = fold_stats(df4_img, 'outerarc')
m4_full = fold_stats(df4_img, 'fullmax')

# ===== Generate comparison bar chart =====
fig, axes = plt.subplots(1, 2, figsize=(12, 4.5), gridspec_kw={'width_ratios': [3, 2]})

# --- Chart 1: 3 methods comparison ---
ax = axes[0]
methods = ['Method 1\n(楕円回帰)', 'Method 6\n(可視部seg+\n境界楕円近似)', 'Method 4\n(Amodal seg+\n全体楕円近似)']
structs = ['Eyelid', 'Iris', 'Pupil']
colors = ['#4472C4', '#ED7D31', '#70AD47']

m1_means = [df1_fold['eyelid'].mean(), df1_fold['iris'].mean(), df1_fold['pupil'].mean()]
m1_stds = [df1_fold['eyelid'].std(), df1_fold['iris'].std(), df1_fold['pupil'].std()]
m6_means = [df6_fold['eyelid'].mean(), df6_fold['iris'].mean(), df6_fold['pupil'].mean()]
m6_stds = [df6_fold['eyelid'].std(), df6_fold['iris'].std(), df6_fold['pupil'].std()]
m4_means = [m4_full['eyelid'].mean(), m4_full['iris'].mean(), m4_full['pupil'].mean()]
m4_stds = [m4_full['eyelid'].std(), m4_full['iris'].std(), m4_full['pupil'].std()]

x = np.arange(3)
w = 0.25
bars1 = ax.bar(x - w, m1_means, w, yerr=m1_stds, label='Method 1', color=colors[0], capsize=3)
bars2 = ax.bar(x, m6_means, w, yerr=m6_stds, label='Method 6', color=colors[1], capsize=3)
bars3 = ax.bar(x + w, m4_means, w, yerr=m4_stds, label='Method 4', color=colors[2], capsize=3)

ax.set_ylabel('Dice Coefficient', fontsize=11)
ax.set_title('手法間比較 (5-fold CV)', fontproperties=fp, fontsize=13)
ax.set_xticks(x)
ax.set_xticklabels(structs, fontsize=10)
ax.set_ylim(0.65, 1.02)
ax.legend(fontsize=9, loc='lower left')
ax.grid(axis='y', alpha=0.3)

# Add value labels
for bars in [bars1, bars2, bars3]:
    for bar in bars:
        h = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., h + 0.005,
                f'{h:.3f}', ha='center', va='bottom', fontsize=7)

# --- Chart 2: Method4 post-processing ablation ---
ax2 = axes[1]
modes = ['楕円近似なし\n(Raw)', '可視部境界\n(OuterArc)', '全体境界\n(FullMax)']
mode_colors = ['#A5A5A5', '#FFC000', '#5B9BD5']

for i, (mode_data, label) in enumerate([(m4_raw, 'Raw'), (m4_outer, 'OuterArc'), (m4_full, 'FullMax')]):
    vals = [mode_data['iris'].mean(), mode_data['pupil'].mean()]
    errs = [mode_data['iris'].std(), mode_data['pupil'].std()]
    ax2.bar(np.arange(2) + i*0.25 - 0.25, vals, 0.25, yerr=errs,
            label=label, color=mode_colors[i], capsize=3)
    for j, (v, e) in enumerate(zip(vals, errs)):
        ax2.text(j + i*0.25 - 0.25, v + 0.005, f'{v:.3f}',
                ha='center', va='bottom', fontsize=7)

ax2.set_ylabel('Dice Coefficient', fontsize=11)
ax2.set_title('Method 4: 楕円近似の効果', fontproperties=fp, fontsize=13)
ax2.set_xticks(np.arange(2))
ax2.set_xticklabels(['Iris', 'Pupil'], fontsize=10)
ax2.set_ylim(0.85, 1.02)
ax2.legend(fontsize=9)
ax2.grid(axis='y', alpha=0.3)

plt.tight_layout()
chart_path = OUT_DIR / "comparison_chart.png"
plt.savefig(chart_path, dpi=200, bbox_inches='tight')
plt.close()
print(f"Chart saved: {chart_path}")

# ===== Generate PDF =====
class JaPDF(FPDF):
    def __init__(self):
        super().__init__()
        self.add_font('Meiryo', '', JP_FONT_PATH)
        self.add_font('Meiryo', 'B', JP_FONT_BOLD_PATH)

    def header_text(self, text, size=16):
        self.set_font('Meiryo', 'B', size)
        self.cell(0, 10, text, new_x="LMARGIN", new_y="NEXT", align='C')
        self.ln(2)

    def section_title(self, text, size=13):
        self.ln(3)
        self.set_font('Meiryo', 'B', size)
        self.set_fill_color(230, 240, 250)
        self.cell(0, 8, text, new_x="LMARGIN", new_y="NEXT", fill=True)
        self.ln(2)

    def body_text(self, text, size=9):
        self.set_font('Meiryo', '', size)
        self.multi_cell(0, 5, text)
        self.ln(1)

    def table(self, headers, rows, col_widths=None, header_color=(70, 114, 196)):
        if col_widths is None:
            col_widths = [self.epw / len(headers)] * len(headers)

        # Header
        self.set_font('Meiryo', 'B', 8)
        self.set_fill_color(*header_color)
        self.set_text_color(255, 255, 255)
        for i, h in enumerate(headers):
            self.cell(col_widths[i], 7, h, border=1, align='C', fill=True)
        self.ln()

        # Rows
        self.set_font('Meiryo', '', 8)
        self.set_text_color(0, 0, 0)
        for j, row in enumerate(rows):
            fill = j % 2 == 1
            if fill:
                self.set_fill_color(240, 245, 250)
            for i, cell in enumerate(row):
                self.cell(col_widths[i], 6, str(cell), border=1,
                         align='C' if i > 0 else 'L', fill=fill)
            self.ln()

pdf = JaPDF()
pdf.set_auto_page_break(auto=True, margin=15)

# ===== Page 1: Title & Methods =====
pdf.add_page()
pdf.header_text("眼画像セグメンテーション手法比較", 16)
pdf.header_text("第130回 日本眼科学会総会 発表資料", 11)
pdf.ln(3)

pdf.section_title("1. 研究概要")
pdf.body_text(
    "前眼部画像における眼瞼・虹彩・瞳孔の自動セグメンテーション精度を3手法で比較した。\n"
    "全手法で共通のU-Net (VGG16-BN encoder) アーキテクチャを使用し、\n"
    "出力設計（楕円回帰 vs 可視部セグメンテーション vs アモーダルセグメンテーション）の\n"
    "影響を検証した。"
)
pdf.body_text(
    "データセット: 122名・1,992画像 (512×512 px)\n"
    "評価: 患者単位5-fold交差検証 (GroupKFold)、最大300エポック (early stopping 30)\n"
    "評価指標: Dice similarity coefficient (眼瞼・虹彩・瞳孔)"
)

pdf.section_title("2. 手法の概要")
headers = ["手法", "出力設計", "クラス数", "損失関数", "後処理"]
rows = [
    ["Method 1", "眼瞼seg + 虹彩/瞳孔\n楕円パラメータ回帰", "眼瞼2ch\n+ 5param×2", "BCE + Smooth-L1", "パラメータ→楕円マスク"],
    ["Method 6", "可視部のみ\nセマンティックseg", "4クラス\n(背景,結膜,\n可視虹彩,可視瞳孔)", "CE + 0.5×Dice", "境界点から\n楕円近似 (limbus)"],
    ["Method 4", "アモーダル\n6クラスseg\n(隠蔽部含む)", "6クラス\n(背景,結膜,\n可視/隠蔽 虹彩/瞳孔)", "CE + 0.5×Dice", "全体マスクから\n楕円近似 (FullMax)"],
]
col_widths = [22, 40, 28, 30, 35]

# Manually render the method table with wrapping
pdf.set_font('Meiryo', 'B', 8)
pdf.set_fill_color(70, 114, 196)
pdf.set_text_color(255, 255, 255)
for i, h in enumerate(headers):
    pdf.cell(col_widths[i], 7, h, border=1, align='C', fill=True)
pdf.ln()

pdf.set_text_color(0, 0, 0)
for j, row in enumerate(rows):
    x_start = pdf.get_x()
    y_start = pdf.get_y()
    max_h = 0
    # Calculate row height
    for i, cell in enumerate(row):
        lines = cell.split('\n')
        h = max(len(lines) * 5, 5)
        max_h = max(max_h, h)

    fill = j % 2 == 1
    if fill:
        pdf.set_fill_color(240, 245, 250)

    pdf.set_font('Meiryo', '', 7)
    for i, cell in enumerate(row):
        x = x_start + sum(col_widths[:i])
        pdf.set_xy(x, y_start)
        # Draw cell border and fill
        pdf.rect(x, y_start, col_widths[i], max_h)
        if fill:
            pdf.rect(x, y_start, col_widths[i], max_h, 'F')
            pdf.rect(x, y_start, col_widths[i], max_h, 'D')
        # Write text
        lines = cell.split('\n')
        y_text = y_start + (max_h - len(lines) * 4.5) / 2
        for line in lines:
            pdf.set_xy(x + 1, y_text)
            pdf.cell(col_widths[i] - 2, 4.5, line, align='C')
            y_text += 4.5

    pdf.set_y(y_start + max_h)

pdf.ln(3)
pdf.body_text(
    "Method 1: 虹彩/瞳孔の楕円パラメータ(中心x,y, 長軸, 短軸, 回転角)を直接回帰する。\n"
    "Method 6: 可視部(眼瞼で隠れていない部分)のみを4クラスseg → 虹彩-結膜境界(limbus)点のみから楕円近似。\n"
    "Method 4: 隠蔽部(occluded iris/pupil)も含めた6クラスamodal seg → 全マスク境界から楕円近似。"
)

# ===== Page 2: Results =====
pdf.add_page()
pdf.section_title("3. 主要結果: 手法間比較 (5-fold CV, Mean ± SD)")

# Main comparison table
headers2 = ["手法", "Eyelid Dice", "Iris Dice", "Pupil Dice", "Mean Dice"]
m1m = df1_fold[['eyelid','iris','pupil','mean']].mean()
m1s = df1_fold[['eyelid','iris','pupil','mean']].std()
m6m = df6_fold[['eyelid','iris','pupil','mean']].mean()
m6s = df6_fold[['eyelid','iris','pupil','mean']].std()
m4m = m4_full.mean()
m4s = m4_full.std()

def fmt(m, s):
    return f"{m:.4f} ± {s:.4f}"

rows2 = [
    ["Method 1 (楕円回帰)", fmt(m1m['eyelid'],m1s['eyelid']), fmt(m1m['iris'],m1s['iris']),
     fmt(m1m['pupil'],m1s['pupil']), fmt(m1m['mean'],m1s['mean'])],
    ["Method 6 (可視部seg)", fmt(m6m['eyelid'],m6s['eyelid']), fmt(m6m['iris'],m6s['iris']),
     fmt(m6m['pupil'],m6s['pupil']), fmt(m6m['mean'],m6s['mean'])],
    ["Method 4 (Amodal seg)", fmt(m4m['eyelid'],m4s['eyelid']), fmt(m4m['iris'],m4s['iris']),
     fmt(m4m['pupil'],m4s['pupil']), fmt(m4m['mean'],m4s['mean'])],
]
cw2 = [38, 32, 32, 32, 32]
pdf.table(headers2, rows2, cw2)

pdf.ln(2)

# Statistical tests table
pdf.section_title("4. 統計解析 (被験者水準, n=122)")
pdf.body_text(
    "Friedman検定 (omnibus): χ²=148.2, p=6.4×10⁻³³ → 3手法間に有意差あり\n"
    "Post-hoc: Wilcoxon符号順位検定 + Holm補正 (3比較)"
)
headers3 = ["比較 (A vs B)", "ΔMean Dice", "ΔIris", "ΔPupil", "p_Holm (Mean)"]
rows3 = [
    ["Method 4 vs Method 1", "+0.0746", "+0.0523", "+0.1762", "4.8×10⁻¹⁹"],
    ["Method 6 vs Method 1", "+0.0675", "+0.0337", "+0.1710", "2.2×10⁻¹⁸"],
    ["Method 4 vs Method 6", "+0.0071", "+0.0186", "+0.0053 (ns)", "1.5×10⁻⁴"],
]
cw3 = [40, 28, 28, 32, 32]
pdf.table(headers3, rows3, cw3)

pdf.ln(1)
pdf.body_text(
    "全比較でHolm補正後も有意。Method 4 (amodal) と Method 6 (可視部) の差は小さいが有意 (p=1.5×10⁻⁴)。\n"
    "Method 1 (楕円回帰) は特に瞳孔で大幅に劣る (Dice 0.723 vs 0.927-0.916)。\n"
    "→ 教師ラベル自体が楕円近似値であり近似誤差を含む。回帰では誤差が5パラメータに\n"
    "  集中するのに対し、segでは数千画素に分散され希釈される (詳細は考察参照)。"
)

# Chart
pdf.ln(2)
pdf.image(str(chart_path), x=10, w=190)

# ===== Page 3: Method 4 Ablation & Discussion =====
pdf.add_page()
pdf.section_title("5. Method 4 後処理アブレーション: 楕円近似の効果")
pdf.body_text(
    "6クラスamodal segmentation (Method 4) の出力に対して、3種類の楕円近似を比較。\n"
    "「可視部境界 (OuterArc)」は Method 6 と同様に、眼瞼で切断された部分を除外した境界点のみから楕円近似する。\n"
    "「全体境界 (FullMax)」はネットワークが予測した隠蔽部も含む全マスク境界から楕円近似する。"
)

headers4 = ["後処理モード", "Eyelid Dice", "Iris Dice", "Pupil Dice", "Mean Dice"]
rows4 = [
    ["楕円近似なし (Raw)", fmt(m4_raw['eyelid'].mean(), m4_raw['eyelid'].std()),
     fmt(m4_raw['iris'].mean(), m4_raw['iris'].std()),
     fmt(m4_raw['pupil'].mean(), m4_raw['pupil'].std()),
     fmt(m4_raw['mean'].mean(), m4_raw['mean'].std())],
    ["可視部境界 (OuterArc)", fmt(m4_outer['eyelid'].mean(), m4_outer['eyelid'].std()),
     fmt(m4_outer['iris'].mean(), m4_outer['iris'].std()),
     fmt(m4_outer['pupil'].mean(), m4_outer['pupil'].std()),
     fmt(m4_outer['mean'].mean(), m4_outer['mean'].std())],
    ["全体境界 (FullMax)", fmt(m4_full['eyelid'].mean(), m4_full['eyelid'].std()),
     fmt(m4_full['iris'].mean(), m4_full['iris'].std()),
     fmt(m4_full['pupil'].mean(), m4_full['pupil'].std()),
     fmt(m4_full['mean'].mean(), m4_full['mean'].std())],
]
cw4 = [38, 32, 32, 32, 32]
pdf.table(headers4, rows4, cw4)

pdf.ln(2)

# Post-processing stats
pdf.set_font('Meiryo', 'B', 9)
pdf.cell(0, 6, "後処理間の統計比較 (Friedman + Wilcoxon/Holm, n=122)", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
pdf.body_text("Friedman検定: χ²=155.0, p=2.2×10⁻³⁴ → 3モード間に有意差あり")
headers5 = ["比較", "ΔMean Dice", "p_Holm"]
rows5 = [
    ["FullMax vs Raw", "+0.0216", "2.8×10⁻²¹"],
    ["OuterArc vs Raw", "+0.0116", "1.2×10⁻¹⁰"],
    ["FullMax vs OuterArc", "+0.0100", "8.0×10⁻¹²"],
]
cw5 = [50, 40, 50]
pdf.table(headers5, rows5, cw5)

pdf.ln(3)
pdf.body_text(
    "全ての後処理モード間で有意差あり。\n"
    "・楕円近似は特に虹彩で大きな効果 (Raw 0.893 → FullMax 0.960, +6.7pp)\n"
    "  → 眼瞼による不規則な切断を楕円形状で正則化する効果\n"
    "・瞳孔はもともと円形に近く、楕円近似の効果は小さい (±0.1pp)\n"
    "・全体境界 (FullMax) は可視部境界 (OuterArc) より有意に優れる (+1.0pp, p<10⁻¹²)\n"
    "  → ネットワークが予測した隠蔽部情報を楕円近似に活用する価値がある"
)

pdf.section_title("6. 考察とまとめ")
pdf.body_text(
    "【主要知見】\n"
    "1. 領域セグメンテーション手法 (Method 4, 6) は楕円パラメータ回帰 (Method 1) より\n"
    "   大幅に優れる (Mean Dice +6.7~7.5pp, p<10⁻¹⁸)。特に瞳孔で顕著 (+17pp)。\n"
    "   原因: 虹彩・瞳孔の教師ラベルは楕円近似であり、近似誤差を含む。\n"
    "   回帰では誤差が5パラメータに集中し学習を不安定にする。\n"
    "   一方segではラベル誤差が境界の数千画素に分散され、個々の画素への影響は微小。\n"
    "   また、パラメータ空間のMSEはマスク空間のDiceと非線形な対応関係にあり\n"
    "   (例: 回転角の小さな誤差でも、小さい楕円ほどマスク変位が相対的に大きい)、\n"
    "   小構造ほど学習が不安定になりやすい。\n\n"
    "2. Amodal segmentation (Method 4) は可視部のみのsegmentation (Method 6) と\n"
    "   同等~やや優れる (Mean Dice +0.7pp, p=1.5×10⁻⁴)。\n"
    "   虹彩では有意差あり (+1.9pp)、瞳孔では差なし (+0.5pp, ns)。\n\n"
    "3. 隠蔽部予測の最大の価値は楕円近似の質の向上にある:\n"
    "   同じMethod 4出力でも、全体境界 (FullMax) は可視部境界 (OuterArc) より\n"
    "   有意に優れる (+1.0pp, p<10⁻¹²)。\n"
    "   この効果はアーキテクチャ非依存 (U-Net, YOLO11, SegFormer で確認済み)。\n\n"
    "4. 損失関数の最適化も重要: CE+Dice併用がDice単独より有意に優れる。\n"
    "   逆周波数重み付けやFocal Lossは改善せず (Diceが暗黙的にバランスを取るため)。"
)

pdf.section_title("7. 補足: アーキテクチャ横断検証")
pdf.body_text(
    "FullMax > OuterArc の優位性を3アーキテクチャで確認:\n"
    "・U-Net (Method 4): FullMax 0.939 vs OuterArc 0.931 (Δ+0.008, p<5×10⁻⁵)\n"
    "・YOLO11l-seg:     FullMax 0.950 vs OuterArc 0.945 (Δ+0.005, p<5×10⁻⁵)\n"
    "・SegFormer-B2:    FullMax 0.956 vs OuterArc 0.944 (Δ+0.012, p<5×10⁻⁵)\n"
    "→ 隠蔽部予測を楕円近似に活用する利点はモデル構造に依存しない。"
)

# ===== Page 4: Per-fold details =====
pdf.add_page()
pdf.section_title("8. 補足: Fold別詳細結果")

# Method 1
pdf.set_font('Meiryo', 'B', 9)
pdf.cell(0, 6, "Method 1 (楕円パラメータ回帰)", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
h_fold = ["Fold", "Eyelid", "Iris", "Pupil", "Mean"]
r_m1 = []
for _, r in df1_fold.iterrows():
    r_m1.append([str(int(r['fold'])), f"{r['eyelid']:.4f}", f"{r['iris']:.4f}",
                 f"{r['pupil']:.4f}", f"{r['mean']:.4f}"])
r_m1.append(["Mean±SD", fmt(m1m['eyelid'],m1s['eyelid']), fmt(m1m['iris'],m1s['iris']),
              fmt(m1m['pupil'],m1s['pupil']), fmt(m1m['mean'],m1s['mean'])])
cw_fold = [20, 35, 35, 35, 35]
pdf.table(h_fold, r_m1, cw_fold)

pdf.ln(3)

# Method 6
pdf.set_font('Meiryo', 'B', 9)
pdf.cell(0, 6, "Method 6 (可視部seg + 境界楕円近似)", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
r_m6 = []
for _, r in df6_fold.iterrows():
    r_m6.append([str(int(r['fold'])), f"{r['eyelid']:.4f}", f"{r['iris']:.4f}",
                 f"{r['pupil']:.4f}", f"{r['mean']:.4f}"])
r_m6.append(["Mean±SD", fmt(m6m['eyelid'],m6s['eyelid']), fmt(m6m['iris'],m6s['iris']),
              fmt(m6m['pupil'],m6s['pupil']), fmt(m6m['mean'],m6s['mean'])])
pdf.table(h_fold, r_m6, cw_fold)

pdf.ln(3)

# Method 4 (FullMax)
pdf.set_font('Meiryo', 'B', 9)
pdf.cell(0, 6, "Method 4 (Amodal seg, FullMax楕円近似)", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
r_m4 = []
for fold_idx in range(5):
    r = m4_full.loc[fold_idx]
    r_m4.append([str(fold_idx), f"{r['eyelid']:.4f}", f"{r['iris']:.4f}",
                 f"{r['pupil']:.4f}", f"{r['mean']:.4f}"])
r_m4.append(["Mean±SD", fmt(m4m['eyelid'],m4s['eyelid']), fmt(m4m['iris'],m4s['iris']),
              fmt(m4m['pupil'],m4s['pupil']), fmt(m4m['mean'],m4s['mean'])])
pdf.table(h_fold, r_m4, cw_fold)

pdf.ln(3)

# Method 4 per-fold by post-processing
pdf.set_font('Meiryo', 'B', 9)
pdf.cell(0, 6, "Method 4: 後処理別 Mean Dice (fold別)", new_x="LMARGIN", new_y="NEXT")
pdf.ln(1)
h_pp = ["Fold", "Raw", "OuterArc", "FullMax", "Δ(Full-Outer)"]
r_pp = []
for fold_idx in range(5):
    raw_v = m4_raw.loc[fold_idx, 'mean']
    out_v = m4_outer.loc[fold_idx, 'mean']
    full_v = m4_full.loc[fold_idx, 'mean']
    r_pp.append([str(fold_idx), f"{raw_v:.4f}", f"{out_v:.4f}", f"{full_v:.4f}",
                 f"+{full_v-out_v:.4f}"])
r_pp.append(["Mean±SD",
              fmt(m4_raw['mean'].mean(), m4_raw['mean'].std()),
              fmt(m4_outer['mean'].mean(), m4_outer['mean'].std()),
              fmt(m4_full['mean'].mean(), m4_full['mean'].std()),
              f"+{m4_full['mean'].mean()-m4_outer['mean'].mean():.4f}"])
cw_pp = [20, 35, 35, 35, 35]
pdf.table(h_pp, r_pp, cw_pp)

pdf.ln(5)
pdf.set_font('Meiryo', '', 8)
pdf.set_text_color(128, 128, 128)
pdf.cell(0, 5, "生成日: 2026-03-26 | データソース: 5-fold CV (300 epochs, early stopping)",
         new_x="LMARGIN", new_y="NEXT", align='C')
pdf.cell(0, 5, "統計検定: Friedman検定 + post-hoc Wilcoxon符号順位検定/Holm補正 (n=122)",
         new_x="LMARGIN", new_y="NEXT", align='C')

# Save
pdf.output(str(OUT_PDF))
print(f"\nPDF saved: {OUT_PDF}")

# Cleanup
os.remove(str(chart_path))
print("Chart temp file cleaned up.")
