"""Reproduz a figura principal e as tabelas do artigo a partir dos CSVs em resultados/.
Uso:  python gerar_figura.py
Saídas: figura_principal.pdf/.png, tabela_f1.tex, tabela_pareada.tex, resumo.txt
"""
import pandas as pd, numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path

R = Path(__file__).parent / "resultados"
df = pd.concat([pd.read_csv(R / "resultados_mlp.csv"), pd.read_csv(R / "resultados_cnn.csv")], ignore_index=True)

CFG = [("MLP_nenhum0", "MLP (treino limpo)"), ("CNN_nenhum0", "CNN, 0% ruído"),
       ("CNN_salt_pepper30", "CNN, 30% ruído"), ("CNN_salt_pepper60", "CNN, 60% ruído")]
TST = [("limpo", "limpo"), ("salt_pepper", "sal e pimenta"), ("gaussian", "gaussiano"),
       ("poisson", "Poisson"), ("rotation", "rotação")]

def ms(cfg, t, met="f1_macro"):
    v = df[(df.config == cfg) & (df.teste == t)].sort_values("semente")[met].values
    return v.mean(), v.std(ddof=1)

# ---- figura
fig, ax = plt.subplots(figsize=(6.4, 3.0))
w = 0.2; hatches = ["", "//", "xx", ".."]; cols = ["#999999", "#1f77b4", "#ff7f0e", "#2ca02c"]
for j, (c, lab) in enumerate(CFG):
    m = [ms(c, t)[0] for t, _ in TST]; s = [ms(c, t)[1] for t, _ in TST]
    ax.bar(np.arange(len(TST)) + (j - 1.5) * w, m, w, yerr=s, capsize=2, label=lab,
           color=cols[j], hatch=hatches[j], edgecolor="black", linewidth=0.5, error_kw=dict(lw=0.8))
ax.set_xticks(range(len(TST))); ax.set_xticklabels([l for _, l in TST], fontsize=8)
ax.set_xlabel("Condição do conjunto de teste (100% das imagens)", fontsize=8)
ax.set_ylabel("F1 macro (média ± desvio, 3 sementes)", fontsize=8)
ax.tick_params(axis="y", labelsize=8); ax.set_ylim(0, 0.45)
ax.legend(fontsize=7, ncol=2, frameon=False, loc="upper right")
plt.tight_layout(); plt.savefig("figura_principal.pdf"); plt.savefig("figura_principal.png", dpi=200)

# ---- tabela principal (negrito = maior média da coluna)
def melhor_claro(t):
    """Config com maior média, só se a folga sobre a 2ª superar o maior dos dois desvios."""
    r = sorted(((ms(c, t), c) for c, _ in CFG), key=lambda x: -x[0][0])
    (m1, s1), c1 = r[0]; (m2, s2), _ = r[1]
    return c1 if (m1 - m2) > max(s1, s2) else None
best = {t: melhor_claro(t) for t, _ in TST}
rows = []
for c, lab in CFG:
    cells = []
    for t, _ in TST:
        m, s = ms(c, t); txt = f"{m:.3f} $\\pm$ {s:.3f}"
        cells.append(f"\\textbf{{{txt}}}" if best[t] == c else txt)
    rows.append(f"{lab.replace(chr(37), chr(92)+chr(37))} & " + " & ".join(cells) + " \\\\")
HEAD = ("\\begin{tabular}{lccccc}\n\\toprule\n"
        "Treino $\\backslash$ Teste & limpo & sal e pimenta & gaussiano & Poisson & rota\u00e7\u00e3o \\\\\n\\midrule\n")
open("tabela_f1.tex", "w").write(HEAD + "\n".join(rows) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- diferenças pareadas por semente (CNN com ruído - CNN limpa)
def vec(cfg, t): return df[(df.config == cfg) & (df.teste == t)].sort_values("semente").f1_macro.values
sem = sorted(df.semente.unique()); prow = []
for c, lab in [("CNN_salt_pepper30", "30\\%"), ("CNN_salt_pepper60", "60\\%")]:
    for t, tl in [("limpo", "teste limpo"), ("salt_pepper", "teste sal e pimenta")]:
        d = vec(c, t) - vec("CNN_nenhum0", t)
        prow.append(f"{lab} & {tl} & " + " & ".join(f"{x:+.3f}" for x in d) + f" & {d.mean():+.3f} \\\\")
HEAD2 = ("\\begin{tabular}{llcccc}\n\\toprule\n"
         "Ru\u00eddo no treino & Teste & sem.\\ 48 & sem.\\ 49 & sem.\\ 50 & m\u00e9dia \\\\\n\\midrule\n")
open("tabela_pareada.tex", "w").write(HEAD2 + "\n".join(prow) + "\n\\bottomrule\n\\end{tabular}\n")

# ---- resumo textual (conferência)
with open("resumo.txt", "w") as f:
    for c, lab in CFG:
        for t, tl in TST:
            m, s = ms(c, t); a, sa = ms(c, t, "acc")
            f.write(f"{c:20s} {tl:14s} F1 {m:.4f}±{s:.4f}  acc {a:.4f}±{sa:.4f}\n")
    g = df.groupby(["config", "semente"])[["tempo_s", "epocas"]].first().groupby("config").agg(["mean", "min", "max"]).round(1)
    f.write("\n" + g.to_string() + "\n")
print(open("resumo.txt").read())
