import os
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import mean_absolute_error, mean_squared_error
from scipy.stats import pearsonr
from statsmodels.stats.anova import anova_lm
import statsmodels.api as sm
from statsmodels.formula.api import ols
from statsmodels.stats.multicomp import pairwise_tukeyhsd

# -------------------------------------------------------------
# CONFIGURAÇÕES INICIAIS
# -------------------------------------------------------------
os.makedirs("../results", exist_ok=True)
output_file = "../results/anova_all_results.txt"
open(output_file, "w").close()  # limpa o arquivo

# -------------------------------------------------------------
# 1. BOXPLOT DE MÉTRICAS DE DESEMPENHO
# -------------------------------------------------------------
dados = pd.read_csv("../results/results.csv")

metricas = ["mAP50", "mAP75", "mAP", "precision", "recall", "f1", "MAE", "RMSE"]
ncols = 3
nrows = int(np.ceil(len(metricas) / ncols))
fig, axes = plt.subplots(nrows, ncols, figsize=(12, 10))
axes = axes.flatten()

for i, metrica in enumerate(metricas):
    sns.boxplot(
        data=dados,
        x="model",
        y=metrica,
        palette="Purples",
        ax=axes[i]
    )
    axes[i].set_title(f"Boxplot for {metrica}")
    axes[i].set_xlabel("Models")
    axes[i].set_ylabel(metrica)
    axes[i].grid(True, linestyle="--", alpha=0.4)

for j in range(i + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
plt.savefig("../results/boxplot.png", dpi=300)
plt.close()

# -------------------------------------------------------------
# 2. CONTAGEM MANUAL x AUTOMÁTICA (RMSE, MAE, MAPE, r)
# -------------------------------------------------------------
dados_contagem = pd.read_csv("../results/counting.csv")
modelos = dados_contagem["ml"].unique()

rmse_values = []
graficos = []

for modelo in modelos:
    subset = dados_contagem[dados_contagem["ml"] == modelo]
    gt = subset["groundtruth"].values
    pred = subset["predicted"].values

    rmse = np.sqrt(mean_squared_error(gt, pred))
    mae = mean_absolute_error(gt, pred)
    mape = np.mean(np.abs((gt - pred) / np.clip(gt, 1e-8, None)))
    r, _ = pearsonr(gt, pred)

    titulo = f"{modelo} RMSE={rmse:.3f} MAE={mae:.3f} MAPE={mape:.3f} r={r:.3f}"
    rmse_values.append({"Model": modelo, "RMSE": rmse})

    plt.figure()
    sns.regplot(x=gt, y=pred, scatter_kws={"s": 15})
    plt.title(titulo)
    plt.xlabel("Contagem Manual (Ground Truth)")
    plt.ylabel("Contagem Preditiva")
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.savefig(f"../results/{modelo}_counting.png", dpi=300)
    plt.close()

rmse_df = pd.DataFrame(rmse_values)
rmse_df.to_csv("../results/rmse_values.csv", index=False)

# -------------------------------------------------------------
# 3. FUNÇÃO ANOVA + TUKEY
# -------------------------------------------------------------
def realizar_anova(df, metric, output_file):
    with open(output_file, "a") as f:
        f.write("\n------------------------------------------------------------\n")
        f.write(f"ANOVA para {metric}\n")

        try:
            model = ols(f"{metric} ~ C(model)", data=df).fit()
            anova_table = anova_lm(model, typ=2)
            f.write(str(anova_table))
            f.write("\n")

            p_value = anova_table["PR(>F)"][0]
            if p_value < 0.05:
                f.write("\nTukey HSD para {}\n".format(metric))
                tukey = pairwise_tukeyhsd(df[metric], df["model"], alpha=0.05)
                f.write(str(tukey.summary()))
                f.write("\n")
        except Exception as e:
            f.write(f"Erro ao realizar ANOVA para {metric}: {e}\n")

# -------------------------------------------------------------
# 4. EXECUTAR ANOVA PARA TODAS AS MÉTRICAS
# -------------------------------------------------------------
metrics = ["mAP", "mAP50", "mAP75", "MAE", "RMSE", "precision", "recall", "f1"]

for metric in metrics:
    realizar_anova(dados, metric, output_file)

print(f"\n✅ Resultados salvos em: {output_file}")
