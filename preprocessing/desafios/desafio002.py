import numpy as np
import pandas as pd

dados = pd.read_csv("desafio002.csv")
print("Antes:\n", dados)

Q1 = dados.select_dtypes(include=np.number).quantile(0.25)
Q3 = dados.select_dtypes(include=np.number).quantile(0.75)
IQR = Q3 - Q1

num_col =  dados.select_dtypes(include=np.number).columns
#indentify_outiliers = dados[~((dados[num_col] < (Q1 - 1.5 * IQR)) | (dados[num_col] > (Q3 + 1.5 * IQR)))]
#dados = dados[indentify_outiliers].drop(axis=0)

indentify_outliers = ((dados[num_col] < (Q1 - 1.5 * IQR)) | (dados[num_col] > (Q3 + 1.5 * IQR))).any(axis=1)
dados = dados[~indentify_outliers]

print("\nDepois:\n", dados)
