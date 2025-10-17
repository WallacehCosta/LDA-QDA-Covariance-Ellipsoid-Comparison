import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split


database = pd.read_csv("database.csv")
print(database.head())

database = database.drop_duplicates() #2 - removendo duplicatas

#3 tratando valores faltantes (nesse caso valores numéricos)
imputer = SimpleImputer(strategy = 'mean') #criando objeto de classe
database[database.select_dtypes(include=np.number).columns] = (imputer.fit_transform(database.select_dtypes(include=np.number))) #A estratégia será aplicada em todas as colunas com valores numéricos

#4 Identificando e removendo outilier
Q1 = database.select_dtypes(include=np.number).quantile(0.25)
Q3 = database.select_dtypes(include=np.number).quantile(0.75)
IQR = Q3 - Q1

num_columns = database.select_dtypes(include=np.number).columns
database = database[~((database[num_columns] < (Q1 - 1.5 * IQR)) | (database[num_columns] > (Q3 + 1.5 * IQR)))]

#5 Separar features de targets
X = database.drop(columns=["Purchased"])
Y = database["Purchased"]

#6 Separar colunas numéricas de categóricas
num_cols = X.select_dtypes(include=np.number).columns
cat_cols = Y.select_dtypes(include="object").columns


#7 Criar transformações para cada tipo de dado
num_pipeline = Pipeline([
    ("scaler", StandardScaler()),
    ("pca", PCA(n_components=0.95))
])

cat_pipeline = Pipeline([
    ("encoder", OneHotEncoder(handle_unknown="ignore"))
])


#8 Criar transformador para todo dataset
preprocessor = ColumnTransformer([
    ("num", num_pipeline, num_cols),
    ("cat", cat_pipeline, cat_cols)
])


#9 Aplicar pré-processamento
x_prepro = preprocessor.fit_transform(X)


# 10. Balancear dados (se necessário) - IDEA DO CHAT
#if y.value_counts(normalize=True).max() > 0.75:  # Se alguma classe for maior que 75% do dataset
#    smote = SMOTE(sampling_strategy="auto", random_state=42)
#    X_transformed, y = smote.fit_resample(X_transformed, y)

#11 separando dados de teste e dados de treino
xtrain, xtest, ytrain, ytest = train_test_split(x_prepro, Y, test_size=0.2)

print(f"Treino: {xtrain.shape}, Teste: {xtest.shape}")
