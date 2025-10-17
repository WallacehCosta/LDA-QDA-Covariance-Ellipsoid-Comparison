import numpy as np
import pandas as pd

def database(filename):
    database = pd.read_csv(filename, delimiter = ';')
    x = database.iloc[:, 1:].values
    y = database.iloc[:, -1].values
    return x, y

def missingValues(x):
    from sklearn.impute import SimpleImputer
    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    x[:, 1:] = imputer.fit_transform(x[:, 1:])
    return x

def labelEncoderDummies(x):
    from sklearn.preprocessing import LabelEncoder
    lencoder_x = LabelEncoder()
    x[:, 0] = lencoder_x.fit_transform(x[:, 0])

    #one-hot encoder
    dummies = pd.get_dummies(x[:, 0])
    x = np.hstack((dummies.values, x[:, 1:]))
    return x

def splitTrainTest(x, y, test_size):
    from sklearn.model_selection import train_test_split
    xtrain, xtest, ytrain, ytest = train_test_split(x, y, test_size=test_size)
    return xtrain, xtest, ytrain, ytest

def simpleImputer(xtrain, xtest):
    from sklearn.impute import SimpleImputer
    scalex = SimpleImputer()
    xtrain = scalex.fit_transform(xtrain)
    xtest = scalex.transform(xtest) #Se usa 'fit_transform' apenas em dados de treino
    return xtrain, xtest

# a regressão linear vai ser treinada pelos dados separados em teste e treino
def linearregressionmodel(xtrain, xtest, ytrain, ytest):
    import matplotlib.pyplot as plt
    from sklearn.linear_model import LinearRegression
    regressor = LinearRegression()
    regressor = regressor.fit(xtrain, ytrain)
    ypred = regressor.predict(xtest) #'prevendo' os valores com base nos dados de teste (a linha)

    plt.scatter(xtest[:, -1], ytest, color='red') #'plt.scatter' cria um gráfico de dispersão (valores reais) - pontos

    # 'plt.plot' cria uma linha de regressão com as previsões (ypred)
    plt.plot(xtest[:, -1], ypred, color = 'blue')
    plt.title("Inscritos X Visualizações")
    plt.xlabel("Total de Inscritos")
    plt.ylabel("Total de Visualizações")
    plt.grid()
    plt.show()

def rodaTudo():
    x, y = database('svbr.csv')
    x = missingValues(x)
    x = labelEncoderDummies(x)
    xtrain, xtest, ytrain, ytest = splitTrainTest(x, y, 0.2)
    xtrain, xtest = simpleImputer(xtrain, xtest)
    linearregressionmodel(xtrain, xtest, ytrain, ytest)

if __name__ == '__main__':
    rodaTudo()
