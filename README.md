# atividade-4-parte-1

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Dados fictícios
data = {
    'Combustível': ['Gasolina', 'Diesel', 'Etanol', 'Gasolina', 'Diesel'],
    'Idade': [5, 3, 1, 7, 2],
    'Quilometragem': [50000, 30000, 15000, 70000, 20000],
    'Preço': [30000, 25000, 20000, 28000, 22000]
}

df = pd.DataFrame(data)

# Separando variáveis independentes e dependentes
X = df[['Combustível', 'Idade', 'Quilometragem']]
y = df['Preço']

# Dividindo os dados em conjunto de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Criando o ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('cat', OneHotEncoder(), ['Combustível']),
        ('num', StandardScaler(), ['Idade', 'Quilometragem'])
    ])

# Criando o pipeline
pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', LinearRegression())
])

# Treinando o modelo
pipeline.fit(X_train, y_train)

# Fazendo previsões
y_pred = pipeline.predict(X_test)

# Calculando o MSE
mse = mean_squared_error(y_test, y_pred)
print(f'Erro Quadrático Médio: {mse}')
