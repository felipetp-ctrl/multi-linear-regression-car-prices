# Regressão Linear Múltipla para Preço de Carros

Este repositório contém minha primeira implementação de um modelo de regressão linear múltipla para estimar preços de carros com base em características como marca, tipo de carroceria, motor, quilometragem, ano etc. O modelo foi realizado a partir do Data Science Bootcamp da 365Data Science

## Objetivos
- Explorar os dados e entender variáveis que mais afetam o preço.
- Aplicar limpeza e tratamento de outliers.
- Transformar o preço para escala logarítmica e interpretar coeficientes como variação percentual aproximada.
- Treinar, avaliar e interpretar o modelo.
- Documentar insights e próximos passos.

## Estrutura
```
src/
  linear_regression.py  # Script principal
notebooks/
  exploratory.ipynb      # EDA
reports/
  figures/               # Gráficos gerados
  model_summary.md       # Resumo OLS (statsmodels)
data/
  raw/                   # Dados originais
```

## Pipeline Resumido
1. Carregar dados (`src/linear_regression.py`)
2. Limpar:
   - Remoção de nulos
   - Remoção de outliers (Price, Mileage, EngineV, Year)
3. Transformar:
   - Log do preço
   - Dummies para categóricas
4. Multicolinearidade:
   - VIF calculado e remoção de Year devido relação com Mileage
5. Treinamento:
   - `LinearRegression` do scikit-learn
6. Avaliação:
   - R² treino e teste
   - MAE e RMSE no espaço original (exp do log)
   - Gráficos (resíduos, previsão vs real)
7. Interpretação:
   - Coeficientes convertidos em % aproximado
   - Resumo estatístico via statsmodels

## Como Reproduzir
```bash
git clone https://github.com/felipetp-ctrl/multi-linear-regression-car-prices.git
cd multi-linear-regression-car-prices
pip install -r requirements.txt
python src/linear_regression.py
```

## Principais Perguntas Respondidas
- Qual o efeito (aprox. percentual) de ser da marca X no preço, controlando por outros fatores?
- Qual a depreciação associada à quilometragem?
- Qual o “premium” de motor maior (EngineV)?
- Registro (sim/não) agrega valor?
- Quais variáveis apresentam maior peso e quais poderiam ser adicionadas?

## Próximos Passos
- Adicionar validação cruzada.
- Testar modelagem com Ridge/Lasso.
- Incluir diagnóstico de heterocedasticidade (Breusch-Pagan).
- Calcular intervalos de confiança dos coeficientes.
- Comparar com modelos baseados em árvore (Random Forest, XGBoost).

## Licença
MIT

## Contato
Estou sempre aberto para sugestões!  Abra uma issue ou conecte-se comigo no [LinkedIn](https://www.linkedin.com/in/felipetpereira/).
