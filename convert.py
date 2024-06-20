import pandas as pd

# Carrega o arquivo CSV
df = pd.read_csv(r'C:\Users\Assistencia\Documents\GitHub\ChatBotLlama\docs\TerceiroSemestre - TerceiroSemestre.csv.csv')

# Salva o DataFrame como arquivo Parquet
df.to_parquet(r'C:\Users\Assistencia\Documents\GitHub\ChatBotLlama\docs\TerceiroSemestre - TerceiroSemestre.parquet')
