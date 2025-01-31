import pandas as pd

data=pd.read_csv("data/census.csv")

spaces_found = data.applymap(lambda x: isinstance(x, str) and ' ' in x).any().any()

print(f"Spaces found: {spaces_found}")
