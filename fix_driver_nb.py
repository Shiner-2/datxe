import nbformat as nbf

path = "notebooks/driver_price_acceptance.ipynb"
with open(path, 'r', encoding='utf-8') as f:
    nb = nbf.read(f, as_version=4)

for cell in nb.cells:
    if cell.cell_type == 'code' and "X_m0 = pd.get_dummies(train" in cell.source:
        cell.source = cell.source.replace("train[[", "df[[")
        cell.source = cell.source.replace("train[\"", "df[\"")

with open(path, 'w', encoding='utf-8') as f:
    nbf.write(nb, f)
print("Fixed train -> df in driver_price_acceptance.ipynb")
