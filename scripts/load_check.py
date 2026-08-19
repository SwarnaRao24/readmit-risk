from ucimlrepo import fetch_ucirepo

# id=296 is the Diabetes 130-US Hospitals dataset
ds = fetch_ucirepo(id=296)
X = ds.data.features
y = ds.data.targets

print("Features shape:", X.shape)
print("Targets shape:", y.shape)
print("\nTarget column(s):", list(y.columns))
print("\nTarget value counts:")
print(y.iloc[:, 0].value_counts())
print("\nFirst few feature columns:", list(X.columns[:12]))
