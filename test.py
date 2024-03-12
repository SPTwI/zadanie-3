import pandas as pd
from sklearn.neighbors import NearestNeighbors

# Wczytanie danych z pliku CSV
data = pd.read_csv('plik.csv')

X = data.drop(columns=['Unnamed: 0'])
X = X.applymap(lambda x: x.strip() if isinstance(x, str) else x)
X = X.replace('x', 0)

# Zamiana wartości na numeryczne
X = X.apply(pd.to_numeric)

knn_model = NearestNeighbors(n_neighbors=2, metric='euclidean')
knn_model.fit(X)

# Przykładowa osoba dla której chcemy uzyskać rekomendacje
sample_person = X.iloc[0].values.reshape(1, -1)

# Znalezienie najbliższych sąsiadów
distances, indices = knn_model.kneighbors(sample_person)

print("Indeksy najbliższych sąsiadów:")
print(indices)

print("Rekomendacje dla danej osoby:")
print(data.iloc[indices[0]])

data.iloc[indices[0]].to_csv('wyniki_rekomendacji.csv', index=False)


