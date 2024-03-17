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
person_index = 0
sample_person = X.iloc[person_index].values.reshape(1, -1)

# Znalezienie najbliższych sąsiadów
neighbors_count = 4
distances, indices = knn_model.kneighbors(sample_person, neighbors_count + 1)

new_data = data.iloc[indices[0]].copy()

column_count = new_data.shape[1]
for column_index in range(column_count):
    if column_index == 0:
        continue

    column = new_data.iloc[:, column_index]
    if pd.to_numeric(column.iloc[0], errors='coerce') >= 4:
        new_data.iloc[0, column_index] = 'TAK'
    elif pd.to_numeric(column.iloc[0], errors='coerce') < 4:
        new_data.iloc[0, column_index] = 'NIE'
    else:
        number_of_people_rating = 0
        rating_sum = 0
        for neighbor_index in range(neighbors_count):
            rating = pd.to_numeric(column.iloc[neighbor_index + 1], errors='coerce')
            if not pd.isnull(rating):
                number_of_people_rating += 1
                rating_sum += rating

        if number_of_people_rating == 0:
            new_data.iloc[0, column_index] = 'BRAK_DANYCH'
        elif rating_sum / number_of_people_rating >= 4:
            new_data.iloc[0, column_index] = 'TAK'
        else:
            new_data.iloc[0, column_index] = 'NIE'

new_data.to_csv('wyniki_rekomendacji.csv', sep=',', index=False)
