import numpy as np
import pandas as pd

np.random.seed(42)

N = 10000

cities = [
    ("São Paulo", 1.0),
    ("Rio de Janeiro", 0.9),
    ("Brasília", 0.85),
    ("Florianópolis", 0.95),
    ("Belo Horizonte", 0.75),
    ("Curitiba", 0.8),
    ("Porto Alegre", 0.78),
    ("Salvador", 0.7),
    ("Recife", 0.72),
    ("Fortaleza", 0.68),
]

def generate_sample():
    city, factor = cities[np.random.randint(len(cities))]

    area_m2 = np.random.randint(30, 250)
    bedrooms = np.random.randint(1, 5)
    bathrooms = max(1, bedrooms + np.random.randint(-1, 2))
    floor = np.random.randint(0, 25)
    parking_spaces = np.random.randint(0, 3)

    neighborhood_score = np.round(np.random.uniform(4.0, 10.0), 1)
    condo_fee = np.round(np.random.uniform(200, 1500), 2)
    age_years = np.random.randint(0, 40)
    distance_to_center_km = np.round(np.random.uniform(0.5, 25), 2)

    # 💰 fórmula de preço realista
    base_price_m2 = np.random.uniform(7000, 15000) * factor

    price = (
        area_m2 * base_price_m2
        + bedrooms * 50000
        + bathrooms * 30000
        + parking_spaces * 40000
        + neighborhood_score * 20000
        - age_years * 8000
        - distance_to_center_km * 10000
        + condo_fee * 10
    )

    price = max(price, 80000)

    return {
        "city": city,
        "area_m2": area_m2,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floor": floor,
        "parking_spaces": parking_spaces,
        "neighborhood_score": neighborhood_score,
        "condo_fee": condo_fee,
        "age_years": age_years,
        "distance_to_center_km": distance_to_center_km,
        "price": round(price, 2),
    }


data = [generate_sample() for _ in range(N)]
df = pd.DataFrame(data)

df.to_csv("data/new_apartments_10k.csv", index=False)

print("Dataset gerado com sucesso:", df.shape)
print(df.head())