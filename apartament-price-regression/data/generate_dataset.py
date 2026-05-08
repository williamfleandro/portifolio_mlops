from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd


# ========================
# CONFIG
# ========================

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

DATA_DIR = Path(__file__).resolve().parent
OUTPUT_PATH = DATA_DIR / "new_apartments_10k.csv"

# Mantém o nome do arquivo para não quebrar o split_dataset.py atual.
# Para alterar o volume:
# PowerShell: $env:N_ROWS="30000"
N = int(os.getenv("N_ROWS", "30000"))


# ========================
# BASE GEOGRÁFICA E REFERÊNCIA DE MERCADO
# ========================

# Observação técnica:
# - O campo base_price_m2 usa valores de referência do FipeZAP Residencial Venda
#   para capitais monitoradas no informe de abril/2026.
# - Para capitais sem cobertura direta no FipeZAP, foi usado proxy regional conservador.
# - O FipeZAP residencial é mais aderente a apartamentos prontos. Para casas,
#   aplicamos um fator sintético de ajuste em property_type_factor.

CAPITALS = [
    # Capitais monitoradas diretamente pelo FipeZAP Residencial Venda - abril/2026.
    {"state": "AL", "city": "Maceió", "region": "Nordeste", "base_price_m2": 9908, "price_source": "FipeZAP"},
    {"state": "AM", "city": "Manaus", "region": "Norte", "base_price_m2": 7513, "price_source": "FipeZAP"},
    {"state": "BA", "city": "Salvador", "region": "Nordeste", "base_price_m2": 8385, "price_source": "FipeZAP"},
    {"state": "CE", "city": "Fortaleza", "region": "Nordeste", "base_price_m2": 9350, "price_source": "FipeZAP"},
    {"state": "DF", "city": "Brasília", "region": "Centro-Oeste", "base_price_m2": 10090, "price_source": "FipeZAP"},
    {"state": "ES", "city": "Vitória", "region": "Sudeste", "base_price_m2": 14818, "price_source": "FipeZAP"},
    {"state": "GO", "city": "Goiânia", "region": "Centro-Oeste", "base_price_m2": 8226, "price_source": "FipeZAP"},
    {"state": "MA", "city": "São Luís", "region": "Nordeste", "base_price_m2": 8627, "price_source": "FipeZAP"},
    {"state": "MT", "city": "Cuiabá", "region": "Centro-Oeste", "base_price_m2": 6931, "price_source": "FipeZAP"},
    {"state": "MS", "city": "Campo Grande", "region": "Centro-Oeste", "base_price_m2": 6839, "price_source": "FipeZAP"},
    {"state": "MG", "city": "Belo Horizonte", "region": "Sudeste", "base_price_m2": 10663, "price_source": "FipeZAP"},
    {"state": "PA", "city": "Belém", "region": "Norte", "base_price_m2": 8882, "price_source": "FipeZAP"},
    {"state": "PB", "city": "João Pessoa", "region": "Nordeste", "base_price_m2": 8081, "price_source": "FipeZAP"},
    {"state": "PR", "city": "Curitiba", "region": "Sul", "base_price_m2": 11694, "price_source": "FipeZAP"},
    {"state": "PE", "city": "Recife", "region": "Nordeste", "base_price_m2": 8615, "price_source": "FipeZAP"},
    {"state": "PI", "city": "Teresina", "region": "Nordeste", "base_price_m2": 5857, "price_source": "FipeZAP"},
    {"state": "RJ", "city": "Rio de Janeiro", "region": "Sudeste", "base_price_m2": 10939, "price_source": "FipeZAP"},
    {"state": "RN", "city": "Natal", "region": "Nordeste", "base_price_m2": 6334, "price_source": "FipeZAP"},
    {"state": "RS", "city": "Porto Alegre", "region": "Sul", "base_price_m2": 7579, "price_source": "FipeZAP"},
    {"state": "SC", "city": "Florianópolis", "region": "Sul", "base_price_m2": 13208, "price_source": "FipeZAP"},
    {"state": "SP", "city": "São Paulo", "region": "Sudeste", "base_price_m2": 12019, "price_source": "FipeZAP"},
    {"state": "SE", "city": "Aracaju", "region": "Nordeste", "base_price_m2": 5529, "price_source": "FipeZAP"},

    # Estimativas por proxy regional para capitais não cobertas diretamente no FipeZAP.
    {"state": "AC", "city": "Rio Branco", "region": "Norte", "base_price_m2": 5200, "price_source": "estimated_regional_proxy"},
    {"state": "AP", "city": "Macapá", "region": "Norte", "base_price_m2": 5400, "price_source": "estimated_regional_proxy"},
    {"state": "RO", "city": "Porto Velho", "region": "Norte", "base_price_m2": 5600, "price_source": "estimated_regional_proxy"},
    {"state": "RR", "city": "Boa Vista", "region": "Norte", "base_price_m2": 5300, "price_source": "estimated_regional_proxy"},
    {"state": "TO", "city": "Palmas", "region": "Norte", "base_price_m2": 5900, "price_source": "estimated_regional_proxy"},
]


NEIGHBORHOOD_PROFILES = [
    {
        "neighborhood": "Centro",
        "neighborhood_factor": 1.05,
        "score_min": 7.0,
        "score_max": 9.3,
        "distance_min": 0.5,
        "distance_max": 4.0,
    },
    {
        "neighborhood": "Área Nobre",
        "neighborhood_factor": 1.25,
        "score_min": 8.5,
        "score_max": 10.0,
        "distance_min": 0.5,
        "distance_max": 7.0,
    },
    {
        "neighborhood": "Zona Sul",
        "neighborhood_factor": 1.12,
        "score_min": 7.5,
        "score_max": 9.8,
        "distance_min": 2.0,
        "distance_max": 12.0,
    },
    {
        "neighborhood": "Zona Norte",
        "neighborhood_factor": 0.88,
        "score_min": 5.5,
        "score_max": 8.0,
        "distance_min": 4.0,
        "distance_max": 18.0,
    },
    {
        "neighborhood": "Zona Leste",
        "neighborhood_factor": 0.86,
        "score_min": 5.0,
        "score_max": 8.0,
        "distance_min": 5.0,
        "distance_max": 22.0,
    },
    {
        "neighborhood": "Zona Oeste",
        "neighborhood_factor": 0.92,
        "score_min": 5.8,
        "score_max": 8.5,
        "distance_min": 5.0,
        "distance_max": 20.0,
    },
    {
        "neighborhood": "Região Residencial",
        "neighborhood_factor": 0.95,
        "score_min": 6.0,
        "score_max": 8.8,
        "distance_min": 3.0,
        "distance_max": 16.0,
    },
    {
        "neighborhood": "Região Comercial",
        "neighborhood_factor": 1.00,
        "score_min": 6.5,
        "score_max": 9.0,
        "distance_min": 1.0,
        "distance_max": 10.0,
    },
    {
        "neighborhood": "Área Universitária",
        "neighborhood_factor": 0.97,
        "score_min": 6.5,
        "score_max": 9.0,
        "distance_min": 2.0,
        "distance_max": 14.0,
    },
    {
        "neighborhood": "Região Periférica",
        "neighborhood_factor": 0.72,
        "score_min": 4.0,
        "score_max": 7.0,
        "distance_min": 10.0,
        "distance_max": 28.0,
    },
]


# ========================
# FUNÇÕES AUXILIARES
# ========================

def choose_bedrooms(area_m2: int, property_type: str) -> int:
    if property_type == "apartment":
        if area_m2 <= 45:
            return int(np.random.choice([1, 2], p=[0.80, 0.20]))
        if area_m2 <= 80:
            return int(np.random.choice([1, 2, 3], p=[0.20, 0.60, 0.20]))
        if area_m2 <= 140:
            return int(np.random.choice([2, 3, 4], p=[0.30, 0.55, 0.15]))
        return int(np.random.choice([3, 4, 5], p=[0.35, 0.50, 0.15]))

    if area_m2 <= 90:
        return int(np.random.choice([2, 3], p=[0.55, 0.45]))
    if area_m2 <= 180:
        return int(np.random.choice([2, 3, 4], p=[0.20, 0.55, 0.25]))
    return int(np.random.choice([3, 4, 5], p=[0.25, 0.55, 0.20]))


def choose_area_floor_and_parking(property_type: str) -> tuple[int, int, int]:
    if property_type == "apartment":
        area_m2 = int(np.random.randint(30, 220))
        floor = int(np.random.randint(0, 36))
        parking_spaces = int(
            np.random.choice([0, 1, 2, 3], p=[0.25, 0.45, 0.25, 0.05])
        )
        return area_m2, floor, parking_spaces

    area_m2 = int(np.random.randint(60, 320))
    floor = 0
    parking_spaces = int(
        np.random.choice([0, 1, 2, 3, 4], p=[0.10, 0.30, 0.35, 0.20, 0.05])
    )
    return area_m2, floor, parking_spaces


# ========================
# GERAÇÃO
# ========================

def generate_sample() -> dict:
    location = CAPITALS[np.random.randint(len(CAPITALS))]
    neighborhood_profile = NEIGHBORHOOD_PROFILES[
        np.random.randint(len(NEIGHBORHOOD_PROFILES))
    ]

    property_type = str(
        np.random.choice(
            ["apartment", "house"],
            p=[0.72, 0.28],
        )
    )

    state = location["state"]
    city = location["city"]
    region = location["region"]
    base_price_m2 = float(location["base_price_m2"])
    price_source = location["price_source"]

    neighborhood = neighborhood_profile["neighborhood"]
    neighborhood_factor = float(neighborhood_profile["neighborhood_factor"])

    area_m2, floor, parking_spaces = choose_area_floor_and_parking(property_type)
    bedrooms = choose_bedrooms(area_m2, property_type)
    bathrooms = int(max(1, min(5, bedrooms + np.random.randint(-1, 2))))

    neighborhood_score = float(
        np.round(
            np.random.uniform(
                neighborhood_profile["score_min"],
                neighborhood_profile["score_max"],
            ),
            1,
        )
    )

    distance_to_center_km = float(
        np.round(
            np.random.uniform(
                neighborhood_profile["distance_min"],
                neighborhood_profile["distance_max"],
            ),
            2,
        )
    )

    age_years = int(np.random.randint(0, 41))

    if property_type == "apartment":
        property_type_factor = 1.00
        condo_factor = 1.00
    else:
        property_type_factor = 0.82
        condo_factor = 0.35

    market_noise = float(np.random.normal(1.0, 0.035))

    adjusted_price_m2 = (
        base_price_m2
        * neighborhood_factor
        * property_type_factor
        * market_noise
    )

    condo_fee = (
        area_m2
        * np.random.uniform(5.0, 14.0)
        * neighborhood_factor
        * condo_factor
        + parking_spaces * 80
        + floor * 5
        + np.random.normal(0, 60)
    )
    condo_fee = float(np.round(max(condo_fee, 0), 2))

    floor_bonus = min(floor, 25) * 1000 if property_type == "apartment" else 0
    house_land_bonus = 0.0

    if property_type == "house":
        house_land_bonus = float(area_m2 * np.random.uniform(500, 1200))

    price = (
        area_m2 * adjusted_price_m2
        + bedrooms * 35000
        + bathrooms * 25000
        + parking_spaces * 45000
        + neighborhood_score * 22000
        + floor_bonus
        + house_land_bonus
        - age_years * 6500
        - distance_to_center_km * 6500
        + condo_fee * 6
    )

    # Ruído controlado: mantém realismo sem esconder completamente a regra do preço.
    price = price + np.random.normal(0, max(price * 0.035, 12000))
    price = max(price, 80000)

    return {
        "property_type": property_type,
        "state": state,
        "region": region,
        "city": city,
        "neighborhood": neighborhood,
        "price_source": price_source,
        "base_price_m2": base_price_m2,
        "area_m2": area_m2,
        "bedrooms": bedrooms,
        "bathrooms": bathrooms,
        "floor": floor,
        "parking_spaces": parking_spaces,
        "neighborhood_score": neighborhood_score,
        "condo_fee": condo_fee,
        "age_years": age_years,
        "distance_to_center_km": distance_to_center_km,
        "price": round(float(price), 2),
    }


def main() -> None:
    data = [generate_sample() for _ in range(N)]
    df = pd.DataFrame(data)

    df.to_csv(OUTPUT_PATH, index=False)

    print("Dataset gerado com sucesso.")
    print(f"Arquivo: {OUTPUT_PATH}")
    print(f"Shape: {df.shape}")
    print()
    print("Amostra:")
    print(df.head())
    print()
    print("Quantidade por tipo de imóvel:")
    print(df["property_type"].value_counts())
    print()
    print("Quantidade por região:")
    print(df["region"].value_counts())
    print()
    print("Quantidade por estado:")
    print(df["state"].value_counts().sort_index())
    print()
    print("Fonte de preço:")
    print(df["price_source"].value_counts())
    print()
    print("Preço por tipo de imóvel:")
    print(df.groupby("property_type")["price"].describe())
    print()
    print("Colunas:")
    print(df.columns.tolist())


if __name__ == "__main__":
    main()
