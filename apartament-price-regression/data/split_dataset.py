from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split


DATA_DIR = Path(__file__).resolve().parent

INPUT_PATH = DATA_DIR / "new_apartments_10k.csv"

TRAIN_PATH = DATA_DIR / "train_apartments.csv"
VALIDATION_PATH = DATA_DIR / "validation_apartments.csv"
TEST_PATH = DATA_DIR / "test_apartments.csv"

REFERENCE_PATH = DATA_DIR / "reference_apartments.csv"
CURRENT_PATH = DATA_DIR / "current_apartments.csv"


def main() -> None:
    df = pd.read_csv(INPUT_PATH)

    train_df, temp_df = train_test_split(
        df,
        test_size=0.30,
        random_state=42,
        shuffle=True,
    )

    validation_df, test_df = train_test_split(
        temp_df,
        test_size=0.50,
        random_state=42,
        shuffle=True,
    )

    train_df.to_csv(TRAIN_PATH, index=False)
    validation_df.to_csv(VALIDATION_PATH, index=False)
    test_df.to_csv(TEST_PATH, index=False)

    # Dados de referência para drift
    train_df.drop(columns=["price"], errors="ignore").to_csv(REFERENCE_PATH, index=False)

    # Simulação de dados atuais/produção
    test_df.drop(columns=["price"], errors="ignore").to_csv(CURRENT_PATH, index=False)

    print("Separação concluída com sucesso.")
    print(f"Total: {len(df)}")
    print(f"Treino: {len(train_df)} -> {TRAIN_PATH}")
    print(f"Validação: {len(validation_df)} -> {VALIDATION_PATH}")
    print(f"Teste: {len(test_df)} -> {TEST_PATH}")
    print(f"Referência drift: {REFERENCE_PATH}")
    print(f"Dados atuais drift: {CURRENT_PATH}")


if __name__ == "__main__":
    main()