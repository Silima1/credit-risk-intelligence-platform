# backend/app/services/features.py

import pandas as pd

# ======================================================
# DATE FEATURES
# ======================================================

def add_date_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    if "application_date" in df.columns:
        df["application_date"] = pd.to_datetime(df["application_date"], errors="coerce")
        df["app_year"] = df["application_date"].dt.year
        df["app_month"] = df["application_date"].dt.month
        df["app_dayofweek"] = df["application_date"].dt.dayofweek
    return df


# ======================================================
# FEATURE COLUMN UTILS
# ======================================================

COMMON_ID_COLS = [
    "ID",
    "id",
    "application_id",
    "customer_id",
    "data_batch_id",
    "Unnamed: 0",
]

def get_feature_columns(model, df: pd.DataFrame):
    """
    Determina colunas de features excluindo IDs e target.
    Compatível com Pipeline e modelos simples.
    """
    if hasattr(model, "feature_names_in_"):
        return list(model.feature_names_in_)

    if hasattr(model, "steps"):
        last = model.steps[-1][1]
        if hasattr(last, "feature_names_in_"):
            return list(last.feature_names_in_)

    non_features = []
    if "fraud_flag" in df.columns:
        non_features.append("fraud_flag")

    for c in COMMON_ID_COLS:
        if c in df.columns:
            non_features.append(c)

    return [c for c in df.columns if c not in non_features]


# ======================================================
# FRIENDLY LABELS (USER-FACING)
# ======================================================

BASE_LABELS = {
    "loan_amount_requested": "Valor do empréstimo pedido",
    "loan_tenure_months": "Prazo do empréstimo (meses)",
    "interest_rate_offered": "Taxa de juro oferecida",
    "purpose_of_loan": "Finalidade do empréstimo",
    "employment_status": "Situação profissional",
    "monthly_income": "Rendimento mensal",
    "yearly_income": "Rendimento anual",
    "cibil_score": "Score de crédito (CIBIL)",
    "existing_emis_monthly": "Prestações mensais atuais (EMIs)",
    "debt_to_income_ratio": "Rácio dívida/rendimento",
    "property_ownership_status": "Habitação",
    "residential_address": "Zona/Endereço residencial",
    "applicant_age": "Idade",
    "number_of_dependents": "Nº de dependentes",
    "loan_amount_usd": "Valor do empréstimo (USD)",
    "credit_utilization_ratio": "Utilização de crédito (%)",
    "annual_bonus": "Bónus anual",
}

VALUE_LABELS = {
    "RENTED": "Arrendada",
    "OWNED": "Própria",
    "MORTGAGED": "Hipotecada",
}


def normalize_raw(name: str) -> str:
    """Remove prefixos técnicos do ColumnTransformer"""
    return name.replace("num__", "").replace("cat__", "")


def pretty_feature_name(raw_feature: str) -> str:
    """
    Converte nomes técnicos em nomes compreensíveis:
      - num__loan_amount_requested
      - cat__purpose_of_loan_Business
      - loan_type_Education Loan
      - gender_Male
    """
    f = raw_feature

    # ColumnTransformer + OneHotEncoder: cat__col_VALOR
    if f.startswith("cat__") and "_" in f:
        tmp = f.replace("cat__", "")
        col, val = tmp.rsplit("_", 1)
        col_label = BASE_LABELS.get(col, col.replace("_", " ").title())
        val_label = VALUE_LABELS.get(val, val.replace("_", " ").title())
        return f"{col_label}: {val_label}"

    # num__col
    if f.startswith("num__"):
        col = f.replace("num__", "")
        return BASE_LABELS.get(col, col.replace("_", " ").title())

    # One-hot simples fora do pipeline
    if f.startswith("loan_type_"):
        return f"Tipo de empréstimo: {f.replace('loan_type_', '').strip()}"

    if f.startswith("gender_"):
        g = f.replace("gender_", "").strip()
        if g.lower() == "male":
            g = "Masculino"
        elif g.lower() == "female":
            g = "Feminino"
        elif g.lower() == "other":
            g = "Outro"
        return f"Género: {g}"

    base = normalize_raw(f)
    return BASE_LABELS.get(base, base.replace("_", " ").title())


def impact_sentence(raw_feature: str, shap_value: float) -> str:
    """
    Gera frase pronta para o utilizador final.
    """
    direction = "aumenta" if shap_value > 0 else "reduz"
    strength = abs(shap_value)

    if strength >= 0.05:
        mag = "muito"
    elif strength >= 0.02:
        mag = "moderadamente"
    else:
        mag = "ligeiramente"

    return f"{pretty_feature_name(raw_feature)} {direction} o risco {mag}."