import pandas as pd

STATUS_WORDS = [
    "COMPLETE", "PENDING", "CLOSED", "PENDING_PAYMENT",
    "PAYMENT_REVIEW", "CANCELED", "PROCESSING", "ON_HOLD"
]

def clean_dataco(df: pd.DataFrame) -> pd.DataFrame:
    # 1. column name cleanup
    df = df.copy()
    df.columns = (
        df.columns
          .str.strip()
          .str.replace('\n', ' ', regex=False)
    )

    # 2. drop duplicates
    df = df.drop_duplicates()

    # 3. drop all-null columns
    null_all_cols = df.columns[df.isna().all()]
    df = df.drop(columns=null_all_cols)

    # 4. drop personal info / unnecessary columns
    cols_to_drop = [
        "Customer Email", "Customer Fname", "Customer Lname",
        "Customer Password", "Product Description", "Product Image"
    ]
    cols_to_drop = [c for c in cols_to_drop if c in df.columns]
    df = df.drop(columns=cols_to_drop)

    # 5. fix misaligned 'Order Status'
    if {"Order State", "Order Status", "Order Zipcode"}.issubset(df.columns):
        df = _fix_order_status(df)

    # 6. unify date columns
    df = _unify_dates(df)

    return df


def _fix_order_status(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    os_state = df["Order State"].astype(str)
    os_status = df["Order Status"].astype(str)
    os_zip = df["Order Zipcode"].astype(str)

    state_upper = os_state.str.upper()
    mask_state_has_statusword = state_upper.str.contains("|".join(STATUS_WORDS), na=False)

    def split_state_and_status(val: str):
        u = val.upper()
        for w in STATUS_WORDS:
            idx = u.find(w)
            if idx != -1:
                region = val[:idx].strip() or None
                status = val[idx:].strip()
                return region, status
        return val, None

    mask_typeA = mask_state_has_statusword & df["Order Status"].isna()

    if mask_typeA.sum() > 0:
        region_status = os_state[mask_typeA].apply(split_state_and_status)
        region_part = region_status.apply(lambda t: t[0])
        status_part = region_status.apply(lambda t: t[1])

        df.loc[mask_typeA, "Order State"] = region_part
        df.loc[mask_typeA, "Order Status"] = status_part

    os_status_stripped = os_status.str.strip()
    os_zip_stripped = os_zip.str.strip()

    mask_state_nan = df["Order State"].isna()
    mask_status_numeric = os_status_stripped.str.fullmatch(r"\d+", na=False)
    mask_typeB = mask_state_nan & mask_status_numeric
    df.loc[mask_typeB, "Order Status"] = pd.NA

    return df


def _unify_dates(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    date_cols_raw = [
        "order date (DateOrders)",
        "shipping date (DateOrders)",
    ]
    date_cols = [c for c in date_cols_raw if c in df.columns]

    if "order date (DateOrders)" in df.columns:
        df["order date (DateOrders)"] = pd.to_datetime(
            df["order date (DateOrders)"],
            errors="coerce",
            dayfirst=True
        )

    if "shipping date (DateOrders)" in df.columns:
        df["shipping date (DateOrders)"] = pd.to_datetime(
            df["shipping date (DateOrders)"],
            errors="coerce",
            format="%m/%d/%Y %H:%M"
        )

    for col in date_cols:
        df[col] = df[col].dt.strftime("%d/%m/%Y")

    return df




