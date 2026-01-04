
import pandas as pd # data frames


df_mod = df.copy() #Create a copy of the data frame

#Convering dates into datetime format. For 'PQ First Sent to Client Data
dt = ['PQ First Sent to Client Date' ,'PO Sent to Vendor Date','Scheduled Delivery Date','Delivered to Client Date', 'Delivery Recorded Date']
for col in dt:
    df[col] = pd.to_datetime(df[col], errors = 'coerce')

df_mod = df_mod.drop(columns=["Dosage"]) #Delete Dosage column

df_mod = df_mod.dropna(subset=["Shipment Mode"]) #Delete registers without shipment Mode

#Weight y freight cost as number

df_mod['Weight (Kilograms)'] = pd.to_numeric(
    df['Weight (Kilograms)'],
    errors='coerce'
)

df_mod['Freight Cost (USD)'] = pd.to_numeric(
    df['Freight Cost (USD)'],
    errors='coerce'
)


#Category variables cleaning

cat_cols = [
    'Shipment Mode',
    'Country',
    'Vendor',
    'Fulfill Via',
    'Managed By',
    'Vendor INCO Term',
    'Product Group'
]
for col in cat_cols:
    df_mod[col] = (
        df_mod[col]
        .where(df_mod[col].notna(), np.nan)
        .str.strip()
        .str.upper()
    )

return df_mod
