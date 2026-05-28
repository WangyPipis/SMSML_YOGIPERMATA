import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load dataset raw
df = pd.read_csv("titanic.csv")

print("Dataset berhasil dimuat")
print(df.head())

# Drop kolom yang tidak digunakan
df.drop(columns=["Name", "Ticket", "Cabin"], inplace=True)

# Handle missing value
df["Age"] = df["Age"].fillna(df["Age"].median())
df["Embarked"] = df["Embarked"].fillna(df["Embarked"].mode()[0])

# Encoding categorical
label_encoder = LabelEncoder()

df["Sex"] = label_encoder.fit_transform(df["Sex"])
df["Embarked"] = label_encoder.fit_transform(df["Embarked"])

# Pisahkan fitur numerik untuk scaling
numeric_cols = ["Age", "Fare"]

scaler = StandardScaler()

df[numeric_cols] = scaler.fit_transform(df[numeric_cols])

# Simpan hasil preprocessing
df.to_csv("titanic_preprocessing.csv", index=False)

print("Preprocessing selesai")
print(df.head())