import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------------------
# LOAD DATASET
# -----------------------------------

df = pd.read_csv("titanic.csv")

print("\n========== FIRST 5 ROWS ==========\n")
print(df.head())

# -----------------------------------
# BASIC INFORMATION
# -----------------------------------

print("\n========== DATASET INFO ==========\n")
print(df.info())

print("\n========== DATASET SHAPE ==========\n")
print(df.shape)

print("\n========== COLUMN NAMES ==========\n")
print(df.columns)

print("\n========== STATISTICAL SUMMARY ==========\n")
print(df.describe())

# -----------------------------------
# CHECK MISSING VALUES
# -----------------------------------

print("\n========== MISSING VALUES ==========\n")
print(df.isnull().sum())

# Fill missing Age values with median
df['Age'].fillna(df['Age'].median(), inplace=True)

# Fill Embarked missing values
df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

# Drop Cabin column because too many missing values
df.drop(columns=['Cabin'], inplace=True)

print("\n========== MISSING VALUES AFTER CLEANING ==========\n")
print(df.isnull().sum())

# -----------------------------------
# CHECK DUPLICATES
# -----------------------------------

print("\n========== DUPLICATE ROWS ==========\n")
print(df.duplicated().sum())

# -----------------------------------
# DATA VISUALIZATION
# -----------------------------------

sns.set(style="whitegrid")

# 1. Survival Count
plt.figure(figsize=(6,4))
sns.countplot(x='Survived', data=df)
plt.title("Survival Count")
plt.savefig("survival_count.png")
plt.show()

# Observation:
# More passengers died than survived.

# -----------------------------------

# 2. Gender Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', data=df)
plt.title("Gender Distribution")
plt.savefig("gender_distribution.png")
plt.show()

# Observation:
# Male passengers were more than female passengers.

# -----------------------------------

# 3. Survival Based on Gender
plt.figure(figsize=(6,4))
sns.countplot(x='Sex', hue='Survived', data=df)
plt.title("Survival Based on Gender")
plt.savefig("survival_gender.png")
plt.show()

# Observation:
# Females had higher survival rate.

# -----------------------------------

# 4. Age Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Age'], bins=30, kde=True)
plt.title("Age Distribution")
plt.savefig("age_distribution.png")
plt.show()

# Observation:
# Most passengers were between 20 and 40 years old.

# -----------------------------------

# 5. Passenger Class Distribution
plt.figure(figsize=(6,4))
sns.countplot(x='Pclass', data=df)
plt.title("Passenger Class Distribution")
plt.savefig("pclass_distribution.png")
plt.show()

# Observation:
# Most passengers traveled in 3rd class.

# -----------------------------------

# 6. Fare Distribution
plt.figure(figsize=(8,5))
sns.histplot(df['Fare'], bins=40, kde=True)
plt.title("Fare Distribution")
plt.savefig("fare_distribution.png")
plt.show()

# Observation:
# Most passengers paid lower fares.

# -----------------------------------

# 7. Boxplot for Outliers
plt.figure(figsize=(8,5))
sns.boxplot(x=df['Fare'])
plt.title("Fare Outliers")
plt.savefig("fare_outliers.png")
plt.show()

# Observation:
# Some passengers paid extremely high fares.

# -----------------------------------

# 8. Correlation Heatmap
plt.figure(figsize=(10,6))

numeric_df = df.select_dtypes(include=np.number)

sns.heatmap(
    numeric_df.corr(),
    annot=True,
    cmap='coolwarm'
)

plt.title("Correlation Heatmap")
plt.savefig("correlation_heatmap.png")
plt.show()

# Observation:
# Passenger class negatively correlates with fare.

# -----------------------------------
# GROUP ANALYSIS
# -----------------------------------

print("\n========== AVERAGE AGE ==========\n")
print(df['Age'].mean())

print("\n========== AVERAGE FARE ==========\n")
print(df['Fare'].mean())

print("\n========== SURVIVAL RATE ==========\n")
print(df['Survived'].mean())

print("\n========== SURVIVAL BY GENDER ==========\n")
print(df.groupby('Sex')['Survived'].mean())

print("\n========== SURVIVAL BY PASSENGER CLASS ==========\n")
print(df.groupby('Pclass')['Survived'].mean())

# -----------------------------------
# FINAL INSIGHTS
# -----------------------------------

print("\n========== FINAL INSIGHTS ==========\n")

print("""
1. Female passengers had a higher survival rate.
2. Most passengers belonged to 3rd class.
3. Younger and middle-aged passengers were dominant.
4. Higher fare passengers had better survival chances.
5. Some extreme fare outliers were present.
6. Passenger class strongly influenced survival.
""")

print("\nEDA PROJECT COMPLETED SUCCESSFULLY")