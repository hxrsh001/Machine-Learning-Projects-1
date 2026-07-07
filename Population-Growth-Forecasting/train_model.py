import pandas as pd
import joblib
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import make_pipeline
import os

# Load dataset
df = pd.read_csv("world_population.csv")

# Store transformed rows
data = []

years = [1970, 1980, 1990, 2000, 2010, 2015, 2020, 2022]

# Convert wide data into long format
for _, row in df.iterrows():
    country = row['Country/Territory']
    for year in years:
        population = row[f'{year} Population']
        data.append([country, year, population])

# Create processed dataframe
processed_df = pd.DataFrame(
    data,
    columns=['Country', 'Year', 'Population']
)

# Save processed dataframe
joblib.dump(processed_df, "processed_data.pkl")

print("Processed data saved successfully")

# Train Polynomial Regression models for each country
print("\nTraining Polynomial Regression models for each country...")

countries = processed_df['Country'].unique()
models = {}

for country in countries:
    country_data = processed_df[processed_df['Country'] == country]
    
    X = country_data[['Year']].values
    y = country_data['Population'].values
    
    # Train Polynomial Regression model (Degree 2 usually works well for population without overfitting into extreme curves)
    model = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model.fit(X, y)
    
    models[country] = model
    print(f"Done: {country}")

# Save all models
joblib.dump(models, "population_models.pkl")
print("\nAll models saved successfully!")