import os
os.environ['OMP_NUM_THREADS'] = '1'

from flask import Flask, render_template, request, jsonify
import pandas as pd
import joblib
import numpy as np

app = Flask(__name__)

# Load processed data
df = joblib.load("processed_data.pkl")

# Load Models
try:
    population_models = joblib.load("population_models.pkl")
except FileNotFoundError:
    population_models = None

# Load original dataset
original_df = pd.read_csv("world_population.csv")

# Country list
countries = sorted(df['Country'].unique())

def calculate_growth_rate(population_history):
    """Calculate growth rate between consecutive years"""
    years = list(population_history.keys())
    populations = list(population_history.values())
    
    if len(populations) < 2:
        return 0
    
    # Calculate growth from 1970 to 2022
    first_pop = populations[0]
    last_pop = populations[-1]
    years_diff = 2022 - 1970
    
    if first_pop == 0:
        return 0
    
    cagr = (pow(last_pop / first_pop, 1 / years_diff) - 1) * 100
    return round(cagr, 2)

@app.route("/", methods=["GET", "POST"])
def home():

    prediction = None
    country_name = None
    year = None
    population_history = None
    country_details = None
    kpis = None
    chart_data = None
    model_type = "XGBoost"

    if request.method == "POST":

        country_name = request.form["country"]
        year = int(request.form["year"])

        if population_models and country_name in population_models:
            # Use Polynomial Regression model for prediction
            model = population_models[country_name]
            result = model.predict([[year]])
            prediction = int(result[0])
        else:
            prediction = None

        # Get historical population data
        row = original_df[
            original_df['Country/Territory'] == country_name
        ].iloc[0]

        population_history = {
            "1970": int(row["1970 Population"]),
            "1980": int(row["1980 Population"]),
            "1990": int(row["1990 Population"]),
            "2000": int(row["2000 Population"]),
            "2010": int(row["2010 Population"]),
            "2015": int(row["2015 Population"]),
            "2020": int(row["2020 Population"]),
            "2022": int(row["2022 Population"])
        }

        # Calculate KPIs
        current_population = int(row["2022 Population"])
        prev_population = int(row["2020 Population"])
        growth_rate = calculate_growth_rate(population_history)
        year_over_year = ((current_population - prev_population) / prev_population * 100) if prev_population > 0 else 0
        
        kpis = {
            "current_population": current_population,
            "growth_rate": growth_rate,
            "yoy_growth": round(year_over_year, 2),
            "density": float(row["Density (per km²)"]) if pd.notna(row["Density (per km²)"]) else 0,
            "area": int(row["Area (km²)"]) if pd.notna(row["Area (km²)"]) else 0,
            "world_percentage": float(row["World Population Percentage"]) if pd.notna(row["World Population Percentage"]) else 0
        }

        # Country details
        country_details = {
            "capital": row["Capital"],
            "continent": row["Continent"],
            "world_percentage": kpis["world_percentage"]
        }

        # Chart data
        chart_data = {
            "years": list(population_history.keys()),
            "populations": list(population_history.values())
        }

    return render_template(
        "index.html",
        countries=countries,
        prediction=prediction,
        country_name=country_name,
        year=year,
        population_history=population_history,
        country_details=country_details,
        kpis=kpis,
        chart_data=chart_data,
        model_type=model_type
    )

if __name__ == "__main__":
    app.run(debug=True, threaded=False)