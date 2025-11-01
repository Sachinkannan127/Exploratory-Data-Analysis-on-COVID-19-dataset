# 1.importing packages

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# 2. Load Dataset
try:
    df = pd.read_csv('covid_data.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Simulating dataset as 'covid_data.csv' was not found.")
    data = {
        'Country/Region': ['USA', 'India', 'Brazil', 'USA', 'India', 'Brazil', 'USA', 'India', 'Brazil'],
        'ObservationDate': ['2020-03-01', '2020-03-01', '2020-03-01', '2020-03-31', '2020-03-31', '2020-03-31', '2021-01-01', '2021-01-01', '2021-01-01'],
        'Confirmed': [1000, 10, 50, 10000, 500, 2000, 25000, 15000, 18000],
        'Deaths': [10, 0, 1, 150, 5, 50, 500, 250, 300],
        'Recovered': [100, 5, 10, 5000, 100, 500, 10000, 8000, 7000],
        'Tests': [10000, 500, 2000, 100000, 15000, 50000, 500000, 300000, 450000]
    }
    df = pd.DataFrame(data)

print("\nInitial DataFrame Head:")
print(df.head())
print("\nInitial DataFrame Info:")
print(df.info())

# 3. Data Cleaning

df.columns = df.columns.str.lower().str.replace('[ /]', '_', regex=True)
df = df.rename(columns={'country_region': 'country', 'observationdate': 'date'})

print("\nStandardized Column Names:")
print(df.columns)

print("\nMissing Values Check:")
print(df.isnull().sum())

df['recovered'] = df['recovered'].fillna(0)
df.dropna(inplace=True)

# Remove duplicates
initial_rows = len(df)
df.drop_duplicates(inplace=True)
rows_after_deduplication = len(df)
print(f"\nRemoved {initial_rows - rows_after_deduplication} duplicate rows.")


# 4. Data Preprocessing

df['date'] = pd.to_datetime(df['date'])
print("\nDataFrame Info after Type Conversion:")
print(df.info())

df['active'] = df['confirmed'] - df['deaths'] - df['recovered']

print("\nDataFrame Head with 'active' column:")
print(df.head())

# Group the data by Country and find the maximum confirmed cases
country_summary = df.groupby('country').agg(
    total_confirmed=('confirmed', 'max'),
    total_deaths=('deaths', 'max'),
    total_recovered=('recovered', 'max')
).reset_index()

top_10_confirmed = country_summary.sort_values(by='total_confirmed', ascending=False).head(10)

plt.figure(figsize=(12, 6))
sns.barplot(x='total_confirmed', y='country', data=top_10_confirmed, palette='viridis')
plt.title('Top 10 Countries by Total Confirmed COVID-19 Cases')
plt.xlabel('Total Confirmed Cases (Max)')
plt.ylabel('Country')
plt.xticks(rotation=0)
plt.show()

plt.figure(figsize=(14, 7))
sns.lineplot(x='date', y='confirmed', data=global_trend, label='Global Confirmed Cases')
plt.title('Global Trend of Total Confirmed COVID-19 Cases Over Time')
plt.xlabel('Date')
plt.ylabel('Total Confirmed Cases')
plt.legend()
plt.xticks(rotation=45)
plt.tight_layout()
plt.show()

# Identify numeric columns
numeric_cols = ['confirmed', 'deaths', 'recovered', 'active', 'tests']

correlation_matrix = df[numeric_cols].corr()
print("\nCorrelation Matrix:")
print(correlation_matrix)

# Visualize the correlation matrix using a heatmap
plt.figure(figsize=(8, 6))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt=".2f", linewidths=.5, linecolor='black')
plt.title('Correlation Matrix of COVID-19 Metrics')
plt.show()

continent_mapping = {
    'USA': 'North America',
    'India': 'Asia',
    'Brazil': 'South America'
}
df['continent'] = df['country'].map(continent_mapping)

if 'continent' in df.columns and df['continent'].nunique() > 1:
    continent_summary = df.groupby('continent').agg({
        'confirmed': 'max',
        'deaths': 'max',
        'recovered': 'max'
    }).reset_index()

    total_cases_per_continent = continent_summary.set_index('continent')['confirmed']

    plt.figure(figsize=(8, 8))
    plt.pie(total_cases_per_continent, labels=total_cases_per_continent.index, autopct='%1.1f%%', startangle=90, wedgeprops={'edgecolor': 'black'})
    plt.title('Total Confirmed Cases by Continent')
    plt.show()
else:
    print("\nSkipping Continent-Level Analysis: 'continent' column is missing or has insufficient unique values.")
