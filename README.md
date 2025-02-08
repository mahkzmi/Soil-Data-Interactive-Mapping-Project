# Soil Data Interactive Mapping Project

This project focuses on analyzing soil data and visualizing it on an interactive map. The goal is to classify soil properties, such as pH levels, nitrogen, phosphorus, and potassium content, to identify suitable locations for apple tree cultivation.

## Features
- **Data Processing:** Reads soil and location data from Excel files.
- **Data Cleaning & Analysis:** Checks for missing values and provides descriptive statistics.
- **Clustering Analysis:** Uses K-Means clustering to group similar soil properties.
- **Machine Learning Prediction:** Applies Linear Regression to predict soil pH based on nutrients.
- **Interactive Map:** Generates a folium-based map displaying soil suitability for apple trees.

---

## Installation & Setup

### Prerequisites
Ensure you have the following installed:
- Python 3.x
- Required libraries: `pandas`, `numpy`, `scikit-learn`, `matplotlib`, `folium`, `openpyxl`

You can install dependencies using:
```sh
pip install pandas numpy scikit-learn matplotlib folium openpyxl
```

---

## Usage
### 1. Load Soil and Location Data
Place your Excel files (`soil_data_apple.xlsx` and `location_data_apple.xlsx`) in the project directory. The script will load and merge these files based on geographical coordinates.

```python
soil_data = pd.read_excel('soil_data_apple.xlsx')
location_data = pd.read_excel('location_data_apple.xlsx')
combined_data = pd.merge(soil_data, location_data, on=['Latitude', 'Longitude'])
```

### 2. Data Analysis & Preprocessing
- Checks for missing values.
- Provides descriptive statistics.
- Computes correlation between numeric attributes.

```python
print(combined_data.describe())
print(combined_data.isnull().sum())
corr_matrix = combined_data.corr()
print(corr_matrix)
```

### 3. Clustering with K-Means
To identify regions with similar soil conditions:
```python
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

clustering_data = combined_data[['Latitude', 'Longitude', 'pH', 'N (%)', 'P (%)', 'K (%)']]
scaler = StandardScaler()
clustering_data_scaled = scaler.fit_transform(clustering_data)
kmeans = KMeans(n_clusters=3, random_state=0)
combined_data['Cluster'] = kmeans.fit_predict(clustering_data_scaled)
```

### 4. Soil pH Prediction
Using a Linear Regression model to predict pH levels:
```python
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X = combined_data[['Latitude', 'Longitude', 'N (%)', 'P (%)', 'K (%)']]
y = combined_data['pH']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

regressor = LinearRegression()
regressor.fit(X_train, y_train)
y_pred = regressor.predict(X_test)
print(f"Mean Squared Error: {mean_squared_error(y_test, y_pred)}")
```

### 5. Generating an Interactive Map
The script creates a map with markers showing soil suitability.
```python
import folium

def classify_for_cultivation(value):
    return "Suitable for Cultivation" if value in [0, 1, 2] else "Not Suitable"

combined_data['Cultivation Suitability'] = combined_data['Cluster'].apply(classify_for_cultivation)
m = folium.Map(location=[35.8322, 50.9917], zoom_start=12)

for _, row in combined_data.iterrows():
    folium.Marker(
        [row['Latitude'], row['Longitude']],
        popup=f"Suitability: {row['Cultivation Suitability']}\n pH: {row['pH']}"
    ).add_to(m)

m.save("apple_growing_map_with_suitability.html")
```

### 6. Running the Project
To execute the script:
```sh
python soil_analysis.py
```
The output will include:
- Processed data in the terminal.
- Clustered map visualization.
- An HTML map (`apple_growing_map_with_suitability.html`) saved in the project directory.

---

## Future Improvements
- **Integration with a Web App**: Allow users to upload Excel files and generate analysis automatically.
- **Expanded Machine Learning Models**: Improve prediction accuracy with advanced algorithms.
- **Enhanced Visualization**: Use web-based dashboards to display soil data interactively.

---

## License
This project is open-source under the MIT License.

---

## Author
Developed by Mahdieh Kazemi. Feel free to contribute or reach out with suggestions!

---

## Contributions
Contributions are welcome! If you find issues or have suggestions, open a pull request or submit an issue on GitHub.

```sh
git clone https://github.com/your-repo/soil-mapping.git
cd soil-mapping
```

Happy coding! 🚀

