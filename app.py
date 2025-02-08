import pandas as pd
import openpyxl as openpyxl
import numpy as np
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import folium



#بارگذاری داده های خاک
soil_data = pd.read_excel('soil_data_apple.xlsx')

#بارگذاری داده های محیط
location_data =pd.read_excel('location_data_apple.xlsx')
# نمایش اولین چند ردیف از داده‌ها
print(soil_data.head())
print(location_data.head())

#بررسی داده ها برای مقادیر گمشده
print(soil_data.isnull())
print(location_data.isnull())

#بررسی اطلاعات عمومی
print(soil_data.describe())
print(location_data.describe())

# فرض می‌کنیم که داده‌های خاک و محیطی با ستون‌هایی مثل latitude و longitude هم‌راستا هستند
# ادغام دو دیتافریم بر اساس موقعیت جغرافیایی
combined_data = pd.merge(soil_data, location_data, on=['Latitude', 'Longitude'])
print(combined_data.head())

#آمارتوصیفی داده ها
descriptive_stats = combined_data.describe()
print(descriptive_stats)


# فقط ستون‌های عددی رو انتخاب میکنیم
numeric_data = combined_data.select_dtypes(include=[np.number])

# محاسبه همبستگی
corr_matrix = numeric_data.corr()

# نمایش همبستگی
print(corr_matrix)

#انتخاب ویژگی ها برای خوشه بندی برای پیدا کردن نقاط جغرافیایی دارای خصوصیات مشابه از نظر شرایط خاک
clustering_data = combined_data[['Latitude', 'Longitude', 'pH', 'N (%)', 'P (%)', 'K (%)']]

# نرمال‌سازی داده‌ها (مقیاس داده‌ها برای الگوریتم‌های خوشه‌بندی مهم است)
scaler = StandardScaler()
clustering_data_scaler = scaler.fit_transform(clustering_data)

# اجرای الگوریتم KMeans
kmeans = KMeans(n_clusters=3, random_state=0)  # تعداد خوشه‌ها را می‌توان تغییر داد
combined_data['Cluster'] = kmeans.fit_predict(clustering_data_scaler)

# نمایش نتایج خوشه‌بندی
plt.scatter(combined_data['Longitude'], combined_data['Latitude'], c=combined_data['Cluster'], cmap='viridis')
plt.title('Clustering of Locations for Apple Trees')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()


#پیش بینی مناطق مناسب کشت درخت سیب
# انتخاب ویژگی‌ها و هدف
X = combined_data[['Latitude', 'Longitude', 'N (%)', 'P (%)', 'K (%)']]  # ویژگی‌های ورودی
y = combined_data['pH']  # ویژگی هدف (pH خاک)

#تقسیم داده ها به دو مجموعه ی تستی و آموزشی
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

#آموزش مدل رگرسیون
regressor = LinearRegression()
regressor.fit(X_train, y_train)

#انجام پیش بینی
y_pred = regressor.predict(X_test)

#ارزیابی مدل 
mse = mean_squared_error( y_test, y_pred)
print(f"Mean Squared Error: {mse}")

# پیش‌بینی pH برای تمام مناطق
combined_data['Predicted pH'] = regressor.predict(X)

# نمایش نتایج
plt.scatter(combined_data['Longitude'], combined_data['Latitude'], c=combined_data['Predicted pH'], cmap='coolwarm')
plt.colorbar(label='Predicted pH')
plt.title('Predicted pH for Apple Tree Planting Locations')
plt.xlabel('Longitude')
plt.ylabel('Latitude')
plt.show()

# اضافه کردن یک ستون برای وضعیت مناسب یا نامناسب برای کشت
# فرض کنیم که پیش‌بینی‌ها یا نتایج خوشه‌بندی در ستون 'Cluster' یا 'Prediction' ذخیره شده باشد

def classify_for_cultivation(value):
    if value in [0, 1, 2]:  # به عنوان مثال، خوشه‌ها یا پیش‌بینی‌های مناسب برای کشت
        return "مناسب برای کشت"
    else:  # خوشه‌ها یا پیش‌بینی‌های نامناسب
        return "نامناسب برای کشت"

# فرض می‌کنیم ستون پیش‌بینی یا خوشه بندی 'Cluster' یا 'Prediction' موجود است
combined_data['Cultivation Suitability'] = combined_data['Cluster'].apply(classify_for_cultivation)

# ایجاد یک نقشه با مرکز کرج
m = folium.Map(location=[35.8322, 50.9917], zoom_start=12)

# اضافه کردن نقاط (Latitude, Longitude) به نقشه
for index, row in combined_data.iterrows():
    folium.Marker(
        [row['Latitude'], row['Longitude']],
        popup=f"این نقطه: {row['Cultivation Suitability']}\n مقدار پی اچ: {row['pH']}%"  # نام ستون رطوبت را بررسی کنید
    ).add_to(m)

# ذخیره نقشه در یک فایل HTML
m.save("apple_growing_map_with_suitability.html")