import pandas as pd

# 1. TAHAP IMPORT DATA
# Membaca file Excel yang berisi data penjualan mobil
data_excel = pd.read_excel('sale.xlsx')  
# Menghapus baris yang memiliki nilai kosong/missing value
data_excel = data_excel.dropna()
print("\nImport Data " + str(data_excel.shape))
print(data_excel.head())

# 2. TAHAP PREPROCESSING - CLEANING NAMA KOLOM
# Menghilangkan spasi di awal/akhir nama kolom dan spasi di tengah
data_excel.columns = data_excel.columns.str.strip().str.replace(' ', '')
print("\n Kolom setelah cleaning:")
print(list(data_excel.columns))

import numpy as np
from sklearn.preprocessing import LabelEncoder

# 3. TAHAP ENCODING DATA KATEGORIKAL
# Mengubah data kategorikal (text) menjadi angka agar bisa diproses algoritma
print("\n Encoding data kategorikal...")

# Encode kolom Gender: Male/Female menjadi 0/1
le_gender = LabelEncoder()
data_excel['Gender'] = le_gender.fit_transform(data_excel['Gender'])
print("Gender classes:", le_gender.classes_)

# Encode kolom satisfied: yes/no menjadi 0/1
le_satisfied = LabelEncoder()
data_excel['satisfied'] = le_satisfied.fit_transform(data_excel['satisfied'])
print("Satisfied classes:", le_satisfied.classes_)

# 4. TAHAP SELEKSI FITUR
# Memilih fitur yang relevan untuk prediksi (mengabaikan UserID karena tidak berpengaruh)
selected_features = ['Gender', 'Age', 'EstimatedSalary', 'satisfied']
feature_data = data_excel[selected_features + ['Purchased']]
print("\n Data setelah seleksi fitur:")
print(feature_data.head())

# 5. TAHAP KONVERSI KE NUMPY ARRAY
# Mengubah DataFrame pandas menjadi numpy array untuk kompatibilitas sklearn
data_np = feature_data.to_numpy()
print("\n Data setelah diubah ke numpy array" + str(data_np.shape))
print(data_np)

# 6. TAHAP PEMISAHAN FITUR DAN TARGET
# Memisahkan data input (X) dan target/label (y)
n_fea = 5  # 4 fitur + 1 target
data = data_np[:, range(n_fea-1)]  # Ambil kolom 0-3 sebagai fitur input
target = data_np[:, n_fea-1]       # Ambil kolom ke-4 sebagai target
print("\n data feature" + str(data.shape))
print(data)
print("\n target" + str(target.shape))
print(target)

# 7. TAHAP PEMBAGIAN DATA TRAINING DAN TESTING
# Membagi data menjadi 75% training dan 25% testing
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.25, random_state=42)
print("\n data inputan train " + str(X_train.shape))
print(X_train)
print("\n data target train " + str(y_train.shape))
print(y_train)
print("\n data inputan testing " + str(X_test.shape))
print(X_test)
print("\n data target testing " + str(y_test.shape))
print(y_test)

# 8. TAHAP INISIALISASI MODEL
# Membuat objek model Gaussian Naive Bayes
from sklearn.naive_bayes import GaussianNB
clf = GaussianNB()

# 9. TAHAP TRAINING DAN EVALUASI MODEL
# Melatih model dengan data training dan menguji dengan data testing
from sklearn.metrics import accuracy_score
clf.fit(X_train, y_train)           # Proses training/pelatihan model
y_pred = clf.predict(X_test)        # Prediksi pada data testing
akurasi = accuracy_score(y_test, y_pred)  # Hitung akurasi
print("\n Akurasi: " + str(akurasi*100) + "%")

# 10. TAHAP PENYIMPANAN MODEL
# Menyimpan model yang sudah dilatih ke file untuk digunakan nanti
import pickle
filename = 'naive_bayes_car_sales.sav'
pickle.dump(clf, open(filename, 'wb'))

# 11. TAHAP PENYIMPANAN ENCODER
# Menyimpan encoder untuk digunakan saat prediksi data baru
encoders = {
    'gender_encoder': le_gender,
    'satisfied_encoder': le_satisfied
}
encoder_filename = 'encoders_car_sales.sav'
pickle.dump(encoders, open(encoder_filename, 'wb'))