import pickle
import numpy as np

# HEADER SISTEM PREDIKSI
print("="*60)
print("🚗 SISTEM PREDIKSI PEMBELIAN MOBIL NAIVE BAYES 🚗")
print("="*60)
print("📋 Sistem ini akan memprediksi apakah customer akan membeli mobil")
print("📊 berdasarkan data: Gender, Usia, Gaji, dan Kepuasan")
print("⚡ Silakan masukkan data customer yang akan diprediksi")
print("="*60)

#########################################################
#           INPUT DATA CUSTOMER                        #
#########################################################
# TAHAP 1: INPUT MANUAL DATA CUSTOMER BARU
# Data customer yang akan diprediksi apakah akan membeli mobil atau tidak

print("\n=== INPUT DATA CUSTOMER ===")

# Input Gender (Male/Female)
while True:
    gender = input("Masukkan Gender (Male/Female): ").strip()
    if gender.lower() in ['male', 'female']:
        gender = gender.capitalize()  # Male atau Female
        break
    else:
        print("❌ Input tidak valid! Masukkan 'Male' atau 'Female'")

# Input Usia customer
while True:
    try:
        age = int(input("Masukkan Usia (tahun): "))
        if 15 <= age <= 100:
            break
        else:
            print("❌ Usia harus antara 15-100 tahun!")
    except ValueError:
        print("❌ Input tidak valid! Masukkan angka untuk usia")

# Input Estimasi gaji tahunan
while True:
    try:
        estimated_salary = int(input("Masukkan Estimasi Gaji Tahunan ($): "))
        if estimated_salary >= 0:
            break
        else:
            print("❌ Gaji tidak boleh negatif!")
    except ValueError:
        print("❌ Input tidak valid! Masukkan angka untuk gaji")

# Input Kepuasan (yes/no)
while True:
    satisfied = input("Apakah customer puas dengan layanan? (yes/no): ").strip().lower()
    if satisfied in ['yes', 'no']:
        break
    else:
        print("❌ Input tidak valid! Masukkan 'yes' atau 'no'")

#########################################################
#                AKHIR DATA                             #
#########################################################

# TAHAP 2: MENAMPILKAN DATA INPUT YANG TELAH DIMASUKKAN
# Konfirmasi data customer yang akan diprediksi
print("\n" + "="*30)
print("📋 RINGKASAN DATA CUSTOMER:")
print("="*30)
print(f"👤 Gender: {gender}")
print(f"🎂 Age: {age} tahun")
print(f"💰 Estimated Salary: ${estimated_salary:,}")
print(f"😊 Satisfied: {satisfied}")
print("="*30)

# TAHAP 3: KONFIRMASI DATA DAN MEMUAT ENCODER
# Konfirmasi sebelum melakukan prediksi
print("\n⏳ Memproses data...")
input("📍 Tekan ENTER untuk melanjutkan prediksi...")

# Memuat encoder yang digunakan saat training untuk mengubah data kategorikal
print("\n🔄 Loading encoders...")
try:
    encoders = pickle.load(open('encoders_car_sales.sav', 'rb'))
    le_gender = encoders['gender_encoder']
    le_satisfied = encoders['satisfied_encoder']
    print("✅ Encoders berhasil dimuat")
except FileNotFoundError:
    print("❌ File encoders tidak ditemukan! Pastikan sudah menjalankan main.py")
    exit()
except Exception as e:
    print(f"❌ Error memuat encoders: {e}")
    exit()

# TAHAP 4: ENCODING DATA INPUT
# Mengubah data kategorikal input menjadi angka menggunakan encoder yang sama dengan training
print("\n🔢 Encoding input data...")

# Encode gender: Male/Female -> 0/1
try:
    gender_encoded = le_gender.transform([gender])[0]
    print(f"✅ Gender encoded: {gender} -> {gender_encoded}")
except ValueError:
    print(f"❌ Gender '{gender}' tidak dikenali, menggunakan default (0)")
    gender_encoded = 0

# Encode satisfied: yes/no -> 0/1  
try:
    satisfied_encoded = le_satisfied.transform([satisfied])[0]
    print(f"✅ Satisfied encoded: {satisfied} -> {satisfied_encoded}")
except ValueError:
    print(f"❌ Satisfied '{satisfied}' tidak dikenali, menggunakan default (0)")
    satisfied_encoded = 0

# TAHAP 5: MEMBUAT ARRAY INPUT
# Menyusun data input dalam format yang sama dengan data training
data2 = np.array([gender_encoded, age, estimated_salary, satisfied_encoded]).reshape(1,-1)
print(f"\n🔢 Input array untuk model:")
print(f"   [Gender, Age, Salary, Satisfied]")
print(f"   {data2[0]}")
print(f"   [{gender_encoded}, {age}, {estimated_salary}, {satisfied_encoded}]")

# TAHAP 6: MEMUAT MODEL DAN MELAKUKAN PREDIKSI
# Memuat model yang sudah dilatih dan melakukan prediksi
print("\n🤖 Loading model dan melakukan prediksi...")
try:
    filename = 'naive_bayes_car_sales.sav'
    clf = pickle.load(open(filename, 'rb'))
    print("✅ Model berhasil dimuat")
    
    # Melakukan prediksi: 0 = Tidak Beli, 1 = Beli
    result = clf.predict(data2)
    print(f"🔍 Hasil prediksi (0=Tidak Beli, 1=Beli): {result[0]}")
    
except FileNotFoundError:
    print("❌ File model tidak ditemukan! Pastikan sudah menjalankan main.py")
    exit()
except Exception as e:
    print(f"❌ Error memuat model: {e}")
    exit()

# TAHAP 7: MENGHITUNG PROBABILITAS
# Menghitung tingkat keyakinan model terhadap prediksi yang dibuat
proba = clf.predict_proba(data2)
prob_tidak_beli = proba[0][0] * 100  # Probabilitas tidak beli dalam persen
prob_beli = proba[0][1] * 100        # Probabilitas beli dalam persen

print(f"\n📊 Distribusi Probabilitas:")
print(f"   🔴 Tidak Beli: {prob_tidak_beli:.2f}%")
print(f"   🟢 Beli      : {prob_beli:.2f}%")

# Visualisasi sederhana dengan bar
bar_tidak_beli = "█" * int(prob_tidak_beli // 5)
bar_beli = "█" * int(prob_beli // 5)
print(f"\n📈 Visualisasi:")
print(f"   🔴 Tidak Beli: {bar_tidak_beli} {prob_tidak_beli:.1f}%")
print(f"   🟢 Beli      : {bar_beli} {prob_beli:.1f}%")

# TAHAP 8: MENAMPILKAN HASIL AKHIR
# Interpretasi hasil prediksi dalam format yang mudah dipahami
print("\n" + "="*50)
print("🎯 HASIL PREDIKSI AKHIR:")
print("="*50)
if(result[0]==1):
    print("✅ CUSTOMER AKAN MEMBELI MOBIL")
    print(f"🔥 Confidence: {prob_beli:.2f}%")
    if prob_beli > 80:
        print("💡 Rekomendasi: Prioritas Tinggi - Follow up segera!")
    elif prob_beli > 60:
        print("💡 Rekomendasi: Prioritas Sedang - Customer potensial")
    else:
        print("💡 Rekomendasi: Prioritas Rendah - Perlu nurturing")
else:
    print("❌ CUSTOMER TIDAK AKAN MEMBELI MOBIL")
    print(f"📊 Confidence: {prob_tidak_beli:.2f}%")
    if prob_tidak_beli > 80:
        print("💡 Rekomendasi: Customer tidak tertarik saat ini")
    else:
        print("💡 Rekomendasi: Customer ragu-ragu, berikan informasi lebih")

print("="*50)