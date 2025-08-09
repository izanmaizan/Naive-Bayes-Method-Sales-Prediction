import joblib

# Load file model
model = joblib.load("naive_bayes_car_sales.sav")
encoders = joblib.load("encoders_car_sales.sav")

print("=== Model Naive Bayes ===")
print("Tipe model:", type(model))
print("\nParameter model:", model.get_params())

if hasattr(model, "classes_"):
    print("\nKelas:", model.classes_)
if hasattr(model, "class_count_"):
    print("\nJumlah data tiap kelas:", model.class_count_)
if hasattr(model, "class_prior_"):
    print("\nProbabilitas awal kelas:", model.class_prior_)
if hasattr(model, "theta_"):
    print("\nRata-rata fitur per kelas:\n", model.theta_)
if hasattr(model, "sigma_"):
    print("\nVariansi fitur per kelas:\n", model.sigma_)

print("\n=== Encoders ===")
for key, encoder in encoders.items():
    print(f"\nEncoder untuk '{key}'")
    if hasattr(encoder, "classes_"):
        print("Kelas:", encoder.classes_)
