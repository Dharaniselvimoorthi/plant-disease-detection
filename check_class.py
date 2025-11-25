from tensorflow.keras.models import load_model

model = load_model(r"C:\plant_disease_detection\model.h5")

print("Model Attributes:")
print([a for a in dir(model) if "class" in a.lower()])

if hasattr(model, "class_indices"):
    print("class_indices:", model.class_indices)
elif hasattr(model, "classes"):
    print("classes:", model.classes)
else:
    print("❌ No class info inside model. Need dataset folder order.")
