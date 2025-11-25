from tensorflow.keras.preprocessing.image import ImageDataGenerator
import json

DATASET_PATH = "PlantVillage"
img_size = 128
batch_size = 32

train_datagen = ImageDataGenerator(rescale=1/255, validation_split=0.2)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=(img_size, img_size),
    batch_size=batch_size,
    class_mode="categorical",
    subset="training"
)

label_map = train_data.class_indices
label_map = {v: k for k, v in label_map.items()}

with open("label_map.json", "w") as f:
    json.dump(label_map, f)

print("label_map.json CREATED SUCCESSFULLY ✔️")
