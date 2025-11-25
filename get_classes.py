import os

DATASET_PATH = r"C:\plant_disease_detection\plantVillage"   # 👈 change this to YOUR dataset main folder

classes = sorted(os.listdir(DATASET_PATH))

print("Total Classes:", len(classes))
print("Class Names in Correct Order:")
for c in classes:
    print(c)
