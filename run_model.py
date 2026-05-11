from ultralytics import YOLO

# 1. Load your newly trained brain
model = YOLO(r"C:\Users\shafa\Downloads\IDS\IDS\Under_Water_trash_plastic_detection\custom model\ocean_waste_best.pt")

# 2. Tell it what to look at (your test images) and save the results
results = model.predict(source=r"C:\Users\shafa\Downloads\IDS\IDS\Under_Water_trash_plastic_detection\Underwater_garbage\test\images", save=True)