import os
import yaml
import matplotlib.pyplot as plt
import seaborn as sns

# 1. Load class names from your data.yaml
yaml_path = r"C:\Users\shafa\Downloads\IDS\IDS\Under_Water_trash_plastic_detection\Underwater_garbage\data.yaml"
with open(yaml_path, 'r') as file:
    data_yaml = yaml.safe_load(file)
class_names = data_yaml['names']

# 2. Count the occurrences of each class in the training labels
label_dir = r"C:\Users\shafa\Downloads\IDS\IDS\Under_Water_trash_plastic_detection\Underwater_garbage\train\labels"
class_counts = {name: 0 for name in class_names}

for label_file in os.listdir(label_dir):
    if label_file.endswith('.txt'):
        with open(os.path.join(label_dir, label_file), 'r') as f:
            lines = f.readlines()
            for line in lines:
                class_id = int(line.split()[0])
                class_counts[class_names[class_id]] += 1

# 3. Generate the Visualization (Histogram/Bar Chart)
plt.figure(figsize=(12, 6))
sns.barplot(x=list(class_counts.values()), y=list(class_counts.keys()), palette="viridis")
plt.title('Distribution of Ocean Waste Classes in Training Data')
plt.xlabel('Number of Instances')
plt.ylabel('Trash Category')
plt.show() 