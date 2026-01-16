import os

DATASET_PATH = "../dataset/images"

total_images = 0
class_count = 0

print("Verifying dataset...\n")

for person_name in os.listdir(DATASET_PATH):

    person_path = os.path.join(DATASET_PATH, person_name)

    if not os.path.isdir(person_path):
        continue

    images = []

    for file_name in os.listdir(person_path):
        if file_name.lower().endswith((".jpg", ".jpeg", ".png")):
            images.append(file_name)

    num_images = len(images)

    print("Class:", person_name)
    print("Number of images:", num_images)
    print("-" * 30)

    total_images += num_images
    class_count += 1

print("\nSummary")
print("Total classes:", class_count)
print("Total images:", total_images)

if class_count > 0:
    print("Average images per class:", total_images // class_count)
