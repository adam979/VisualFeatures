import os
import csv
import shutil
from sklearn.model_selection import train_test_split

# Set the paths for the dataset directories
root_dir = "C:\\Users\\hassa\\Desktop\\SWIN\\archive"
train_dir = os.path.join(root_dir, "train")
val_dir = os.path.join(root_dir, "val")
test_dir = os.path.join(root_dir, "test")

# Create the train, validation, and test directories if they don't exist
os.makedirs(train_dir, exist_ok=True)
os.makedirs(val_dir, exist_ok=True)
os.makedirs(test_dir, exist_ok=True)

# Set the paths to the downloaded dataset and CSV files
dataset_dir = os.path.join(root_dir, "images")
projections_file = os.path.join(root_dir, "indiana_projections.csv")
reports_file = os.path.join(root_dir, "indiana_reports.csv")

# Read the projection information from the CSV file
projections = {}
with open(projections_file, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        uid = row["uid"]
        filename = row["filename"]
        projection = row["projection"]
        projections[filename] = projection

# Dictionary to store the UID and the concatenated report
reports = {}

with open(reports_file, "r") as file:
    reader = csv.DictReader(file)
    for row in reader:
        uid = row["uid"]
        findings = row["findings"]
        impression = row["impression"]

        # Concatenate the findings and impressions to form the report
        report = f"Findings: {findings}\nImpressions: {impression}"

        # Store the UID and the report in the dictionary
        reports[uid] = report

# Get the list of UIDs
uids = list(reports.keys())

# Split the UIDs into train, validation, and test sets
train_uids, val_test_uids = train_test_split(uids, test_size=0.2, random_state=42)
val_uids, test_uids = train_test_split(val_test_uids, test_size=0.5, random_state=42)

print("Number of train UIDs:", len(train_uids))
print("Number of validation UIDs:", len(val_uids))
print("Number of test UIDs:", len(test_uids))

# Iterate over the files in the dataset directory
for filename in os.listdir(dataset_dir):
    # Extract the UID from the filename
    uid = filename.split("_")[0]
    print("UID:", uid)

    # Check if the UID has null or empty values in impressions or findings
    if (
        uid not in reports
        or reports[uid] is None
        or not reports[uid].strip()  # Modified condition
    ):
        print(f"Skipping UID: {uid}")
        if uid in reports:
            print(f"Report: {reports[uid]}")
        continue

    print("Copying image", filename)

    # Get the projection from the projections dictionary
    projection = projections.get(filename, "")
    print("Projection:", projection)

    # Determine whether the image belongs to training, validation, or test data
    if uid in train_uids:
        dest_dir = os.path.join(train_dir, uid)
    elif uid in val_uids:
        dest_dir = os.path.join(val_dir, uid)
    else:
        dest_dir = os.path.join(test_dir, uid)

    print("Destination directory:", dest_dir)

    # Create the destination directory if it doesn't exist
    os.makedirs(dest_dir, exist_ok=True)

    # Copy the image file to the destination directory
    src_path = os.path.join(dataset_dir, filename)
    dest_path = os.path.join(dest_dir, filename)

    try:
        shutil.copy2(src_path, dest_path)
        print("Copied image", filename, "to", dest_dir)
    except Exception as e:
        print("Error copying image", filename, ":", str(e))

# Verify the contents of the train, validation, and test directories
print(f"\nNumber of files in train directory: {len(os.listdir(train_dir))}")
print(f"Number of files in validation directory: {len(os.listdir(val_dir))}")
print(f"Number of files in test directory: {len(os.listdir(test_dir))}")
