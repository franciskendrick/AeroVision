import csv
import os

# Define the directory containing your CSV files
csv_dir = "CNN 4/CSVs"
output_file = "CNN 4/combined_data.csv"

# Collect all CSV files in the directory
csv_files = [f for f in os.listdir(csv_dir) if f.endswith('.csv')]

# Combine CSVs
with open(output_file, mode='w', newline='') as outfile:
    writer = None
    for filename in csv_files:
        with open(os.path.join(csv_dir, filename), mode='r') as infile:
            reader = csv.reader(infile)
            header = next(reader)
            if writer is None:
                writer = csv.writer(outfile)
                writer.writerow(header)  # Write header once
            for row in reader:
                writer.writerow(row)

print(f"Combined {len(csv_files)} CSVs into {output_file}")
