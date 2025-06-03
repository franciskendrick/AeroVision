import pandas as pd
import numpy as np

# Load the CSV
df = pd.read_csv("D:\Programming Projects\Repositories\AeroVision\LSTM 2\data.csv")

# List of landmark indices (same ones you're using)
landmarks = [11, 12, 13, 14, 15, 16, 23, 24]

# Remove z-coordinates and the last 4 columns (angle data)
columns_to_keep = ['class']
for lm in landmarks:
    columns_to_keep.extend([f"{lm}x", f"{lm}y"])
df = df[columns_to_keep]

# Prepare a new DataFrame to store normalized data
normalized_rows = []

for _, row in df.iterrows():
    new_row = {'class': row['class']}
    
    # Get shoulder coordinates
    x11, y11 = row['11x'], row['11y']
    x12, y12 = row['12x'], row['12y']
    
    # Compute center (cx, cy) between shoulders
    cx = (x11 + x12) / 2
    cy = (y11 + y12) / 2
    
    # Compute shoulder distance (scale)
    shoulder_dist = np.sqrt((x11 - x12)**2 + (y11 - y12)**2)
    
    # Avoid division by zero
    if shoulder_dist == 0:
        shoulder_dist = 1e-6
    
    # Normalize all (x, y) coordinates
    for lm in landmarks:
        x = row[f"{lm}x"]
        y = row[f"{lm}y"]
        x_norm = (x - cx) / shoulder_dist
        y_norm = (y - cy) / shoulder_dist
        new_row[f"{lm}x"] = x_norm
        new_row[f"{lm}y"] = y_norm
    
    normalized_rows.append(new_row)

# Save to new CSV
normalized_df = pd.DataFrame(normalized_rows)
normalized_df.to_csv("normalized_output.csv", index=False)

print("✅ Normalization complete. Saved as 'normalized_output.csv'")
