import pandas as pd

# Load the CSV file
file_path = '/mnt/data/miRCancer.csv'
df = pd.read_csv(file_path, header=None)

# Modify the third column by capitalizing the first letter after the third comma
df[3] = df[3].str.capitalize()

# Save the modified DataFrame to a new CSV file
output_path = '/mnt/data/modified_miRCancer.csv'
df.to_csv(output_path, index=False, header=False)

# Display the first few rows of the modified dataframe to the user
import ace_tools as tools; tools.display_dataframe_to_user(name="Modified miRCancer", dataframe=df)

output_path
