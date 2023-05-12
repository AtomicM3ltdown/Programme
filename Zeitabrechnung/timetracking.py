import pandas as pd
import re

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:/Users/jensc/Desktop/Timestamp.csv', header=None, names=['data']).reset_index()

# Initialize empty DataFrames for pairs and unmatched entries
pairs_df = pd.DataFrame(columns=['Arbeitsbeginn', 'Arbeitsende'])
unmatched_df = pd.DataFrame(columns=['Arbeitsbeginn'])

# Initialize variables to keep track of the previous 'Arbeitsbeginn' and its date
prev_begin = None
prev_begin_date = None

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    entry = row['data']  # Replace 'your_column_name' with the actual column name

    # Check if the entry contains 'Arbeitsbeginn'
    if 'Arbeitsbeginn' in entry:
        # Extract the date and time from the entry
        date = entry[15:25]
        time = entry[-5:]
        print(date)

        # Check if there was a previous 'Arbeitsbeginn'
        if prev_begin is not None:
            # Check if the previous 'Arbeitsbeginn' and the current entry have the same date
            if prev_begin_date == date:
                # Check if the previous 'Arbeitsbeginn' already has a matching 'Arbeitsende'
                if prev_begin not in pairs_df['Arbeitsbeginn'].values:
                    # Add the pair to pairs_df
                    pairs_df = pd.concat([pairs_df, pd.DataFrame({'Arbeitsbeginn': [prev_begin], 'Arbeitsende': [entry]})], ignore_index=True)
                else:
                    # If the previous 'Arbeitsbeginn' already has a matching 'Arbeitsende',
                    # add it to the unmatched entries DataFrame
                    unmatched_df = pd.concat([unmatched_df, pd.DataFrame({'Arbeitsbeginn': [prev_begin]})], ignore_index=True)

        # Update the previous 'Arbeitsbeginn' and its date
        prev_begin = entry
        prev_begin_date = date

    # Check if the entry contains 'Arbeitsende'
    elif 'Arbeitsende' in entry:
        # Extract the date and time from the entry
        date = entry[15:25]
        time = entry[-5:]

        # Check if the previous 'Arbeitsbeginn' has a matching 'Arbeitsende'
        if prev_begin_date == date:
            # Check if the previous 'Arbeitsbeginn' already has a matching 'Arbeitsende'
            if prev_begin not in pairs_df['Arbeitsbeginn'].values:
                # Add the pair to pairs_df
                pairs_df = pd.concat([pairs_df, pd.DataFrame({'Arbeitsbeginn': [prev_begin], 'Arbeitsende': [entry]})], ignore_index=True)
            else:
                # If the previous 'Arbeitsbeginn' already has a matching 'Arbeitsende',
                # add it to the unmatched entries DataFrame
                unmatched_df = pd.concat([unmatched_df, pd.DataFrame({'Arbeitsbeginn': [prev_begin]})], ignore_index=True)

# Print the resulting DataFrames
print("Pairs:")
print(pairs_df)
print("\nUnmatched:")
print(unmatched_df)






