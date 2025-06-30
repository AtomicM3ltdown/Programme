import pandas as pd
import re

# Read the CSV file into a pandas DataFrame
df = pd.read_csv('C:/Users/jensc/Desktop/Timestamp.csv', header=None, names=['data']).reset_index()

# Initialize empty DataFrames for pairs and unmatched entries
pairs_df = pd.DataFrame(columns=['Arbeitsbeginn', 'Arbeitsende', 'Beginn Time', 'End Time', 'Time Difference'])
unmatched_df = pd.DataFrame(columns=['Arbeitsbeginn', 'Beginn Time'])

# Initialize the sum of time differences
total_time_diff = pd.Timedelta(seconds=0)

# Iterate over each row in the DataFrame
for index, row in df.iterrows():
    entry = row['data']  # Replace 'your_column_name' with the actual column name

    # Check if the entry contains 'Arbeitsbeginn'
    if 'Arbeitsbeginn' in entry:
        # Extract the date and time from the entry
        date = entry[15:25]
        match = re.search(r'(\d{2}:\d{2})', entry)
        if match:
            time = match.group(1)
        else:
            print(f"⚠️ Keine gültige Uhrzeit in: {entry}")
            continue

        # Check if there is a corresponding 'Arbeitsende' entry with the same date
        end_entry = df[(df['data'].str.contains('Arbeitsende')) & (df['data'].str.contains(date))]
        if not end_entry.empty:
            # Extract the end time from the 'Arbeitsende' entry
            end_time = end_entry.iloc[0]['data'][26:31]
            # Calculate the time difference
            print(time)
            end_entry_str = end_entry.iloc[0]['data']
            match_end = re.search(r'(\d{2}:\d{2})', end_entry_str)
            if match_end:
                end_time = match_end.group(1)
            else:
                print(f"⚠️ Keine gültige Endzeit in: {end_entry_str}")
                continue

            begin_time = pd.to_datetime(time, format='%H:%M')
            end_time = pd.to_datetime(end_time, format='%H:%M')
            time_diff = end_time - begin_time

            # Add the pair to pairs_df with date, time, end time, and time difference
            pairs_df = pd.concat([pairs_df, pd.DataFrame({'Arbeitsbeginn': [entry], 'Arbeitsende': [end_entry.iloc[0]['data']],
                                                          'Beginn Time': [time], 'End Time': [end_time],
                                                          'Time Difference': [time_diff]})], ignore_index=True)

            # Add the time difference to the total sum
            total_time_diff += time_diff
        else:
            # If there is no matching 'Arbeitsende' entry, add it to the unmatched entries DataFrame with date and time
            unmatched_df = pd.concat([unmatched_df, pd.DataFrame({'Arbeitsbeginn': [entry], 'Beginn Time': [time]})], ignore_index=True)

# Convert total_time_diff to hours
total_hours = total_time_diff.total_seconds() // 3600

# Print the resulting DataFrames
print("Pairs:")
print(pairs_df)
print("\nUnmatched:")
print(unmatched_df)
# Print the total sum of time differences in hours
print("\nTotal Time Difference (in hours):")
print(f"{total_hours} hours")
print(total_time_diff)






