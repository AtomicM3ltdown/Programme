import re
import pandas as pd

# Prompt the user to enter the file path
file_path = "C:/Users/jensc/Desktop/Timestamp.csv"

try:
    # Initialize empty lists to store the extracted data
    arbeitsbeginn_lines = []
    arbeitsende_lines = []

    with open(file_path, 'r', encoding='cp1252') as file:
        # Read the contents of the file
        contents = file.readlines()

        # Iterate over each line in the file
        for line in contents:
            # Use regular expressions to extract the date and time
            match = re.search(r'\d{4}-\d{2}-\d{2} \| \d{2}:\d{2}', line)

            if match:
                date_time = match.group()
                date, time = date_time.split(' | ')

                # Check if it is 'Arbeitsbeginn' or 'Arbeitsende'
                if 'Arbeitsbeginn' in line:
                    arbeitsbeginn_lines.append({'Arbeitsbeginn': line.strip(), 'Date_start': date, 'Time_start': time})
                elif 'Arbeitsende' in line:
                    arbeitsende_lines.append({'Arbeitsende': line.strip(), 'Date_end': date, 'Time_end': time})

    # Create DataFrames from the extracted data
    df_arbeitsbeginn = pd.DataFrame(arbeitsbeginn_lines)
    df_arbeitsende = pd.DataFrame(arbeitsende_lines)

    # Remove duplicates from the DataFrames based on column 1 (Date_start)
    date_index = pd.date_range('1/12/2022', periods=365, freq='D')
    df_arbeitsbeginn = df_arbeitsbeginn.sort_values('Time_start').drop_duplicates(subset='Date_start', keep='first')
    df_arbeitsbeginn = df_arbeitsbeginn.sort_values('Date_start')
    df_arbeitsbeginn.dropna()
    df_arbeitsbeginn.reindex(date_index)
    print(df_arbeitsbeginn)

    df_arbeitsende = df_arbeitsende.sort_values('Time_end').drop_duplicates(subset='Date_end', keep='last')
    df_arbeitsende = df_arbeitsende.sort_values('Date_end')
    df_arbeitsende.dropna()
    df_arbeitsende.reindex(date_index)
    df_arbeitsbeginn.reindex_like(df_arbeitsende)
    print(df_arbeitsende)
    # Merge the DataFrames on the 'Date' and 'Time' columns
    df_merged = pd.concat([df_arbeitsbeginn, df_arbeitsende], axis=1)
    #print(df_merged)

    # Sort the DataFrame by column 1 and 4 to have the same date in each row
    df_merged = df_merged.sort_values(['Date_start', 'Date_end']).reset_index(drop=True)

    # Print the modified DataFrame
    #print(df_merged)

except FileNotFoundError:
    print('File not found. Please enter a valid file path.')







