import os
import pandas as pd
import shutil
import glob

# specify the path to the folder containing the files
folder_path = r'D:\Nextcloud\Vulkameter\Rohdaten'

# load the Excel sheet containing the data
df = pd.read_excel(r'D:\Nextcloud\Vulkameter/Average.xlsx', sheet_name='Tabelle2')

# iterate over the files in the folder
for filename in os.listdir(folder_path):
    # check if the file is a regular file
    if os.path.isfile(os.path.join(folder_path, filename)):
        # extract the filename without extension
        file_basename = os.path.splitext(filename)[0]
        print(file_basename)
        # search for a match in the Excel sheet
        match = df[(df['jlph_t'].str.contains('JC') | df['jl_t'].str.contains('JC')) & df['Rdatafile'].eq(file_basename)]
        if not match.empty:
            # extract the content of the column 'jlph_t' or 'jl_t' in the same row containing"JC"
            column_name = 'jlph_t' if 'JC' in match['jlph_t'].iloc[0] else 'jl_t'
            content = match[column_name].iloc[0]
            print(f"The file {filename} matches the data in the Excel sheet. Content: {content}")
            # Define the destination folder path
            dest_folder = f'D:/Nextcloud/Vulkameter/Versuche/{content}'

            # Create the destination folder if it doesn't exist
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            # Define the source and destination file paths
            src_file = os.path.join(folder_path, filename)
            dest_file = os.path.join(dest_folder, filename)

            # Move the file to the destination folder
            shutil.move(src_file, dest_file)
            print(f"{file_basename} has been moved successfully to {content}")
        else:
            print(f"No match found for the file {filename}.")

def remove_quotes(s):
    return s.replace('"', '')

# Define counter
x = 0

# Path to the main folder containing the "Versuche" folder
main_folder = "D:/Nextcloud/Vulkameter"

#Define Calculation DataFrame
calculation_df = pd.DataFrame(columns=['Time (s)'])

# Loop through all subfolders in the "Versuche" folder
for folder_name in os.listdir(os.path.join(main_folder, "Versuche")):
    folder_path = os.path.join(main_folder, "Versuche", folder_name)

    # Get the paths of all the txt files in the folder
    txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

    # Check if there are exactly 3 txt files in the folder
    if len(txt_files) == 3:
        y = len(txt_files)
        print(y)

        # Load the data from the three txt files into separate DataFrames
        dfs = []
        for file_name in os.listdir(folder_path):
            if file_name.endswith(".txt"):
                file_path = os.path.join(folder_path, file_name)
                with open(file_path, 'r') as file:
                    contents = file.read()
                    contents = remove_quotes(contents)
                    print(file_name)
                    x += 1

                # Split the contents into rows
                columns = contents.split('\n')

                # Split the rows into columns and store them in a list
                data = []
                for row in columns:
                    columns = row.split()
                    data.append(columns)

                # Construct a DataFrame object from the data and append it to dfs
                df = pd.DataFrame(data, columns=['Time (s)', 'Shear Modulus (dNm)', 'Tan a', 'Upper Lid Temp',
                                                 'Lower Lid Temp'])
                # Convert numeric columns from strings to floats
                df['Time (s)'] = df['Time (s)'].astype(float)
                df['Shear Modulus (dNm)'] = df['Shear Modulus (dNm)'].astype(float)
                df['Tan a'] = df['Tan a'].astype(float)
                df['Upper Lid Temp'] = df['Upper Lid Temp'].astype(float)
                df['Lower Lid Temp'] = df['Lower Lid Temp'].astype(float)

                #Copy the first column into calculation DF
                calculation_df["Time (s)"] = df.iloc[:, 0].astype(float)

                # Add the second column of this DataFrame to the "calculation_df" DataFrame
                column_name = os.path.splitext(file_name)[0]
                calculation_df[column_name] = df.iloc[:, 1].astype(float)
                dfs.append(df)

        # Calculate the mean and deviation
        calculation_df["Mean"] = calculation_df.iloc[:, 1:].mean(axis=1)
        calculation_df["Deviation"] = calculation_df.iloc[:, 1:4].std(axis=1)
        # Save the "Calculation" DataFrame to a CSV file
        calculation_df.to_csv(os.path.join(folder_path, f"{folder_name}.csv"), index=False)
        print(folder_path)
        calculation_df = pd.DataFrame(None)

    else:
        print(f"Der Ordner {folder_path} besitzt keine 3 TXT-Dateien!")
