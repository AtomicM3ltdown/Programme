import os
import pandas as pd
import shutil
import glob
import matplotlib as plt

folder_path = r'D:/Nextcloud/Vulkameter/Rohdaten'
versuche_path = 'D:/Nextcloud/Vulkameter/Versuche'


def load_excel_sheet(file_path, sheet_name):
    """Loads an Excel sheet into a Pandas DataFrame."""
    return pd.read_excel(file_path, sheet_name=sheet_name)


def extract_file_basename(source_folder, file_extension):
    """Extracts the basename of files in a folder that match a certain extension."""
    file_basenames = []
    for filename in os.listdir(source_folder):
        if os.path.isfile(os.path.join(source_folder, filename)) and filename.endswith(file_extension):
            file_basename = os.path.splitext(filename)[0]
            file_basenames.append(file_basename)
    return file_basenames


def move_files_to_destination_folder(source_folder, file_extension, excel_df, dest_root_folder):
    """Moves files matching a certain criterion to a destination folder based on data in an Excel sheet."""
    file_basenames = extract_file_basename(source_folder, file_extension)
    for file_basename in file_basenames:
        print(file_basename)
        # Search for a match in the Excel sheet
        match = excel_df[(excel_df['jlph_t'].str.contains('JC') | excel_df['jl_t'].str.contains('JC'))
                         & excel_df['Rdatafile'].eq(file_basename)]
        if not match.empty:
            # Extract the content of the column 'jlph_t' or 'jl_t' in the same row containing "JC"
            column_name = 'jlph_t' if 'JC' in match['jlph_t'].iloc[0] else 'jl_t'
            content = match[column_name].iloc[0]
            print(f"The file {file_basename} matches the data in the Excel sheet. Content: {content}")
            # Define the destination folder path
            dest_folder = os.path.join(dest_root_folder, content)

            # Create the destination folder if it doesn't exist
            if not os.path.exists(dest_folder):
                os.makedirs(dest_folder)

            # Define the source and destination file paths
            src_file = os.path.join(source_folder, file_basename + '.' + file_extension)
            dest_file = os.path.join(dest_folder, file_basename + '.' + file_extension)

            # Move the file to the destination folder
            shutil.copy(src_file, dest_file)
            print(f"{file_basename} has been moved successfully to {content}")
        else:
            print(f"No match found for the file {file_basename}.")


def remove_quotes(s):
    """Removes double quotes from a string."""
    return s.replace('"', '')


def load_txt_files_into_dataframe(master_folder_path):
    """Loads three TXT files from subfolders of a master folder into separate DataFrames and returns a list of these
    DataFrames."""
    dfs = []
    # Define Calculation DataFrame
    calculation_df = pd.DataFrame(columns=['Time (s)'])
    for folder_name in os.listdir(os.path.join(master_folder_path)):
        folder_path = os.path.join(master_folder_path, folder_name)
        if not os.path.isdir(folder_path):
            print(f"{folder_path} is not a directory")
            return
        if folder_name == 'Test':
            print(f"{folder_path} ist der Testordner")
            return
        # Get the paths of all the txt files in the folder
        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

        # Check if there are exactly 3 txt files in the folder
        if len(txt_files) >= 3:
            y = len(txt_files)
            print(y)

            for file_name in os.listdir(folder_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as file:
                        contents = file.read()
                        contents = remove_quotes(contents)
                        print(file_name)

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

                    # Copy the first column into calculation DF
                    calculation_df["Time (s)"] = df.iloc[:, 0].astype(float)
                    # Add the second column of this DataFrame to the "calculation_df" DataFrame
                    column_name = os.path.splitext(file_name)[0]
                    calculation_df[column_name] = df.iloc[:, 1].astype(float)
                    dfs.append(df)
                    print('added a column')
        # Calculate the mean and deviation
        calculation_df["Mean"] = calculation_df.iloc[:, 1:].mean(axis=1)
        calculation_df["Deviation"] = calculation_df.iloc[:, 1:4].std(axis=1)
        # Save the "Calculation" DataFrame to a CSV file
        calculation_df.to_csv(os.path.join(folder_path, f"{folder_name}.csv"), index=False)
        calculation_df = pd.DataFrame(None)
    return
'''
def plot(master_folder):
    for folder in os.listdir(master_folder):
        folder_path = os.path.join(master_folder, folder)

        # Check if the folder contains a CSV file
        csv_file = None
        for file in os.listdir(folder_path):
            if file.endswith(".csv"):
                csv_file = os.path.join(folder_path, file)
                break

        # If a CSV file was found, load it into a pandas DataFrame
        if csv_file is not None:
            data = pd.read_csv(csv_file)
            print(data)
            # Extract the x and y data from the DataFrame
            x_data = data['Time (s)'] / 60
            y_data = data['Mean']
            deviation_data = data['Deviation']

            # Plot the data and add labels and deviation as shaded region
            fig, ax = plt.subplots()
            ax.plot(x_data, y_data, label=f'{folder}')
            ax.fill_between(x_data, y_data - data['Deviation'], y_data + data['Deviation'], alpha=0.3, label=None)
            ax.set_xlabel('Time (min)')
            ax.set_ylabel('Torque (dNm)')
            ax.set_title(f'{folder}')
            tick_spacing = 1
            ax.xaxis.set_major_locator(plt.MultipleLocator(tick_spacing))
            ax.yaxis.set_major_locator(plt.MultipleLocator(tick_spacing))
            plt.autoscale(enable=True, axis='both')
            plt.xlim(left=0)
            plt.ylim(bottom=0)

            ax.legend(loc="lower right", )
            print(f'{folder_path}')

            # Save graph
            save_png = os.path.join(folder_path, folder)

            # Save and print the graph
            plt.savefig(save_png)
            plt.show()

        else:
            print(f"No CSV file found in {folder_path}.")
'''
if __name__ == "__main__":
    
    df_excel = load_excel_sheet(r'D:/Nextcloud/Vulkameter/Average.xlsx', 'Tabelle2')

    # print(df_excel['Rdatafile'])
    basenames = extract_file_basename(folder_path, 'txt')
    # print(basenames)
    move_files_to_destination_folder(folder_path, 'txt', df_excel, versuche_path)
    load_txt_files_into_dataframe(versuche_path)
    
    #plot(versuche_path)
