import os
import pandas as pd


def replace_comma(s):
    return s.replace(',', '.')


# Get the list of files in the folder
folder_path = 'D:/Nextcloud/Zugdehnung/Ergebnis_CSV'
file_list = os.listdir(folder_path)

for filename in file_list:
    file_path = os.path.join(folder_path, filename)
    file_basename = os.path.splitext(filename)[0]
    dest_path = os.path.join(folder_path, file_basename)

    if filename.endswith(".TXT"):
        df1 = pd.DataFrame(columns = ['Stat', 'Value'])
        # file_path = "F:/Zugdehnung/JH_JCR1311.TXT"
        print(file_path)
        with open(file_path, 'r') as file:
            contents = file.read()
            contents = replace_comma(contents)
            # print(contents)
        # Load the CSV file into a DataFrame
        #df = pd.read_csv(file_path, on_bad_lines='skip', sep='\t')

        # Split the DataFrame into two based on the number of rows
        df = pd.read_csv(file_path, on_bad_lines='skip', sep='\t', encoding= 'unicode_escape', names=['Stat'], engine='python').iloc[:31, :]
        #df1 = df.iloc[:30, :]  # The first 30 rows
        df1[['Stat', 'Value']] = df['Stat'].str.split(expand=True, pat='}:')
        df2 = pd.read_csv(file_path,
                          skiprows=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23,
                                    24, 25, 26, 27, 28, 29, 30, 31, 32, 33], sep='\t', on_bad_lines='skip',encoding= 'unicode_escape',
                          names=['Specimen number', '\u03C3' + ' (MPa)',
                                 '\u03B5 Fmax' + ' (%)',
                                 'F (N)', '\u03C3R' + ' (MPa)','\u03B5' + ' (%)',
                                 '\u03B5 (50) (%)', '\u03B5 (100) (%)',
                                 '\u03B5 (200) (%)',
                                 '\u03B5 (300) (%)', '\u03B5 (500) (%)',
                                 'L (mm)', 'a (mm)', 'b (mm)',
                                 'Area (A\u2080) (mm\u00b2)'])



        #   Print out the two DataFrames to check that they look right
        print(df1)
        print(df2)
        # Create the destination folder if it doesn't exist
        if not os.path.exists(dest_path):
            os.makedirs(dest_path)

        df1.to_csv(os.path.join(dest_path, f"{file_basename}_stat.csv"), index=False, doublequote=False)
        df2.to_csv(os.path.join(dest_path, f"{file_basename}_table.csv"), index=False, doublequote=False)
        df1 = None
        df2 = None
