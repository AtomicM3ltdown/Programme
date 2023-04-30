import pandas as pd
import os

# Declare variables
master_folder = 'D:/Nextcloud/Zugdehnung/Ergebnis_CSV'


def replace_comma(s):
    return s.replace(',', '.')


def split_alpha_numeric(s):
    alpha = ''
    numeric = ''
    inside_braces = False
    for char in s:
        if char == '{':
            inside_braces = True
            alpha += char
        elif char == '}':
            inside_braces = False
            alpha += char
        elif char.isalpha() or inside_braces:
            alpha += char
        elif char.isnumeric() or (char == '.') or not inside_braces and not (char == ':'):
            if char != ' ':
                numeric += char
        elif char == ':':
            alpha += char
    if numeric == '':
        numeric = '0'
    return alpha, float(numeric)


def process_txt_file(file_path):
    df1 = pd.DataFrame(columns=['Stat', 'Value'])
    with open(file_path, 'r') as file:
        contents = file.read()
        contents = replace_comma(contents)
        file.seek(0)
        with open(file_path, 'w') as file:
            file.write(contents)
    df1 = pd.read_csv(file_path, on_bad_lines='skip', sep='\t', encoding='unicode_escape', names=['Stat']).iloc[:31, :]
    df1[['Stat', 'Value']] = df1['Stat'].apply(lambda x: pd.Series(split_alpha_numeric(x)))
    df2 = pd.read_csv(file_path, skiprows=list(range(31)), sep='\t', on_bad_lines='skip', encoding='unicode_escape',
                      names=['\u03C3 (MPa)', '\u03B5 Fmax (%)', 'F (N)', '\u03C3R (MPa)', '\u03B5 (%)',
                             '\u03B5 (50) (%)',
                             '\u03B5 (100) (%)', '\u03B5 (200) (%)', '\u03B5 (300) (%)', '\u03B5 (500) (%)',
                             'Lc (mm)', 'L0 (mm)', 'a (mm)', 'b (mm)', 'Area (A\u2080) (mm\u00b2)'])
    df2.insert(0, 'Specimen number', range(1, len(df2) + 1))
    return df1, df2


def process_csv_files(folder_path):
    data_frames = []
    for filename in os.listdir(folder_path):
        if filename.endswith(".csv") and filename.endswith("table.csv"):
            csv_path = os.path.join(folder_path, filename)
            df = pd.read_csv(csv_path)
            data_frames.append(df)
    return data_frames


def process_folder(folder_path, excel_writer):
    folder_name = os.path.basename(folder_path)
    sheet_name = folder_name
    existing_sheets = pd.read_excel(excel_writer.path, sheet_name=None)
    if sheet_name in existing_sheets:
        return  # Sheet already exists, skip
    df1_list = []
    df2_list = []
    for filename in os.listdir(folder_path):
        file_path = os.path.join(folder_path, filename)
        if filename.endswith(".TXT"):
            df1, df2 = process_txt_file(file_path)
            df1_list.append(df1)
            df2_list.append(df2)
    if len(df2_list) == 0:
        return  # No CSV files found, skip
     df1 = pd.concat(df1_list, ignore_index=True)
    df2 = pd.concat(df2_list, ignore_index=True)
    df2.insert(0, 'Folder name', folder)
