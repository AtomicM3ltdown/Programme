import os
import shutil
import glob
import pandas as pd

def load_excel_sheet(file_path, sheet_name):
    return pd.read_excel(file_path, sheet_name=sheet_name)

def extract_file_basename(source_folder, file_extension):
    return [os.path.splitext(f)[0] for f in os.listdir(source_folder)
            if os.path.isfile(os.path.join(source_folder, f)) and f.endswith(file_extension)]

def move_files_to_destination_folder(source_folder, file_extension, excel_df, dest_root_folder, log_func=print):
    """
    Verschiebt Dateien aus source_folder nach dest_root_folder,
    basierend auf den Excel-Metadaten.
    """
    file_basenames = extract_file_basename(source_folder, file_extension)
    for file_basename in file_basenames:
        log_func(file_basename)

        # Excel-Match
        match = excel_df[(excel_df['jlph_t'].str.contains('JC') | excel_df['jl_t'].str.contains('JC'))
                         & excel_df['Rdatafile'].eq(file_basename)]
        if not match.empty:
            column_name = 'jlph_t' if 'JC' in match['jlph_t'].iloc[0] else 'jl_t'
            content = match[column_name].iloc[0]
            dest_folder = os.path.join(dest_root_folder, content)
            os.makedirs(dest_folder, exist_ok=True)

            src_file = os.path.join(source_folder, file_basename + '.' + file_extension)
            dest_file = os.path.join(dest_folder, file_basename + '.' + file_extension)

            # Falls Datei schon existiert → überspringen
            if os.path.exists(dest_file):
                log_func(f"Übersprungen: {dest_file} existiert bereits")
                continue

            shutil.copy(src_file, dest_file)
            log_func(f"{file_basename} erfolgreich nach {content} kopiert")
        else:
            log_func(f"Kein Match gefunden für {file_basename}")

def remove_quotes(s):
    return s.replace('"', '')

def load_txt_files_into_dataframe(master_folder_path, log_func=print):
    """
    Liest alle TXT-Dateien in den Unterordnern und erzeugt pro Unterordner eine CSV.
    Falls die CSV bereits existiert → wird übersprungen.
    """
    calculation_df = pd.DataFrame(columns=['Time (s)'])

    for folder_name in os.listdir(master_folder_path):
        folder_path = os.path.join(master_folder_path, folder_name)
        if not os.path.isdir(folder_path) or folder_name == 'Test':
            continue

        result_csv = os.path.join(folder_path, f"{folder_name}.csv")
        if os.path.exists(result_csv):
            log_func(f"Übersprungen: {result_csv} existiert bereits")
            continue

        txt_files = glob.glob(os.path.join(folder_path, "*.txt"))
        if len(txt_files) >= 3:
            for file_name in os.listdir(folder_path):
                if file_name.endswith(".txt"):
                    file_path = os.path.join(folder_path, file_name)
                    with open(file_path, 'r') as file:
                        contents = remove_quotes(file.read())

                    rows = [row.split() for row in contents.split('\n')]
                    df = pd.DataFrame(rows, columns=['Time (s)', 'Shear Modulus (dNm)', 'Tan a',
                                                     'Upper Lid Temp', 'Lower Lid Temp'])
                    df = df.astype({'Time (s)': float,
                                    'Shear Modulus (dNm)': float,
                                    'Tan a': float,
                                    'Upper Lid Temp': float})
                    calculation_df["Time (s)"] = df["Time (s)"]
                    column_name = os.path.splitext(file_name)[0]
                    calculation_df[column_name] = df["Shear Modulus (dNm)"]

            calculation_df["Mean"] = calculation_df.iloc[:, 1:].mean(axis=1)
            calculation_df["Deviation"] = calculation_df.iloc[:, 1:4].std(axis=1)
            calculation_df.to_csv(result_csv, index=False)
            log_func(f"CSV für {folder_name} erstellt: {result_csv}")

            calculation_df = pd.DataFrame(columns=['Time (s)'])
    return
