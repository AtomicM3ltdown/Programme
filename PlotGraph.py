import os
import pandas as pd
import matplotlib.pyplot as plt

# Set the path to the master folder
master_folder = "D:/Nextcloud/Vulkameter/Versuche"

# Loop through the folders in the master folder
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

