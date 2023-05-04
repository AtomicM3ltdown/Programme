import os
import pandas as pd
import matplotlib.pyplot as plt

# Set the path to the master folder
master_folder = "D:/Nextcloud/Vulkameter/Versuche"

# Prompt the user for up to five folders to load and plot
folders = []
labels = []
for i in range(5):
    folder = input(f"Enter folder {i+1} to load and plot (or leave blank to end): ")
    if not folder:
        break
    label = input(f"Enter label for {folder}: ")
    folders.append(folder)
    labels.append(label)

# Create the figure and axes objects
fig, ax = plt.subplots()

# Loop through the selected folders
for folder, label in zip(folders, labels):
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

        ax.plot(x_data, y_data, label=label)
        ax.fill_between(x_data, y_data - data['Deviation'], y_data + data['Deviation'], alpha=0.3, label=None)

        # Set the axis labels and limits
        ax.set_xlabel('Time (min)')
        ax.set_ylabel('Torque (dNm)')
        ax.set_title('Torque vs Time')
        plt.autoscale(enable=True, axis='both')
        ax.set_xlim(left=0)
        ax.set_ylim(bottom=0)

        # Set the tick locators
        tick_spacing = 1
        ax.xaxis.set_major_locator(plt.MultipleLocator(tick_spacing))
        ax.yaxis.set_major_locator(plt.MultipleLocator(tick_spacing))

        # Add the legend
        ax.legend(loc="lower right", fancybox=True, framealpha=0.5)

        # Print the folder path
        print(f'{folder_path}')

    else:
        print(f"No CSV file found in {folder_path}.")

# Save graph
save_png = os.path.join(master_folder, 'merged')

# Save and print the graph
plt.savefig(save_png, dpi=300, transparent=True)

# Show the final plot
plt.show()
