import os
import pandas as pd
import matplotlib.pyplot as plt
#test

data = pd.read_csv('D:/Nextcloud/Vulkameter/Versuche/JCR 150/JCR 150.csv')

# Extract the x and y data from the DataFrame
x_data = data['Time (s)']/60
y_data = data['Mean']
deviation_data = data['Deviation']

# Plot the data and add labels and deviation as shaded region
fig, ax = plt.subplots()
ax.plot(x_data, y_data, label='Data')
ax.fill_between(x_data, y_data - data['Deviation'], y_data + data['Deviation'], alpha=0.3, label='Deviation')
ax.set_xlabel('X-axis label')
ax.set_ylabel('Y-axis label')
ax.set_title('Plot')
plt.xlim(0, 15)
plt.ylim(0, 15)
ax.legend()
plt.show()
