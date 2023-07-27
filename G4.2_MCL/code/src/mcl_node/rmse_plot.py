import numpy as np
import matplotlib.pyplot as plt


# Read the RMSE data from the text file
data_1 = np.genfromtxt('rmse_data_400_U_K.txt', delimiter=' ')

# Extract time and RMSE values from the data
time_set1 = data_1[0]
rmse_set1 = data_1[1]

time_set2 = data_1[2]
rmse_set2 = data_1[3]

time_set3 = data_1[4]
rmse_set3 = data_1[5]

time_set4 = data_1[6]
rmse_set4 = data_1[7]

time_set5 = data_1[8]
rmse_set5 = data_1[9]

time_set6 = data_1[10]
rmse_set6 = data_1[11]


# Set the figure size and create subplots
fig, axes = plt.subplots(2, 1, figsize=(8, 6))  # Adjust the width and height as desired

# Plot RMSE with respect to time for both sets on the same graph
axes[0].plot(time_set1, rmse_set1, label='Set 1', color='blue')
axes[0].plot(time_set2, rmse_set2, label='Set 2', color='red')
axes[0].plot(time_set3, rmse_set3, label='Set 3', color='green')
axes[0].plot(time_set4, rmse_set4, label='Set 4', color='purple')
axes[0].plot(time_set5, rmse_set5, label='Set 5', color='yellow')
axes[0].plot(time_set6, rmse_set6, label='Set 6', color='orange')
axes[0].set_xlabel('Time')
axes[0].set_ylabel('RMSE')
axes[0].set_ylim(0, None)
axes[0].set_title('Evolution of RMSE over Time (400 particles)')
axes[0].grid(True)
axes[0].xaxis.set_major_locator(plt.MaxNLocator(10))
axes[0].yaxis.set_major_locator(plt.MaxNLocator(10))
axes[0].legend()

# Read the RMSE data from the text file
data_3 = np.genfromtxt('rmse_data_1600_U_K.txt', delimiter=' ')

# Extract time and RMSE values from the data
time_set1 = data_3[0]
rmse_set1 = data_3[1]

time_set2 = data_3[2]
rmse_set2 = data_3[3]

time_set3 = data_3[4]
rmse_set3 = data_3[5]

time_set4 = data_3[6]
rmse_set4 = data_3[7]

time_set5 = data_3[8]
rmse_set5 = data_3[9]

time_set6 = data_3[10]
rmse_set6 = data_3[11]

# Plot RMSE with respect to time for both sets on the same graph
axes[1].plot(time_set1, rmse_set1, label='Set 1', color='blue')
axes[1].plot(time_set2, rmse_set2, label='Set 2', color='red')
axes[1].plot(time_set3, rmse_set3, label='Set 3', color='green')
axes[1].plot(time_set4, rmse_set4, label='Set 4', color='purple')
axes[1].plot(time_set5, rmse_set5, label='Set 5', color='yellow')
axes[1].plot(time_set6, rmse_set6, label='Set 6', color='orange')
axes[1].set_xlabel('Time')
axes[1].set_ylabel('RMSE')
axes[1].set_title('Evolution of RMSE over Time (1600 particles)')
axes[1].grid(True)
axes[1].xaxis.set_major_locator(plt.MaxNLocator(10))
axes[1].yaxis.set_major_locator(plt.MaxNLocator(10))
axes[1].legend()

# Adjust spacing between subplots
plt.subplots_adjust(hspace=0.5)

plt.show()

