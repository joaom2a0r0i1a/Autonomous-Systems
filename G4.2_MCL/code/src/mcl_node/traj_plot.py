import numpy as np
import matplotlib.pyplot as plt


extracted_array_ref_1 = [[ 1.27655e-02, -1.85759e-03],
 [ 9.16363e-01, 7.27455e-02],
 [ 2.39808e+00, 2.81628e-01],
 [ 3.98471e+00, 5.14648e-01],
 [ 4.69497e+00, 6.06864e-01]]
 
 # Extract x and y values
x_ref_1 = [row[0]/ 0.050000 + 300 for row in extracted_array_ref_1]
y_ref_1 = [row[1]/ 0.050000 + 302 for row in extracted_array_ref_1]

extracted_array_ref_2 = [[ 1.31330e+01, 1.93447e+00],
 [ 1.31423e+01, 1.94793e+00],
 [ 1.31547e+01, 2.22726e+00],
 [ 1.31112e+01, 4.37224e+00],
 [ 1.31039e+01, 5.54104e+00],
 [ 1.30469e+01, 6.12867e+00],
 [ 1.08971e+01, 5.86934e+00],
 [ 9.94722e+00, 5.73184e+00],
 [ 9.92665e+00, 5.68380e+00],
 [ 9.94563e+00, 5.66042e+00]]
 
 # Extract x and y values
x_ref_2 = [row[0]/ 0.050000 + 300 for row in extracted_array_ref_2]
y_ref_2 = [row[1]/ 0.050000 + 302 for row in extracted_array_ref_2]
 
 
extracted_array_1 = [[0.99056, 0.07819],
 [2.77463, 0.37645],
 [4.46976, 0.66628],
 [4.73321, 0.70405],
 [4.73321, 0.70405],
 [4.73321, 0.70405],
 [4.73321, 0.70405],
 [5.85596, 0.80171],
 [7.90248, 1.12978],
 [8.74419, 1.30131],
 [8.72199, 1.55196],
 [7.77093, 3.80597],
 [7.77093, 3.80597],
 [7.77093, 3.80597]]

# Extract x and y values
x_1 = [row[0]/ 0.050000 + 300 for row in extracted_array_1]
y_1 = [row[1]/ 0.050000 + 302 for row in extracted_array_1]

extracted_array_2 = [[ 7.62947e-02, 7.63254e-02],
 [ 1.35576e+00, 1.96560e-01],
 [ 3.37340e+00, 4.66174e-01],
 [ 5.01452e+00, 7.44066e-01],
 [ 5.14087e+00, 8.18942e-01],
 [ 5.14191e+00, 8.23266e-01],
 [ 5.14237e+00, 8.23235e-01],
 [ 5.76654e+00, 5.86073e+00],
 [ 1.25200e+01, 3.92567e+00],
 [ 1.29498e+01, 5.24159e+00],
 [ 1.28192e+01, 5.99714e+00],
 [ 1.23235e+01, 5.93454e+00],
 [ 9.62503e+00, 5.61525e+00],
 [ 9.60772e+00, 5.57171e+00]]

# Extract x and y values
x_2 = [row[0]/ 0.050000 + 300 for row in extracted_array_2]
y_2 = [row[1]/ 0.050000 + 302 for row in extracted_array_2]

extracted_array_3 = [[ 8.00000e-03, 0.00000e+00],
 [ 2.80000e-02, 0.00000e+00],
 [ 1.04900e+00, -1.00000e-03],
 [ 2.94900e+00, -1.00000e-03],
 [ 4.87000e+00, -1.00000e-03],
 [ 5.09700e+00, -1.00000e-03],
 [ 5.09700e+00, -1.00000e-03],
 [ 5.09700e+00, -1.00000e-03],
 [ 5.09700e+00, -1.00000e-03],
 [ 6.14700e+00, -2.00000e-03],
 [ 8.08500e+00, -2.00000e-03],
 [ 8.91000e+00, 1.70000e-02],
 [ 8.92200e+00, 3.72000e-01],
 [ 9.01200e+00, 3.06200e+00]]
 
# Extract x and y values
x_3 = [row[0]/ 0.050000 + 300 for row in extracted_array_3]
y_3 = [row[1]/ 0.050000 + 302 for row in extracted_array_3]

im = plt.imread("map_01.png")
fig, ax = plt.subplots()
im = ax.imshow(im)

# Flip the y-axis tick values
ax.set_ylim(ax.get_ylim()[::-1])
# Plot the values
ax.plot(x_ref_1, y_ref_1, '-o', label='Reference', color='red', linewidth=2, markersize=5, markerfacecolor='white')
ax.plot(x_ref_2, y_ref_2, '-o', color='red', linewidth=2, markersize=5, markerfacecolor='white')
ax.plot(x_1, y_1, '-o', label='AMCL / Reference', color='purple', markersize=3)
ax.plot(x_2, y_2, '-o', label='Algorithm', color='blue', markersize=3)
ax.plot(x_3, y_3, '-o', label='Odometry', color='green', markersize=3)

ax.set_xlabel('X-axis')
ax.set_ylabel('Y-axis')
ax.set_title('Trajectory')
ax.grid(True)
ax.legend()
plt.show()
