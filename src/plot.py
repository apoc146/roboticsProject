import matplotlib.pyplot as plt

# Define the y-values for the points
y_values = [1.84, 1.41, 1.58, 1.44, 1.32, 1.29,  1.25, 1.21, 1.19, 1.15]

# Generate the x-values as a range from 0 to the number of y-values minus 1
x_values = range(len(y_values))

# Plot the points
plt.plot(x_values, y_values)

# Add labels and title
plt.xlabel('every 10 epochs')
plt.ylabel('Loss')
plt.title('Loss plot')

# Show the plot
plt.show()
