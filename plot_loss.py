import matplotlib.pyplot as plt

# Function to read loss data from file
def read_loss_data(file_path):
    loss_data = []
    with open(file_path, 'r') as file:
        for line in file:
            if 'Loss' in line:
                loss_value = float(line.split('Loss')[1].strip())
                loss_data.append(loss_value)
    return loss_data

# Path to the file containing loss data
file_path = 'Models/gan_loss'

# Read the loss data from the file
loss_data = read_loss_data(file_path)

# Plotting the loss data
plt.figure(figsize=(10, 5))
plt.plot(loss_data, linestyle='-', color='b')
plt.title('GAN Loss')
plt.xlabel('Iteration')
plt.ylabel('Loss')
plt.grid(True)
plt.show()