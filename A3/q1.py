import numpy as np
import matplotlib.pyplot as plt
from scipy.spatial.distance import cdist

# Function to generate random dataset
def generate_dataset(num_points, num_dimensions):
    return np.random.uniform(0, 1000, size=(num_points, num_dimensions))

# Function to compute distances using L1, L2, and Linf norms
def compute_distances(query_points, dataset,query_indices):
    l1_distances = []
    l2_distances = []
    linf_distances = []
    for i in range(len(query_points)):
        l1_dist = cdist([query_points[i]], np.delete(dataset, query_indices[i], axis=0), metric='cityblock').flatten()
        l1_distances.append(l1_dist)
        
        l2_dist = cdist([query_points[i]], np.delete(dataset, query_indices[i], axis=0), metric='euclidean').flatten()
        l2_distances.append(l2_dist)
        
        linf_dist = cdist([query_points[i]], np.delete(dataset, query_indices[i], axis=0), metric='chebyshev').flatten()
        linf_distances.append(linf_dist)
    
    return np.array(l1_distances), np.array(l2_distances), np.array(linf_distances)

# Function to calculate the ratio of farthest and nearest distances
def calculate_ratio(farthest_distances, nearest_distances):
    return farthest_distances / nearest_distances

# Function to plot the average ratio versus d for the three distance measures
def plot_ratio_vs_dimension(average_ratios, dimensions, filename=None):
    plt.figure(figsize=(10, 6))
    
    # Plot for L1 Norm
    plt.plot(dimensions, average_ratios['L1'], color='blue')
    plt.xlabel('Dimension (d)')
    plt.ylabel('Average Ratio of Farthest and Nearest Distances')
    plt.title('Average Ratio vs. Dimension for L1 Norm')
    plt.grid(True)
    if filename:
        plt.savefig(filename + '_L1.png')
    else:
        plt.show()
    plt.close()
    
    # Plot for L2 Norm
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, average_ratios['L2'], color='orange')
    plt.xlabel('Dimension (d)')
    plt.ylabel('Average Ratio of Farthest and Nearest Distances')
    plt.title('Average Ratio vs. Dimension for L2 Norm')
    plt.grid(True)
    if filename:
        plt.savefig(filename + '_L2.png')
    else:
        plt.show()
    plt.close()

    # Plot for Linf Norm
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, average_ratios['Linf'], color='green')
    plt.xlabel('Dimension (d)')
    plt.ylabel('Average Ratio of Farthest and Nearest Distances')
    plt.title('Average Ratio vs. Dimension for Linf Norm')
    plt.grid(True)
    if filename:
        plt.savefig(filename + '_Linf.png')
    else:
        plt.show()
    plt.close()

    # Common plot for all distance measures
    plt.figure(figsize=(10, 6))
    plt.plot(dimensions, average_ratios['L1'], label='L1 Norm', color='blue')
    plt.plot(dimensions, average_ratios['L2'], label='L2 Norm', color='orange')
    plt.plot(dimensions, average_ratios['Linf'], label='L∞ Norm', color='green')
    plt.xlabel('Dimension (d)')
    plt.ylabel('Average Ratio of Farthest and Nearest Distances')
    plt.title('Average Ratio vs. Dimension for Different Distance Measures')
    plt.legend()
    plt.grid(True)
    if filename:
        plt.savefig(filename + '_combined.png')
    else:
        plt.show()
    plt.close()


# Define dimensions
dimensions = [1, 2, 4, 6, 8, 10, 12, 14, 16]

# Define number of points and query points
num_points = 1000000
num_query_points = 100

# Generate dataset for each dimension
datasets = {}
for d in dimensions:
    datasets[d] = generate_dataset(num_points, d)

# Generate query points and compute distances for each dimension
average_ratios = {'L1': [], 'L2': [], 'Linf': []}
for d in dimensions:
    print(f"Computing distances for d={d}...")
    dataset = datasets[d]

    query_indices = np.random.choice(len(dataset), size=num_query_points , replace=False)
    query_points = dataset[query_indices]
    # Remove the query points from the dataset
    # dataset = np.delete(dataset, query_indices, axis=0)
    
    l1_distances, l2_distances, linf_distances = compute_distances(query_points, dataset, query_indices)
    # Find the nearest and farthest distances for each query point
    nearest_distances = np.min(l1_distances, axis=1)
    farthest_distances = np.max(l1_distances, axis=1)
    
    # Calculate average ratio for L1 norm
    ratio = calculate_ratio(farthest_distances, nearest_distances)
    average_ratios['L1'].append(np.mean(ratio))
    
    # Repeat for L2 and Linf norms
    nearest_distances = np.min(l2_distances, axis=1)
    farthest_distances = np.max(l2_distances, axis=1)
    ratio = calculate_ratio(farthest_distances, nearest_distances)
    average_ratios['L2'].append(np.mean(ratio))
    
    nearest_distances = np.min(linf_distances, axis=1)
    farthest_distances = np.max(linf_distances, axis=1)
    ratio = calculate_ratio(farthest_distances, nearest_distances)
    average_ratios['Linf'].append(np.mean(ratio))

# Plot the average ratio versus d for the three distance measures and save to file
plot_ratio_vs_dimension(average_ratios, dimensions, filename='q1')
