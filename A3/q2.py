import pandas as pd
import numpy as np
import time
from sklearn.decomposition import PCA
from scipy.spatial import KDTree
import matplotlib.pyplot as plt
from sklearn.neighbors import BallTree
import sys
import falconn 

def seqtopk(points, query, k):
    euclidean_distances = np.linalg.norm(points - query, axis=1 , ord = 2 )
    nearest_indices = np.argsort(euclidean_distances)[:k]
    return nearest_indices

def compute_jaccard_similarity(true_neighbors, predicted_neighbors):
    jaccard_similarities = []
    for true, pred in zip(true_neighbors, predicted_neighbors):
        true_set = set(true)
        pred_set = set(pred)
        jaccard_sim = len(true_set.intersection(pred_set)) / len(true_set.union(pred_set))
        jaccard_similarities.append(jaccard_sim)
    return np.mean(jaccard_similarities)

def plot_results(results):
    plt.figure(figsize=(10, 6))
    for index_struct in set(results['Index Structure']):
        data_subset = results[results['Index Structure'] == index_struct]
        plt.errorbar(data_subset['Dimension'], data_subset['Average Time'], yerr=data_subset['Std Dev'], label=index_struct, fmt='-o')
    plt.xlabel('Dimension')
    plt.ylabel('Average Time (seconds)')
    plt.title('Average Running Time of 5-NN Query Across Dimensions')
    plt.legend()
    plt.grid(True)
    plt.savefig('Q2_c.png')
    plt.close()
    plt.figure(figsize=(10, 6))
    for index_struct in set(results['Index Structure']):
        plt.figure(figsize=(10, 6))
        data_subset = results[results['Index Structure'] == index_struct]
        plt.errorbar(data_subset['Dimension'], data_subset['Average Time'], yerr=data_subset['Std Dev'], label=index_struct, fmt='-o')
        plt.xlabel('Dimension')
        plt.ylabel('Average Time (seconds)')
        plt.title(f'Average Running Time of 5-NN Query Across Dimensions - {index_struct}')
        plt.legend()
        plt.grid(True)
        plt.savefig(f'{index_struct}.png')
        plt.close()

s = time.time()

sub_part = sys.argv[1]
dataset_path = sys.argv[2]

if sub_part == 'c':
    data = pd.read_csv(dataset_path, header=None, delimiter=" ")
    data = data.iloc[:, :-1]

    dimensions = [2, 4, 10, 20]
    K = 5
    num_trials = 100

    number_of_tables = 25

    results = {'Dimension': [], 'Index Structure': [], 'Average Time': [], 'Std Dev': []}

    jaccard_accuracies = [] 

    for dim in dimensions:
        pca = PCA(n_components=dim, random_state=42)
        print("PCA done for d =", dim)
        reduced_data = pca.fit_transform(data)
        
        query_index = np.random.randint(0, len(data), num_trials)
        query_points = [reduced_data[query_index[i]] for i in range(num_trials)]

        df = pd.DataFrame(data=reduced_data, columns=[f"PC_{i+1}" for i in range(dim)])
        for column in df.columns:
            mean = df[column].mean()
            std = df[column].std()
            df[column] = (df[column] - mean) / std

        points = df.to_numpy()
        queries = np.array([points[query_index[i]] for i in range(num_trials)])

        #Sequential
        answers = []
        avg_time = 0
        std_dev = 0
        for query_point in queries:
            start_time = time.time()
            answers.append(seqtopk(points, query_point , K))
            end_time = time.time()
            avg_time += (end_time - start_time)  
            std_dev += (end_time - start_time) ** 2

        avg_time /= num_trials
        std_dev = np.sqrt(std_dev / num_trials)

        results['Dimension'].append(dim)
        results['Index Structure'].append('Sequential')
        results['Average Time'].append(avg_time)
        results['Std Dev'].append(std_dev)
        
        print('Dimension =', dim, ', Index Structure = None(Sequential)', ", Average time =", avg_time, ", Std Dev =", std_dev)

        #LSH
        params_cp = falconn.LSHConstructionParameters()
        params_cp.dimension = df.shape[1]

        params_cp.lsh_family = falconn.LSHFamily.Hyperplane
        params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
        params_cp.l = number_of_tables
        # params_cp.k = 20

        params_cp.num_rotations = 1
        params_cp.seed = 5721840

        params_cp.num_setup_threads = 0
        params_cp.storage_hash_table = falconn.StorageHashTable.LinearProbingHashTable

        falconn.compute_number_of_hash_functions(10, params_cp) 

        table = falconn.LSHIndex(params_cp)
        table.setup(points)

        query_object = table.construct_query_object()

        number_of_probes = number_of_tables

        score = 0
        cur_answers = []
        avg_time = 0
        std_dev = 0
        for (i, query) in enumerate(queries):
            start_time = time.time()
            points_ind = query_object.find_k_nearest_neighbors(query , k = K)
            answer_set = points[points_ind]
            end_time = time.time()
            avg_time += (end_time - start_time)  
            std_dev += (end_time - start_time) ** 2
            cur_answers.append(points_ind)

        avg_time /= num_trials
        std_dev = np.sqrt(std_dev / num_trials)
        
        results['Dimension'].append(dim)
        results['Index Structure'].append('LSH')
        results['Average Time'].append(avg_time)
        results['Std Dev'].append(std_dev)
        
        print('Dimension =', dim, ', Index Structure = LSH', ", Average time =", avg_time, ", Std Dev =", std_dev)

        cur_acc = compute_jaccard_similarity(answers, cur_answers)
        jaccard_accuracies.append(100*cur_acc)

        #KD-Tree
        avg_time = 0
        std_dev = 0
        tree = KDTree(reduced_data)
        for query_point in query_points:           
            start_time = time.time()
            dists, _ = tree.query(query_point.reshape(1, -1), k=K)
            end_time = time.time()
            
            avg_time += (end_time - start_time)  
            std_dev += (end_time - start_time) ** 2
        
        avg_time /= num_trials
        std_dev = np.sqrt(std_dev / num_trials)
        
        results['Dimension'].append(dim)
        results['Index Structure'].append('KD-tree')
        results['Average Time'].append(avg_time)
        results['Std Dev'].append(std_dev)
        
        print('Dimension =', dim, ', Index Structure = KD-Tree', ", Average time =", avg_time, ", Std Dev =", std_dev)

        #M-Tree
        avg_time = 0
        std_dev = 0
        ball_tree = BallTree(reduced_data)
        for q in query_points:
            start_time = time.time()
            distances, indices = ball_tree.query([q], k=K)
            end_time = time.time()
            avg_time += (end_time - start_time)  
            std_dev += (end_time - start_time) ** 2
        
        avg_time /= num_trials
        std_dev = np.sqrt(std_dev / num_trials)
        
        results['Dimension'].append(dim)
        results['Index Structure'].append('M-tree')
        results['Average Time'].append(avg_time)
        results['Std Dev'].append(std_dev)
        
        print('Dimension =', dim, ', Index Structure = M-Tree', ", Average time =", avg_time, ", Std Dev =", std_dev)

    plt.figure(1)
    equally_spaced_x = range(len(dimensions))

    plt.plot(equally_spaced_x, jaccard_accuracies, marker='o', linestyle='-')
    plt.xticks(equally_spaced_x, dimensions)

    plt.title('Jaccard Accuracy vs Dimensions')
    plt.xlabel('Dimension')
    plt.ylabel('Jaccard Accuracy(%)')
    # plt.xticks(dimensions)
    plt.grid(True)
    plt.savefig("Q2_c_DimVsAcc.png")
    plot_results(pd.DataFrame(results))


if sub_part == 'd':
    def seqtopk(points, query, k):
        euclidean_distances = np.linalg.norm(points - query, axis=1)
        nearest_indices = np.argsort(euclidean_distances)[:k]
        return nearest_indices


    def compute_jaccard_similarity(true_neighbors, predicted_neighbors):
        jaccard_similarities = []
        for true, pred in zip(true_neighbors, predicted_neighbors):
            true_set = set(true)
            pred_set = set(pred)
            jaccard_sim = len(true_set.intersection(pred_set)) / len(true_set.union(pred_set))
            jaccard_similarities.append(jaccard_sim)
        return np.mean(jaccard_similarities)
    number_of_tables = 25
    df = pd.read_csv(dataset_path , delimiter=' ', header=None)
    print("read data")
    df = df.iloc[:, :-1]
    print(df[0:10])
    #-----------------remove--------------------
    # df = df[0:10000]
    #-----------------remove--------------------
    dimensions = [2, 4, 10, 20]

    # Initialize dictionary to store DataFrames
    dataframes = {}
    Ks = [1, 5, 10, 50, 100, 500]
    # Perform PCA for each dimension and store in dataframes
    for dim in dimensions:
        pca = PCA(n_components=dim, random_state=42)
        pca_result = pca.fit_transform(df)
        df_pca = pd.DataFrame(data=pca_result, columns=[f"PC_{i+1}" for i in range(dim)])
        dataframes[dim] = df_pca


    for dim in dimensions:
        jaccard_accuracies = [] 
        for K in Ks:
            print(f"Using number of dimensions = {dim}")
            df = dataframes[dim]
            for column in df.columns:
                mean = df[column].mean()
                std = df[column].std()
                df[column] = (df[column] - mean) / std
            

            queries = df.sample(n = 100 , random_state =42)

            # queries = df.to_numpy()
            queries = queries.to_numpy()
            points = df.to_numpy()

            print("Finding top - k using sequential Scan ... ")
            t1 = time.time()

            answers = []

            for query in queries:
                answers.append(seqtopk(points, query , K))
            t2 = time.time()
            print('Query time for sequential scan algorithm : {}'.format((t2 - t1) / len(queries)))

            params_cp = falconn.LSHConstructionParameters()
            params_cp.dimension = df.shape[1]

            params_cp.lsh_family = falconn.LSHFamily.Hyperplane
            params_cp.distance_function = falconn.DistanceFunction.EuclideanSquared
            params_cp.l = number_of_tables
            # params_cp.k = 20

            params_cp.num_rotations = 1
            params_cp.seed = 5721840

            params_cp.num_setup_threads = 0
            params_cp.storage_hash_table = falconn.StorageHashTable.LinearProbingHashTable

            falconn.compute_number_of_hash_functions(10, params_cp) 
            print('Constructing the LSH table')
            t1 = time.time()
            table = falconn.LSHIndex(params_cp)
            table.setup(points)
            t2 = time.time()
            print('Done')
            print('Construction time: {}'.format(t2 - t1))


            query_object = table.construct_query_object()

            number_of_probes = number_of_tables

            # query_object.set_num_probes(number_of_probes)

            t1 = time.time()

            score = 0
            cur_answers = []
            times = [] 
            for (i, query) in enumerate(queries):
                t_query1 = time.time()
                points_ind = query_object.find_k_nearest_neighbors(query , k = K)
                answer_set = points[points_ind]
                # if query in answer_set:
                #     points_ind = points_ind[1:]
                #     answer_set = answer_set[1:]
                # else :
                #     points_ind = points_ind[:-1]
                #     answer_set = answer_set[:-1]
                t_query2 = time.time()
                times.append(t_query2 - t_query1)
                cur_answers.append(points_ind)

            mean = np.mean(times)
            std_dev = np.std(times)


            cur_acc = compute_jaccard_similarity(answers, cur_answers)
            jaccard_accuracies.append(cur_acc*100)

            t2 = time.time()


            print('Query time: {}'.format((t2 - t1) / len(queries)))

        equally_spaced_x = range(len(Ks))

        plt.plot(equally_spaced_x, jaccard_accuracies, marker='o', linestyle='-' , label = f"dim = {dim}")
        plt.title('Jaccard Accuracy vs k')
        plt.xlabel('k')
        plt.ylabel('Jaccard Accuracy %')
        plt.xticks(equally_spaced_x, Ks)
        plt.grid(True)
        plt.legend()
    plt.savefig(f"Q2_d_KvsAccuracy.png")


e = time.time()

print("Total time =", (e - s))