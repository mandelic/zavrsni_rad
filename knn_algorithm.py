import numpy as np
import pandas as pd
from math import *
import time
from sklearn.datasets import make_classification
import heapq

class Distance:
    def __init__(self, metric = 'euclidean', n_power = 2):
        if metric == 'euclidean':
            self.distance = self.euclidean_distance
        elif metric == "manhattan":
            self.distance = self.manhattan_distance
        elif metric == "minkowski":
            self.distance = self.minkowski_distance
            self.n_power = n_power
        elif metric == "cosine":
            self.distance = self.cosine_distance
        else:
            raise ValueError(f"Invalid metric: {metric}")


    def calculate(self, p, q, dim):
        self.p = p[:dim]
        self.q = q[:dim]
        return self.distance()

    def euclidean_distance(self):
        return np.sqrt(np.sum((self.p - self.q) ** 2))
    
    def manhattan_distance(self):
        return np.sum(np.abs(self.p - self.q))
    
    def minkowski_distance(self):
        return np.power(np.sum(np.abs(self.p - self.q) ** self.n_power), 1/self.n_power)
    
    def cosine_distance(self):
        #p_normalized = self.p / np.linalg.norm(self.p)
        #q_normalized = self.q / np.linalg.norm(self.q)
        #return  1 - np.dot(p_normalized, q_normalized)
        return 1 - np.dot(self.p, self.q) / (np.linalg.norm(self.p) * np.linalg.norm(self.q))


class KdNode:
    def __init__(self, median = None, parent = None, data = None, left = None, right = None, axis = None, side = None):
        self.median = median
        self.parent = parent
        self.data = data
        self.left = left
        self.right = right
        self.axis = axis
        self.side = side
    def is_leaf(self):
        return not self.data is None

class KdTree_:
    def __init__(self, root = None):
        self.root = root

class Ball:
    def __init__(self, ctr, pts, r = 0):
        self.ctr = ctr # center of the ball
        self.pts = pts # points in the ball
        self.r = r
    def calculate_radius(self, distance, n_dim):
        self.r = max([distance.calculate(self.ctr, p, n_dim) for p in self.pts])

class BallNode:
    def __init__(self, bl, lt = None, rt = None):
        self.bl = bl #ball
        self.lt = lt #left child
        self.rt = rt #right child

    def is_leaf(self):
        return self.lt is None and self.rt is None

class BallTree_:
    def __init__(self, root = None):
        self.root = root # root node

class BruteForce:
    def __init__(self, distance, X = None, y = None):
        self.X = X
        self.y = y
        self.distance = distance
        self.fit(X, y)

    def fit(self, X, y, leaf_size = None):
        if not X is None and not y is None:
            self.X = X
            self.y = y
            self.n_dim = X.shape[1]
            dataset = pd.DataFrame(self.X) 
            dataset.columns = ['X{}'.format(i+1) for i in range(X.shape[1])]
            dataset['y'] = self.y
            self.dataset = dataset

    def query(self, query_data, k):
        return self.brute_force_knn(query_data, k)

    def brute_force_knn(self, query_data, k):
        self.dataset['distance'] = self.dataset.apply(lambda x: self.distance.calculate([x[f'X{i+1}'] for i in range(self.n_dim)], query_data, self.n_dim), axis=1)
        dataset_with_distance = self.dataset.sort_values(by="distance")[:k]
        print("K({}) smallest distances [brute_force]: {}".format(k, dataset_with_distance['distance'].to_numpy().tolist()))
        print("K({}) nearest points [brute_force]: {}".format(k, [x[:-1] for x in  dataset_with_distance.to_numpy().tolist()]))
        return dataset_with_distance["y"].value_counts().idxmax()

class KDTree:
    def __init__(self, distance, X = None, y = None, leaf_size = 1):
        self.X = X
        self.y = y
        self.distance = distance
        self.leaf_size = leaf_size
        self.fit(X, y)
        
    def fit(self, X, y, leaf_size = 1):
        if not X is None and not y is None:
            self.X = X
            self.y = y
            self.leaf_size = leaf_size
            self.n_dim = X.shape[1]
            dataset = pd.DataFrame(self.X) 
            dataset.columns = ['X{}'.format(i+1) for i in range(X.shape[1])]
            dataset['y'] = self.y
            self.dataset = dataset
            self.tree = KdTree_(self.construct_tree())

    def construct_tree(self):
        data_array = self.dataset.to_numpy()
        return self.construction(data_array)
    
    def construction(self, data_array, kd_node = None):
        if len(data_array) <= self.leaf_size:
            axis = (kd_node.axis + 1) % (self.n_dim)
            return KdNode(data = data_array, parent = kd_node, axis = axis)
        if kd_node is None:
            axis = 0 
        else:
            axis = (kd_node.axis + 1) % (self.n_dim)
        data_array = sorted(data_array, key=lambda x: x[axis])
        median_index = len(data_array) // 2 if len(data_array) % 2 == 0 else (len(data_array) - 1) // 2  
        median = data_array[median_index].tolist()
        kd_node = KdNode(median = median, axis = axis, parent = kd_node) 
        kd_node.left = self.construction(data_array[:median_index], kd_node)
        kd_node.left.side = 0
        kd_node.right = self.construction(data_array[median_index + 1:], kd_node)
        kd_node.right.side = 1
        return kd_node

    def query(self, query_data, k):
        return self.kd_tree_knn(np.asarray(query_data), k)

    def fake_dfs(self, query_data, root_node):
        node = root_node
        while not node.is_leaf():
            axis = node.axis
            if query_data[axis] < node.median[axis]:
                node = node.left
            else:
                node = node.right
        return node

    def kd_tree_knn(self, query_data, k):
        best_distances = []
        visited = []
        node = self.fake_dfs(query_data, self.tree.root)
        while True:
            visited.append(node)
            if node.is_leaf():
                for point in node.data:
                    dist = self.distance.calculate(query_data, point, self.n_dim)
                    if len(best_distances) < k:
                        heapq.heappush(best_distances, (dist, point.tolist()))
                        heapq._heapify_max(best_distances)
                    else:
                        maxmin = heapq.heappop(best_distances)
                        if dist < maxmin[0]:
                            heapq.heappush(best_distances, (dist, point.tolist()))
                        else:
                            heapq.heappush(best_distances, maxmin)
                        heapq._heapify_max(best_distances)
            else:
                dist = self.distance.calculate(query_data, node.median, self.n_dim)
                if len(best_distances) < k:
                    heapq.heappush(best_distances, (dist, node.median))
                    heapq._heapify_max(best_distances)
                else:
                    maxmin = heapq.heappop(best_distances)
                    if dist < maxmin[0]:
                        heapq.heappush(best_distances, (dist, node.median))
                    else:
                        heapq.heappush(best_distances, maxmin)
                    heapq._heapify_max(best_distances)
            if node.parent is None:
                break
            else:
                axis = node.parent.axis
                if len(best_distances) < k or abs(node.parent.median[axis] - query_data[axis]) <= best_distances[0][0]:
                    if node.side == 0:
                        node = self.fake_dfs(query_data, node.parent.right)
                    else:
                        node = self.fake_dfs(query_data, node.parent.left)
                    while node in visited:
                        node = node.parent
                    continue
            node = node.parent
            
        best_distances = sorted(best_distances)
        cnt1 = np.sum([1 for x in best_distances if x[1][-1] == 1])
        print("K({}) smallest distances [kd_tree]: {}".format(k, [x[0] for x in best_distances]))
        print("K({}) nearest points [kd_tree]: {}".format(k, [x[1] for x in best_distances]))
        return 1 if cnt1 > k // 2 else 0

class BallTree:
    def __init__(self, distance, X = None, y = None, leaf_size = 1):
        self.X = X
        self.y = y
        self.leaf_size = leaf_size
        self.distance = distance
        self.fit(X, y)

    def fit(self, X, y, leaf_size = 1):
        if not X is None and not y is None:
            self.X = X
            self.y = y
            self.n_dim = X.shape[1]
            dataset = pd.DataFrame(self.X) 
            dataset.columns = ['X{}'.format(i+1) for i in range(X.shape[1])]
            dataset['y'] = self.y
            self.dataset = dataset
            self.tree = BallTree_(self.construct_tree())
    
    def construct_tree(self):
        data_array = self.dataset.to_numpy()
        return self.construction(data_array)

    def construction(self, data_array):
        if len(data_array) <= self.leaf_size:
            centroid = np.mean(data_array, axis = 0)
            ball = Ball(centroid, data_array)
            ball.calculate_radius(self.distance, self.n_dim)
            return BallNode(ball)
        elif len(data_array) == 0:
            return
        spread = [np.ptp(data_array[:, i]) for i in range(data_array.shape[1])]
        max_spread_idx = np.argmax(spread)
        sorted_indices = np.argsort(data_array[:, max_spread_idx])
        data_array = data_array[sorted_indices]
        median_index = len(data_array) // 2 if len(data_array) % 2 == 0 else (len(data_array) - 1) // 2   
        centroid = np.mean(data_array, axis = 0)
        ball = Ball(centroid, data_array)
        ballNode = BallNode(ball)
        ballNode.lt = self.construction(data_array[:median_index])
        ballNode.rt = self.construction(data_array[median_index:])
        ball.calculate_radius(self.distance, self.n_dim)
        return ballNode
    
    def query(self, query_data, k):
        best_distances = self.ball_tree_knn(query_data, k, self.tree.root, [])
        cnt1 = np.sum([1 for x in best_distances if x[1][-1] == 1])
        best_distances = sorted(best_distances)
        print("K({}) smallest distances [ball_tree]: {}".format(k, [x[0] for x in best_distances]))
        print("K({}) nearest points [ball_tree]: {}".format(k, [x[1] for x in best_distances]))
        return 1 if cnt1 > k // 2 else 0

    def ball_tree_knn(self, query_data, k, ball_node, best_distances = []):
        heapq._heapify_max(best_distances)
        dist_query_center = self.distance.calculate(query_data, ball_node.bl.ctr, self.n_dim)
        worst_best_point = best_distances[0][1] if best_distances else 0
        dist_query_max_best = self.distance.calculate(query_data, worst_best_point, self.n_dim) if best_distances else 0
        if len(best_distances) == k and dist_query_center - ball_node.bl.r >= dist_query_max_best: return best_distances
        elif ball_node.is_leaf():
            for p in ball_node.bl.pts:
                p_dist = self.distance.calculate(p, query_data, self.n_dim)
                if p_dist < dist_query_max_best or len(best_distances) < k:
                    if len(best_distances) == k:
                        heapq.heappop(best_distances)
                    heapq.heappush(best_distances, (p_dist, p.tolist()))
                    heapq._heapify_max(best_distances)
                    if worst_best_point == 0 or best_distances[0][1] != worst_best_point:
                        worst_best_point = best_distances[0][1]
                        dist_query_max_best = self.distance.calculate(query_data, worst_best_point, self.n_dim)
        else:
            flag = 0
            if ball_node.rt is None:
                flag = 1
            dist_lt = self.distance.calculate(ball_node.lt.bl.ctr, query_data, self.n_dim)
            if not flag:
                dist_rt = self.distance.calculate(ball_node.rt.bl.ctr, query_data, self.n_dim)
                if dist_lt > dist_rt:
                    child1 = ball_node.rt   
                    child2 = ball_node.lt
                else:
                    child1 = ball_node.lt
                    child2 = ball_node.rt
                self.ball_tree_knn(query_data, k, child1, best_distances)
                self.ball_tree_knn(query_data, k, child2, best_distances)
            else:
                self.ball_tree_knn(query_data, k, ball_node.lt, best_distances)

        return best_distances

class KNN:
    def __init__(self, n_neighbors = 5, algorithm='brute', metric = 'euclidean', leaf_size = 30, n_power = 2):
        self.n_neighbors = n_neighbors
        self.distance = Distance(metric, n_power)
        self.algorithm = algorithm
        if self.algorithm == 'brute':
            self.model = BruteForce(self.distance)
        elif self.algorithm == 'kd_tree':
            self.model = KDTree(self.distance)
        elif self.algorithm == 'ball_tree':
            self.model = BallTree(self.distance)
        else:
            raise ValueError(f"Invalid algorithm: {algorithm}")
        self.leaf_size = leaf_size

    def fit(self, X, y):
        self.model.fit(X, y, self.leaf_size)

    def predict(self, query_data):
        outputs = []
        for data in query_data:
            outputs.append(self.model.query(data, self.n_neighbors))
        return np.array(outputs)



################################################################################

def my_data():
    array = np.array([[-1,-2, 0], [3, 5, 1], [-2, 1, 0], 
                  [-2, 0, 1], [0, 4, 1], [1, 8, 1], 
                  [2, 2, 0], [2, 5, 0], [3, 1, 1], [4, 0, 1]])
    #array = np.array([[-1, -1, 0], [-4, -4, 0], [-3, -3, 0], [-2, -2, 0], [0,0,0], [1, 1, 1], [2, 2, 1], [3, 3, 1], [4, 4, 1]])
    # creating a list of column names
    column_values = ['X1', 'X2', 'y']
    # creating the dataframe
    df = pd.DataFrame(data = array,
                      columns = column_values)
    return df

def main():

    #---UNCOMMENT FOR RANDOM DATASETS
    X, y = make_classification(n_samples = 576, n_features=4, n_informative=2, n_redundant=0, n_classes=2)
    dataset = pd.DataFrame(X) 
    dataset.columns = ['X1', 'X2', 'X3', 'X4']
    dataset['y'] = y

    #---UNCOMMENT TO SAVE RANDOM DATA
    #dataset.to_pickle("dataset.pkl")
    #np.savetxt('X2.csv', X, delimiter=',')
    #np.savetxt('y2.csv', y, delimiter=',')


    #---UNCOMMENT TO READ SAVED DATASETS
    #dataset = pd.read_pickle("dataset.pkl")
    #X = np.loadtxt('X2.csv', delimiter=',')
    #y = np.loadtxt('y2.csv', delimiter=',')

    #---UNCOMMENT FOR CUSTOM DATASET
    #dataset = my_data()
    #X = np.array([[-1,-2], [3, 5], [-2, 1], 
    #              [-2, 0], [0, 4], [1, 8], 
    #           [2, 2], [2, 5], [3, 1], [4, 0]])
    #y = dataset['y']
    #X = np.array([[-1, -1], [-4, -4], [-3, -3], [-2, -2], [0,0], [1, 1], [2, 2], [3, 3], [4, 4]])

    k = 7

    #data_array = dataset[["X1","X2","y"]].to_numpy()

    print("----------------------")
    query_data = np.array([10, 0, -1, -6])

    knn_brute = KNN(algorithm='brute', metric = 'euclidean', n_neighbors=k)
    knn_brute.fit(X, y)
    print(knn_brute.predict([query_data]))
    start = time.time()
    print(time.time() - start)

    knn_kd_tree = KNN(algorithm='kd_tree', leaf_size = 30, metric = 'euclidean', n_neighbors= k)
    knn_kd_tree.fit(X, y)
    print(knn_kd_tree.predict([query_data]))
    start = time.time()
    print(time.time() - start)

    knn_ball_tree = KNN(algorithm='ball_tree', leaf_size = 30, metric = 'euclidean', n_neighbors=k)
    knn_ball_tree.fit(X, y)
    print(knn_ball_tree.predict([query_data]))
    start = time.time()
    print(time.time() - start)

main()