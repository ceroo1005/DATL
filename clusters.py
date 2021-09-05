import torch
import torch.nn.functional as F
from random import shuffle
import numpy as np

class CosineClusters():
    def __init__(self, num_clusters=100, Euclidean=False):
        self.clusters = []      # 储存各个集群
        self.item_cluster = {}  # 储存每个样本所属集群
        self.Euclidean = Euclidean

        # 初始化集群
        for i in range(0, num_clusters):
            self.clusters.append(Cluster(self.Euclidean))

    def add_random_training_items(self, items):
        '''随机分配样本给集群'''

        cur_index = 0
        for index, item in enumerate(items):
            self.clusters[cur_index].add_to_cluster(item)
            textid = item[0]
            self.item_cluster[textid] = self.clusters[cur_index]
            
            cur_index += 1
            if cur_index >= len(self.clusters):
                cur_index = 0 


    def add_items_to_best_cluster(self, items):
        """无监督聚类"""
        added = 0
        for item in items:
            new = self.add_item_to_best_cluster(item)
            if new:
                added += 1
                
        return added

    def add_item_to_best_cluster(self, item):     
        best_cluster = None 
        best_fit = float("-inf")        
        previous_cluster = None
        
        # 从当前集群中删除后再匹配
        textid = item[0]
        if textid in self.item_cluster:
            previous_cluster = self.item_cluster[textid]
            previous_cluster.remove_from_cluster(item)
            
        for cluster in self.clusters:
            fit = cluster.cosine_similary(item, Euclidean=self.Euclidean)
            if fit > best_fit:
                best_fit = fit
                best_cluster = cluster 
        
        # 重新匹配后得添加到最佳的样本库中
        best_cluster.add_to_cluster(item)
        self.item_cluster[textid] = best_cluster
        
        if best_cluster == previous_cluster:
            return False
        else:
            return True
 
 
    def get_items_cluster(self, item):  
        textid = item[0]
        
        if textid in self.item_cluster:
            return self.item_cluster[textid]
        else:
            return None      
        
        
    def get_centroids(self, number_per_cluster=1):  
        centroids = []
        for cluster in self.clusters:
            centroids.append(cluster.get_centroid(number_per_cluster))
        
        return centroids
    
        
    def get_outliers(self, number_per_cluster=1):  
        outliers = []
        for cluster in self.clusters:
            outliers.append(cluster.get_outlier(number_per_cluster))
        
        return outliers
 
         
    def get_randoms(self, number_per_cluster=1):  
        randoms = []
        for cluster in self.clusters:
            randoms.append(cluster.get_random_members(number_per_cluster))
        
        return randoms
   
      
    def shape(self):  
        lengths = []
        for cluster in self.clusters:
            lengths.append(cluster.size())
        
        return str(lengths)



class Cluster():

    def __init__(self, Euclidean = False):
        self.members = {}          # 该集群中样本ID
        self.feature_vector = None # 该集群整体特征
        self.distance = []         # 集群中的样本到该集群中心的距离
        self.Euclidean = Euclidean


    def add_to_cluster(self, item):
        dataid = item[0]
        data = item[1]

        self.members[dataid] = item  
        try:
            if  self.feature_vector == None:
                self.feature_vector = data
        except:
            self.feature_vector = self.feature_vector + data
        
            
    def remove_from_cluster(self, item):
        """从集群中删除某一个元素"""
        dataid = item[0]
        data = item[1]
        
        exists = self.members.pop(dataid, False)
        if exists:
            self.feature_vector = self.feature_vector - data
    
    
    def cosine_similary(self, item, Euclidean=False):
        '''计算某样本距离集群中心的余弦距离'''
        data = item[1]        
        center_vec = self.feature_vector / len(list(self.members.keys()))

        item_tensor = torch.FloatTensor(data)
        center_tensor = torch.FloatTensor(center_vec)
        
        if Euclidean:
            # print('欧式距离',end='\r')
            similarity = - np.sqrt(np.sum(np.square(data - center_vec)))
            return similarity
        else:
            # print('余弦距离',end='\r')
            similarity = F.cosine_similarity(item_tensor, center_tensor, 0)
            return similarity.item() # item() converts tensor value to float
    
    
    def size(self):
        return len(self.members.keys())
 
 
    def distance_sort(self):
        self.distance = []
        for textid in self.members.keys():
            item = self.members[textid]
            similarity = self.cosine_similary(item, Euclidean=self.Euclidean)
            self.distance.append([similarity, item[0], item[1]])
        self.distance.sort(reverse=True, key=lambda x: x[0])
        return self.distance

    def get_centroid(self, number=1):
        if len(self.members) == 0:
            return []
        return self.distance_sort()[:number]

    def get_outlier(self, number=1):
        if len(self.members) == 0:
            return []
        return self.distance_sort()[-number:]

    def get_random_members(self, number=1):
        if len(self.members) == 0:
            return []        
        _ = self.distance_sort()
        randoms = []
        for i in range(0, number):
            randoms.append(_[np.random.randint(len(self.members))])
                
        return randoms

         

if __name__ == '__main__':

    from sklearn.datasets import make_blobs
    n_samples = 1500
    random_state = 170
    X, y = make_blobs(n_samples=n_samples, random_state=random_state)
    num_clusters = 4
    max_epochs = 10
    data = X


    NEWdata = [[str(index), item] for index, item in enumerate(data)]
    # shuffle(NEWdata)
    # print(NEWdata)
    # raise 'pass'
    # shuffle(NEWdata)
    cosine_clusters = CosineClusters(num_clusters, Euclidean=True)
    cosine_clusters.add_random_training_items(NEWdata)
    for index, cluster in enumerate(cosine_clusters.clusters):
        print(cluster.feature_vector)
    print(set(cosine_clusters.item_cluster.values()))


    for i in range(0, max_epochs):
        print("Epoch "+str(i))
        added = cosine_clusters.add_items_to_best_cluster(NEWdata)
        if added == 0:
            break

    # centroids_per = list(set(cosine_clusters.item_cluster.values()))
    sample_y = [cosine_clusters.clusters.index(_) for _ in cosine_clusters.item_cluster.values()]
    # print(sample_y)

    centroids = cosine_clusters.get_centroids(2)
    outliers = cosine_clusters.get_outliers(2)
    randoms = cosine_clusters.get_randoms(2)

    centroids + outliers + randoms
    # print(set(cosine_clusters.item_cluster.values()))
    # print(cosine_clusters.clusters)

    for index, cluster in enumerate(cosine_clusters.clusters):
        sample_sort = cluster.distance_sort()
        # print('centroids:\t',centroids[index])
        # print('outliers:\t',outliers[index])
        # print('randoms:\t',randoms[index])
        # assert sample_sort[0][1] == centroids[index][0]
        # assert sample_sort[-1][1] == outliers[index][0]

    D_id_color = [u'orchid', u'darkcyan', u'dodgerblue', u'turquoise', u'darkviolet']
    import matplotlib.pyplot as plt


    plt.figure(figsize=(18, 6))

    plt.subplot(131)
    plt.scatter(X[:, 0], X[:, 1])
    
    plt.subplot(132)
    for label in [*range(len(cosine_clusters.clusters))]:
        indices = [i for i, l in enumerate(sample_y) if l == label]
        current_tx = np.take(data[:, 0], indices)
        current_ty = np.take(data[:, 1], indices)
        color = D_id_color[label]
        print(current_tx.shape)
        plt.scatter(current_tx, current_ty, c=color, label=label)
    plt.legend(loc='best')

    plt.subplot(133)
    plt.scatter(X[:, 0], X[:, 1], alpha=0.2, color='gray')
    f2 = lambda x:[_[2] for _ in x]
    for label in [*range(len(cosine_clusters.clusters))]:
        color = D_id_color[label]
        plt.scatter(np.array(f2(centroids[label]))[:,0], np.array(f2(centroids[label]))[:,1], c=color, label=f'{label} centroids')
        plt.scatter(np.array(f2(outliers[label]))[:,0], np.array(f2(outliers[label]))[:,1], marker='*', c=color, label=f'{label} outliers')
        plt.scatter(np.array(f2(randoms[label]))[:,0], np.array(f2(randoms[label]))[:,1], marker='^', c=color, label=f'{label} randoms')
            
        # sample_sort = cluster.distance_sort()
        # print('centroids:\t',centroids[index])
        # print('outliers:\t',outliers[index])
        # print('randoms:\t',randoms[index])


    # for index, cluster in enumerate(cosine_clusters.clusters):
    #     for item in outliers[index]:
    #         plt.scatter(item[-1][0], current_ty, c=color, label=label)
    
    plt.legend(loc='best')
    plt.show()