from sklearn.neighbors import NearestNeighbors
import cv2
import numpy as np

class NovelSearch(object):    
    def __init__(self, k):
        self.k = k
        self.nbrs = NearestNeighbors(n_neighbors=k, algorithm='kd_tree')
        self.X = []
        self.d = 0

    def fit(self, new_data):           
        new_data = self.flatten(new_data)
        new_data = [obs for obs in new_data if np.random.rand() > 0.95]
        if self.d == 0:
            self.d = new_data[0].size
        self.X = [x for x in self.X if np.random.rand() > 0.1] #throw out 1/10 of old data at random to keep the size down
        self.X += new_data
        self.nbrs.fit(self.X)

    def score(self, obss):
        obss = self.flatten(obss)
        if len(self.X) == 0:
            return np.zeros([len(obss),1])
        else:
            neighbors = self.nbrs.kneighbors(obss, return_distance=True)[0]
            return np.sum(np.square(neighbors), 1) / (self.k * self.d * self.d)

    def flatten(self, obss):
        return [np.reshape(cv2.resize(obs, None, fx=0.25, fy=0.25), [-1]) for obs in obss] #subsample by 1/4 on each axis
