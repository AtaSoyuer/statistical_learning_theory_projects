#ATA BARTU SOYUER
#LEGI: 20-946-257

import sklearn as skl
from sklearn.utils.validation import check_is_fitted
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_blobs
import sklearn.svm as svm
from sklearn import cluster
import warnings
import copy


import pandas as pd
import numpy as np
with warnings.catch_warnings():
    warnings.filterwarnings("ignore",category=DeprecationWarning)
    from treelib import Tree

import time

import matplotlib.pyplot as plt




def read_data_csv(sheet, y_names=None):
    """Parse a column data store into X, y arrays

    Args:
        sheet (str): Path to csv data sheet.
        y_names (list of str): List of column names used as labels.

    Returns:
        X (np.ndarray): Array with feature values from columns that are not
        contained in y_names (n_samples, n_features)
        y (dict of np.ndarray): Dictionary with keys y_names, each key
        contains an array (n_samples, 1) with the label data from the
        corresponding column in sheet.
    """

    data = pd.read_csv(sheet)
    feature_columns = [c for c in data.columns if c not in y_names]
    X = data[feature_columns].values
    y = dict([(y_name, data[[y_name]].values) for y_name in y_names])
    # print(X)
    # print(y)

    return X, y


#read_data_csv('wine-data.csv',('quality','color'))


class DeterministicAnnealingClustering(skl.base.BaseEstimator,
                                       skl.base.TransformerMixin):
    """Template class for DAC

    Attributes:
        cluster_centers (np.ndarray): Cluster centroids y_i
            (n_clusters, n_features)
        cluster_probs (np.ndarray): Assignment probability vectors
            p(y_i | x) for each sample (n_samples, n_clusters)
        bifurcation_tree (treelib.Tree): Tree object that contains information
            about cluster evolution during annealing.

    Parameters:
        n_clusters (int): Maximum number of clusters returned by DAC.
        random_state (int): Random seed.
    """

    def __init__(self, n_clusters=8, random_state=42, T_min_scaler=10, cooling_parameter = 0.99, metric="euclidian"):
        self.n_clusters = n_clusters
        self.random_state = random_state
        self.metric = metric
        self.T = None   #Initialized in fit()
        self.T_min = None #Emprically chosen from paper
        self.T_min_scaler = T_min_scaler
        self.cooling_param = cooling_parameter #Emprically chosen
        self.sigma = 0.05

        self.cluster_centers = None #Initialized in fit()
        self.effective_centers = None #Initialized in fit()
        self.cluster_probs = None
        self.p_i_array = np.array([1],dtype='float64') #Single centroid initially, p(y1)=1, new elements inserted later

        self.n_eff_clusters = list()
        self.temp = list()
        self.distortion = list()

        self.bifurcation_tree = Tree()

        # Not necessary, depends on your implementation
        self.bifurcation_tree_cut_idx = None

        # Add more parameters, if necessary. You can also modify any other
        # attributes defined above

    def fit(self, samples):
        """Compute DAC for input vectors X

        Preferred implementation of DAC as described in reference [1].

        Args:
            samples (np.ndarray): Input array with shape (n_samples, n_features)
        """
        # TODO:

        # Initialize centroids as sample avg
        sample_avg = (1 / samples.shape[0]) * np.sum(samples, axis=0)
        print(sample_avg)
        self.cluster_centers = np.array([sample_avg, ] * self.n_clusters)
        self.effective_centers = np.unique(self.cluster_centers, axis=0)  # Only the K distinct centroids remain
        #print(self.effective_centers)

        #Initialize tempurature as above 2lamdamax
        de_mean = samples - np.repeat(sample_avg[np.newaxis,:], repeats=samples.shape[0], axis=0)
        sample_cov = (1/samples.shape[0])*np.matmul(np.transpose(de_mean), de_mean)
        w, v = np.linalg.eigh(sample_cov)
        T_critical = 2 * np.max(w)
        self.T = 1.2* T_critical
        print('T_critical is:',T_critical)
        #if(samples.shape[1]==2):
        self.T_min = T_critical.copy()/self.T_min_scaler
        #else:
            #self.T_min = T_critical.copy()/1000
        #self.T_min = 1

        self.bifurcation_tree.create_node('First Centroid', 'first_centroid', data ={
            'cluster_id': 0, 'centroid':self.effective_centers[0], 'distance': np.array(np.linalg.norm(self.effective_centers[0], ord=2)), 'splitted': False})  # Root node
        #print(self.bifurcation_tree.get_node('first_centroid').data['centroid'])
        #The effective centroids have 2 codevectors superposed at each distinct center
        ith_child = 0
        if self.metric == "euclidian":
            np.random.seed(self.random_state)#IMPORTANT!!!!
            number_of_splits = 0 #To monitor centroid2
            while self.T > self.T_min:
                # print(self.effective_centers.shape)
                convergence = False
                counter = 0
                while (convergence == False):
                    # print(counter)
                    distance = self.get_distance(samples.copy(), self.effective_centers.copy())
                    self.cluster_probs = self._calculate_cluster_probs(distance.copy(), self.T.copy())

                    self.p_i_array = (1.0 / samples.shape[0]) * np.sum(self.cluster_probs, axis=0)

                    temp_matrix = np.multiply(np.tile(samples, (1, self.effective_centers.shape[0])),
                                              np.repeat(self.cluster_probs, repeats=samples.shape[1], axis=1))
                    # print(temp_matrix.shape)
                    temp_matrix_2 = (1.0 / samples.shape[0]) * np.sum(temp_matrix, axis=0)
                    #print(temp_matrix_2.shape)
                    y_new = temp_matrix_2.reshape((self.effective_centers.shape[0], samples.shape[1]))/self.p_i_array[:,None]
                    #print(y_new.shape)
                    y_old = self.effective_centers.copy()
                    for centroid_idx in range(self.effective_centers.shape[0]):
                        self.effective_centers[centroid_idx,:] = y_new.copy()[centroid_idx,:]  # Update ith distinct centroid yo yt+1
                    convergence = np.allclose(y_new, y_old)  # Check convergence
                    # print(y_new)
                    counter = counter + 1
                #print(counter)

                self.temp.append(self.T.copy())
                self.n_eff_clusters.append(self.effective_centers.shape[0])


                assigned_clusters = np.argmax(self.cluster_probs,axis=1)   #Array for each sample with each element as assigned cluster index
                expected_dist = 0.0
                for i in range(samples.shape[0]):
                    expected_dist = expected_dist + distance[i,assigned_clusters[i]]

                expected_dist = expected_dist/samples.shape[0]

                self.distortion.append(expected_dist)

                #Create as many dicts as the number of effective clusters each time a split occurs containing the position of the centroid
                #and an array of distances. Use centroid_num to identify the parent that has been splitted and to create two child nodes
                #and stop updating the node that currently corresponds to centroid_num which has been splitted. update the centroids and
                #distances at each tempurature for the plot



                #print(self.T)
                #print(self.effective_centers)

                if(self.effective_centers.shape[0] < self.n_clusters):  #Since always two duplicates in one centroid
                    centroid_num = 0
                    is_critical = False
                    temp_array = self.effective_centers.copy()
                    self.p_i_array = (np.sum(self.cluster_probs, axis=0))/(samples.shape[0])
                    temp_p_i_array = self.p_i_array.copy()
                    while centroid_num < (self.effective_centers.shape[0]):
                        #print(centroid_num)
                        difference = samples - np.repeat(self.effective_centers[centroid_num,:][np.newaxis,:], repeats=samples.shape[0], axis=0)
                        C_x_y = np.zeros((samples.shape[1],samples.shape[1]))
                        #for i in range(samples.shape[0]):
                        multiplicant = np.multiply(np.sqrt(self.cluster_probs[:,centroid_num]/self.p_i_array[centroid_num])[:,np.newaxis],difference)
                        C_x_y = (1.0/samples.shape[0]) * np.matmul(np.transpose(multiplicant),multiplicant)
                        #print('Cxy is',C_x_y)

                        w, v = np.linalg.eig(C_x_y)
                        T_critical = 2 * np.max(w)



                        if(self.T <= T_critical):    #SElf-handling, since new clusters will be added, cxy will change?
                        #CORRECTION

                            is_critical = True
                            #Add new perturbed centroid
                            print('Tcritical is', T_critical)
                            #print(self.cluster_probs.shape)
                            #print(is_critical)
                            splitted_centroid_num = copy.copy(centroid_num)
                            temp_array = np.insert(temp_array, splitted_centroid_num+1, temp_array[splitted_centroid_num,:] +self.sigma*np.random.rand(self.effective_centers.shape[1]), 0)


                            #Update the fraction probabilities: half the original probability and assign it to the newly emerged centroid
                            temp_value = temp_p_i_array[splitted_centroid_num].copy()/2
                            temp_p_i_array[splitted_centroid_num] = temp_value
                            temp_p_i_array = np.insert(temp_p_i_array.copy(), splitted_centroid_num+1, temp_p_i_array[splitted_centroid_num].copy())
                            #print('P_i is:',self.p_i_array)
                            print(temp_p_i_array)
                            #print(self.effective_centers.shape)

                        centroid_num = centroid_num + 1


                    #print(self.effective_centers)
                    if (is_critical == True):
                       #print('critical is true')
                        number_of_splits = number_of_splits + 1
                        self.effective_centers = temp_array.copy()
                        self.p_i_array = temp_p_i_array
                        print(self.effective_centers.shape)


                        for node in list(self.bifurcation_tree.all_nodes_itr()):
                            #print(node)
                            #if (self.effective_centers.shape[0] == 5):
                                #print(self.effective_centers[3])
                                #print('spliitednum',splitted_centroid_num)
                                #print(self.effective_centers[splitted_centroid_num])
                            #print(splitted_centroid_num)
                            if (np.isclose(node.data['centroid'], self.effective_centers[splitted_centroid_num]).all()):
                                #print('node found')
                                ith_child = ith_child + 1
                                self.bifurcation_tree.create_node('Centroid'+str(ith_child), 'centroid'+str(ith_child), parent = node,
                                                        data = {'cluster_id': ith_child, 'centroid':self.effective_centers[copy.deepcopy(splitted_centroid_num)], 'centroid_idx':copy.deepcopy(splitted_centroid_num),'distance': np.zeros(1), 'splitted': False} )
                                ith_child = ith_child + 1
                                self.bifurcation_tree.create_node('Centroid' + str(ith_child), 'centroid' + str(ith_child), parent=node,
                                                        data = {'cluster_id': ith_child, 'centroid':self.effective_centers[copy.deepcopy(splitted_centroid_num)+1], 'centroid_idx':copy.deepcopy(splitted_centroid_num)+1, 'distance': np.zeros(1), 'splitted': False} )

                                node.data['splitted'] = True

                        #if(number_of_splits==2):
                            #self.bifurcation_tree.get_node('centroid1').data['centroid'] = self.effective_centers[0]
                        modified_identifiers = []
                        for node in list(self.bifurcation_tree.all_nodes_itr()):
                            #print(node)
                            if (node.data['splitted'] == False and
                                    np.where((self.effective_centers == tuple(node.data['centroid'])).all(axis=1))[
                                        0] != splitted_centroid_num
                                    and
                                    np.where((self.effective_centers == tuple(node.data['centroid'])).all(axis=1))[
                                        0] != (splitted_centroid_num + 1)):
                                #print(splitted_centroid_num)
                                #print('corresponding centroid', self.effective_centers[np.where(
                                    #(self.effective_centers == tuple(node.data['centroid'])).all(axis=1))[0],:])
                                modified_identifiers.append(node.identifier)

                        for id in modified_identifiers:
                            idx = np.where(
                                    (self.effective_centers == tuple(self.bifurcation_tree.get_node(id).data['centroid'])).all(axis=1))[0][0]
                            #print(idx)
                            self.bifurcation_tree.get_node(id).data['centroid'] = self.effective_centers[idx]
                            #print(self.bifurcation_tree.get_node(id).data['centroid'])


                            #print(self.bifurcation_tree.get_node('centroid2').data['centroid']==self.effective_centers[2])
                                #print(node)
                                #print('Children created')
                                #print(self.bifurcation_tree.get_node(self.bifurcation_tree.get_node('centroid1').bpointer).data)
                                #print(self.bifurcation_tree.all_nodes())
                                #print(self.bifurcation_tree)
                                #print(self.bifurcation_tree.all_nodes_itr())
                        #print(self.bifurcation_tree.all_nodes()) #FOR DEBUGGINNNGGGG!!!!!
                        #print(self.effective_centers)
                            #elif(node.data['splitted']==False and node.bpointer == 'first_centroid'):
                                #node[]
                for node in self.bifurcation_tree.all_nodes_itr():
                    #print(node)

                    if (node.data['splitted'] == False):
                        if (node.data['cluster_id'] != 0):
                            #print('yes')
                            node.data['distance'] = np.append(node.data['distance'], np.array(np.linalg.norm(node.data['centroid']-self.bifurcation_tree.get_node(node.predecessor(self.bifurcation_tree.identifier)).data['centroid'], ord=2)))
                            warnings.filterwarnings("ignore", category=DeprecationWarning)
                    #print('completed')
                #print(self.bifurcation_tree.all_nodes())
                self.T = copy.copy(self.T) * self.cooling_param
            print(self.bifurcation_tree)
            #print(self.bifurcation_tree.all_nodes())

        elif self.metric == "ratioscale":
            pass

        self.cluster_centers = self.effective_centers.copy()  # Finally the fitted effective centers are assigned to cluster centers for prediction
        #print(self.cluster_centers)

    def _calculate_cluster_probs(self, dist_mat, temperature):
        """Predict assignment probability vectors for each sample in X given
            the pairwise distances

        Args:
            dist_mat (np.ndarray): Distances (n_samples, n_centroids)
            temperature (float): Temperature at which probabilities are
                calculated

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        # TODO:
        nominator = np.exp(-dist_mat/temperature) * self.p_i_array.copy() #Nominator is still a matrix since we compute for all xi
        row_sums = nominator.sum(axis=1)  #/2 since overlapping cetroids don't actually affect normalization

        #print(np.max(nominator))
        probs = (skl.preprocessing.normalize((nominator), norm='l1', axis=1))
        #probs = nominator / row_sums[:, np.newaxis]
        #probs = np.array(probs)

        #print(probs)
        #print(self.T)

        return probs

    def get_distance(self, samples, clusters):
        """Calculate the distance matrix between samples and codevectors
        based on the given metric

        Args:
            samples (np.ndarray): Samples array (n_samples, n_features)
            clusters (np.ndarray): Codebook (n_centroids, n_features)

        Returns:
            D (np.ndarray): Distances (n_samples, n_centroids)
        """

        dist_mat = np.zeros((samples.shape[0], clusters.shape[0]))
        # TODO:
        #if self.metric == "euclidian":
            #for i in range(samples.shape[0]):
                #for j in range(clusters.shape[0]):
                    #dist_mat[i,j] = np.square(np.linalg.norm(samples[i,:]-clusters[j,:],ord=2))

        if self.metric == 'euclidian':

            cluster_num = clusters.shape[0]
            sample_num = samples.shape[0]
            dist_mat = np.zeros((sample_num, cluster_num))

            #for i in range(cluster_num):
            temp_matrix_dist = np.tile(samples, (cluster_num,1)) - np.repeat(clusters, repeats = sample_num, axis=0)
            #print(temp_matrix_dist.shape)
            temp_matrix_dist2 = np.square(np.linalg.norm(temp_matrix_dist, ord=2, axis=1))
            #print(temp_matrix_dist2.shape)
            dist_mat = np.transpose(np.reshape(temp_matrix_dist2.transpose(), (cluster_num, sample_num)))

        elif self.metric == "ratioscale":
            pass

        #print(dist_mat)
        return dist_mat

    def predict(self, samples):
        """Predict assignment probability vectors for each sample in X.

        Args:
            samples (np.ndarray): Input array with shape (new_samples, n_features)

        Returns:
            probs (np.ndarray): Assignment probability vectors
                (new_samples, n_clusters)
        """
        distance_mat = self.get_distance(samples, self.cluster_centers)
        probs = self._calculate_cluster_probs(distance_mat, self.T_min)
        #print(self.cluster_centers)
        #plt.scatter(self.cluster_centers[:,0],self.cluster_centers[:,1])
        return probs

    def transform(self, samples):
        """Transform X to a cluster-distance space.

        In the new space, each dimension is the distance to the cluster centers

        Args:
            samples (np.ndarray): Input array with shape
                (new_samples, n_features)

        Returns:
            Y (np.ndarray): Cluster-distance vectors (new_samples, n_clusters)
        """
        check_is_fitted(self, ["cluster_centers"])

        # Your code goes here
        distance_mat = self.get_distance(samples, self.cluster_centers)
        return distance_mat

    def plot_bifurcation(self):
        """Show the evolution of cluster splitting

        This is a pseudo-code showing how you may be using the tree
        information to make a bifurcation plot. Your implementation may be
        entire different or based on this code.
        """
        check_is_fitted(self, ["bifurcation_tree"])

        # clusters = [[] for _ in range(len(np.unique(self.n_eff_clusters)))]
        # for node in self.bifurcation_tree.all_nodes_itr():
        #     c_id = node.data['cluster_id']
        #     my_dist = node.data['distance']
        #
        #     if c_id > 0 and len(clusters[c_id]) == 0:
        #         clusters[c_id] = list(np.copy(clusters[c_id-1]))
        #     clusters[c_id].append(my_dist)
        #
        # # Cut the last iterations, usually it takes too long
        # cut_idx = self.bifurcation_tree_cut_idx + 20

        dist_c0_c1 = self.bifurcation_tree.get_node('first_centroid').data['distance']*np.ones(len(self.bifurcation_tree.get_node('centroid1').data['distance'])) - self.bifurcation_tree.get_node('centroid1').data['distance']
        dist_c0_c2 = self.bifurcation_tree.get_node('first_centroid').data['distance']*np.ones(len(self.bifurcation_tree.get_node('centroid2').data['distance'])) + self.bifurcation_tree.get_node('centroid2').data['distance']
        dist_c2_c3 = dist_c0_c2[-1]*np.ones(len(self.bifurcation_tree.get_node('centroid3').data['distance'])) + self.bifurcation_tree.get_node('centroid3').data['distance']
        dist_c2_c4 = dist_c0_c2[-1]*np.ones(len(self.bifurcation_tree.get_node('centroid4').data['distance'])) - self.bifurcation_tree.get_node('centroid4').data['distance']
        dist_c3_c5 = dist_c2_c3[-1]*np.ones(len(self.bifurcation_tree.get_node('centroid5').data['distance'])) + self.bifurcation_tree.get_node('centroid5').data['distance']
        dist_c3_c6 = dist_c2_c3[-1]*np.ones(len(self.bifurcation_tree.get_node('centroid6').data['distance'])) - self.bifurcation_tree.get_node('centroid6').data['distance']

        beta = [np.log(1 / t) for t in self.temp]
        temp_length = len(dist_c0_c1)
        #print(len(beta))
        #print(temp_length)
        dist_c0 = self.bifurcation_tree.get_node('first_centroid').data['distance']*np.ones(len(beta) - temp_length)

        branch1 = np.concatenate((dist_c0, dist_c0_c1))
        branch2 = np.concatenate((dist_c0, dist_c0_c2, dist_c2_c4[:-1:]))
        branch3 = np.concatenate((dist_c0, dist_c0_c2, dist_c2_c3[:-1:], dist_c3_c5[:-1:]))
        branch4 = np.concatenate((dist_c0, dist_c0_c2, dist_c2_c3[:-1:], dist_c3_c6[:-1:]))

        plt.figure(figsize=(10, 5))

        plt.plot(branch1, beta, 'r', label='Path of Cluster 1')
        plt.plot(branch2, beta, 'g', label='Path of Cluster 2')
        plt.plot(branch3, beta, 'b', label='Path of Cluster 3')
        plt.plot(branch4, beta, 'k', label='Path of Cluster 4')

        #plt.figure(figsize=(10, 5))
        # for c_id, s in enumerate(clusters):
        #     plt.plot(s[:cut_idx], beta[:cut_idx], '-k',
        #              alpha=1, c='C%d' % int(c_id),
        #              label='Cluster %d' % int(c_id))
        #print(self.bifurcation_tree.get_node('centroid1').data['centroid'])
        plt.legend()
        plt.xlabel("distance to parent")
        plt.ylabel(r'$1 / T$')
        plt.title('Bifurcation Plot')
        plt.show()



    def plot_phase_diagram(self):
        """Plot the phase diagram

        This is an example of how to make phase diagram plot. The exact
        implementation may vary entirely based on your self.fit()
        implementation. Feel free to make any modifications.
        """
        t_max = np.log(max(self.temp))
        d_min = np.log(min(self.distortion))
        y_axis = [np.log(i) - d_min for i in self.distortion]
        x_axis = [t_max - np.log(i) for i in self.temp]

        plt.figure(figsize=(12, 9))
        plt.plot(x_axis, y_axis)

        region = {}
        for i, c in list(enumerate(self.n_eff_clusters)):
            if c not in region:
                region[c] = {}
                region[c]['min'] = x_axis[i]
            region[c]['max'] = x_axis[i]
        for c in region:
            if c == 0:
                continue
            plt.text((region[c]['min'] + region[c]['max']) / 2, 0.2,
                     'K={}'.format(c), rotation=90)
            plt.axvspan(region[c]['min'], region[c]['max'], color='C' + str(c),
                        alpha=0.2)
        plt.title('Phases diagram (log)')
        plt.xlabel('Temperature')
        plt.ylabel('Distortion')
        plt.show()

