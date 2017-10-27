from numpy.random import RandomState
import operator
import sys
import time

from scipy.special import gammaln

import numpy as np
import scipy.sparse as sp


class DpMixtureGibbs:

    def __init__(self, data, alpha, beta, nsamps=200, burnin=100, seed=-1):

        self.word_counts_list = [np.array(x) for x in data]
        self.W = len(self.word_counts_list[0])
        self.N = len(data)
        assert self.N == len(self.word_counts_list)

        self.alpha = alpha
        self.beta = beta

        self.nsamps = nsamps
        self.burn_in = burnin
        self.seed = int(seed)
        if self.seed > 0:
            self.random_state = RandomState(self.seed)
        else:
            self.random_state = RandomState()

        self.Z = None
        self.ZZ_all = sp.lil_matrix((self.N, self.N), dtype=np.float)
        self.samples_obtained = 0


    def run(self):

        print "Sampling begins"
        sys.stdout.flush()

        # initialise all rows under one cluster
        K = 1
        cluster_counts = np.array([float(self.N)])
        all_word_counts = np.zeros(self.W)
        for wc in self.word_counts_list:
            all_word_counts += wc
        cluster_word_sums = [all_word_counts]
        current_ks = np.zeros(self.N, dtype=np.int)

        # start sampling
        self.samples_obtained = 0
        for s in range(self.nsamps):

            start_time = time.time()

            # loop through the objects in random order
            random_order = range(self.N)
            self.random_state.shuffle(random_order)
            processed = 0
            for n in random_order:

                processed += 1
                sys.stdout.flush()

                current_word_counts = self.word_counts_list[n]
                k = current_ks[n] # the current cluster of this item

                # remove from model, detecting empty table if necessary
                cluster_counts[k] = cluster_counts[k] - 1
                cluster_word_sums[k] = cluster_word_sums[k] - current_word_counts

                # if empty table, delete this cluster
                if cluster_counts[k] == 0:
                    K = K - 1
                    cluster_counts = np.delete(cluster_counts, k) # delete k-th entry
                    del cluster_word_sums[k]
                    current_ks = self._reindex(k, current_ks) # remember to reindex all the clusters

                # compute prior probability for K existing table and new table
                prior = np.array(cluster_counts)
                prior = np.append(prior, self.alpha)
                prior = prior / prior.sum()

                log_likelihood = np.zeros_like(prior)
                for k_idx in range(K): # the finite portion
                    wcb = cluster_word_sums[k_idx] + self.beta
                    log_likelihood[k_idx] = self._C(wcb+current_word_counts) - self._C(wcb)
                # the infinite bit
                wcb = np.zeros(self.W) + self.beta
                log_likelihood[-1] = self._C(wcb+current_word_counts) - self._C(wcb)

                # sample from posterior
                post = log_likelihood + np.log(prior)
                post = np.exp(post - post.max())
                post = post / post.sum()
                random_number = self.random_state.rand()
                cumsum = np.cumsum(post)
                new_k = 0
                for new_k in range(len(cumsum)):
                    c = cumsum[new_k]
                    if random_number <= c:
                        break

                # (new_k+1) because indexing starts from 0 here
                if (new_k+1) > K:
                    # make new cluster and add to it
                    K = K + 1
                    cluster_counts = np.append(cluster_counts, 1)
                    cluster_word_sums.append(current_word_counts)
                else:
                    # put into existing cluster
                    cluster_counts[new_k] = cluster_counts[new_k] + 1
                    cluster_word_sums[new_k] = cluster_word_sums[new_k] + current_word_counts

                # assign object to the cluster new_k, regardless whether it's current or new
                current_ks[n] = new_k

                assert len(cluster_counts) == K, "len(cluster_counts)=%d != K=%d)" % (len(cluster_counts), K)
                assert len(cluster_word_sums) == K, "len(cluster_word_sums)=%d != K=%d)" % (len(cluster_word_sums), K)
                assert current_ks[n] < K, "current_ks[%d] = %d >= %d" % (n, current_ks[n])

                # end objects loop

            time_taken = time.time() - start_time
            if s >= self.burn_in:
                print('\tSAMPLE %d\ttime %4.2f\tnumClusters %d' % ((s+1), time_taken, K))
                self.Z = self._get_Z(self.N, K, current_ks)
                self.ZZ_all += self._get_ZZ(self.Z)
                self.samples_obtained += 1
            else:
                print('\tBURN-IN %d\ttime %4.2f\tnumClusters %d' % ((s+1), time_taken, K))
            sys.stdout.flush()

        # end sample loop
        self.last_K = K
        self.last_assignment = current_ks
        print "DONE!"

    def _C(self, arr):
        sum_arr = np.sum(arr)
        sum_log_gamma = np.sum(gammaln(arr))
        res = sum_log_gamma - gammaln(sum_arr)
        return res

    def _reindex(self, deleted_k, current_ks):
        pos = np.where(current_ks > deleted_k)
        current_ks[pos] = current_ks[pos] - 1
        return current_ks

    def _get_Z(self, N, K, current_ks):
        Z = sp.lil_matrix((N, K))
        for n in range(len(current_ks)):
            k = current_ks[n]
            Z[n, k] = 1
        return Z

    def _get_ZZ(self, Z):
        return Z.tocsr() * Z.tocsr().transpose()

    def __repr__(self):
        return "Gibbs sampling for DP mixture model\n" + self.hyperpars.__repr__() + \
        "\nn_samples = " + str(self.n_samples)