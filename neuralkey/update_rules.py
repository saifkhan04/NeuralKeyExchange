__author__ = "Saif Khan"

#Update rules for tree parity machine

import numpy as np

class LearningRule:

	def theta(self, t1, t2):
		return 1 if t1 == t2 else 0

	def hebbian(self, W, X, sigma, tau1, tau2, l):
		k, n = W.shape
		for (i, j), _ in np.ndenumerate(W):
			W[i, j] += X[i, j] * tau1 * self.theta(sigma[i], tau1) * self.theta(tau1, tau2)
			W[i, j] = np.clip(W[i, j] , -l, l)

	def anti_hebbian(self, W, X, sigma, tau1, tau2, l):
		k, n = W.shape
		for (i, j), _ in np.ndenumerate(W):
			W[i, j] -= X[i, j] * tau1 * self.theta(sigma[i], tau1) * self.theta(tau1, tau2)
			W[i, j] = np.clip(W[i, j], -l, l)

	def random_walk(self, W, X, sigma, tau1, tau2, l):
		k, n = W.shape
		for (i, j), _ in np.ndenumerate(W):
			W[i, j] += X[i, j] * self.theta(sigma[i], tau1) * self.theta(tau1, tau2)
			W[i, j] = np.clip(W[i, j] , -l, l)
