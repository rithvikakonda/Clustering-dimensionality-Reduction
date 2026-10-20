import numpy as np
import matplotlib.pyplot as plt


class MyGMM:
    def __init__(self, k=3, max_iter=100, tol=1e-4, random_state=None, reg_covar=1e-6):
        self.k = k
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = np.random.RandomState(random_state)
        self.reg_covar = reg_covar

    def initialize_parameters(self, X):
        n_samples, n_features = X.shape
        self.pi = np.ones(self.k) / self.k
        # KMeans-like random initialization (random samples)
        indices = self.random_state.choice(n_samples, self.k, replace=False)
        self.means = X[indices].astype(float).copy()
        base_cov = np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features)
        self.covariances = np.array([base_cov.copy() for _ in range(self.k)])
        
    def multivariate_gaussian(self, X, mean, cov):
        """Manual implementation of multivariate Gaussian PDF."""
        n_samples, n_features = X.shape
        cov_det = np.linalg.det(cov)
        cov_inv = np.linalg.inv(cov)

        # Numerical stability
        if cov_det <= 0:
            cov_det = 1e-16

        norm_const = 1.0 / (np.power((2 * np.pi), n_features / 2) * np.sqrt(cov_det))
        diff = X - mean
        exp_term = np.exp(-0.5 * np.sum(diff @ cov_inv * diff, axis=1))
        return norm_const * exp_term


    def compute_responsibilities(self, X):
        # compute responsibilities (does not change internal state)
        n_samples = X.shape[0]
        gamma = np.zeros((n_samples, self.k))
        for c in range(self.k):
            pdf_vals = self.multivariate_gaussian(X, self.means[c], self.covariances[c])
            gamma[:, c] = self.pi[c] * pdf_vals
            # rv = multivariate_normal(mean=self.means[c], cov=self.covariances[c], allow_singular=True)
            # gamma[:, c] = self.pi[c] * rv.pdf(X)
        denom = gamma.sum(axis=1, keepdims=True)
        denom[denom == 0] = 1e-16
        gamma = gamma / denom
        return gamma

    def e_step(self, X):
        self.gamma = self.compute_responsibilities(X)

    def m_step(self, X):
        n_samples, n_features = X.shape
        Nk = self.gamma.sum(axis=0)
        # update means
        for c in range(self.k):
            if Nk[c] < 1e-8:
                idx = self.random_state.randint(0, n_samples)
                self.means[c] = X[idx]
            else:
                self.means[c] = (self.gamma[:, c][:, np.newaxis] * X).sum(axis=0) / Nk[c]

        # update covariances
        self.covariances = np.zeros((self.k, n_features, n_features))
        for c in range(self.k):
            if Nk[c] < 1e-8:
                self.covariances[c] = np.cov(X, rowvar=False) + self.reg_covar * np.eye(n_features)
                continue
            diff = X - self.means[c]
            weighted = diff * self.gamma[:, c][:, np.newaxis]
            cov = (weighted.T @ diff) / Nk[c]
            cov += self.reg_covar * np.eye(n_features)
            self.covariances[c] = cov

        # update mixing coefficients
        self.pi = Nk / n_samples
        self.pi[self.pi <= 0] = 1e-12
        self.pi = self.pi / self.pi.sum()

    def compute_log_likelihood(self, X):
        n_samples = X.shape[0]
        likelihood = np.zeros((n_samples, self.k))
        for c in range(self.k):
            pdf_vals = self.multivariate_gaussian(X, self.means[c], self.covariances[c])
            likelihood[:, c] = self.pi[c] * pdf_vals
            # rv = multivariate_normal(mean=self.means[c], cov=self.covariances[c], allow_singular=True)
            # likelihood[:, c] = self.pi[c] * rv.pdf(X)
        total = likelihood.sum(axis=1)
        total[total == 0] = 1e-16
        return np.sum(np.log(total))

    def fit(self, X, callback=None):
        # X: (n_samples, n_features)
        self.initialize_parameters(X)
        self.log_likelihoods = []
        for i in range(self.max_iter):
            self.e_step(X)
            self.m_step(X)
            # compute responsibilities for current parameters (for consistent plotting)
            self.gamma = self.compute_responsibilities(X)
            log_likelihood = self.compute_log_likelihood(X)
            self.log_likelihoods.append(log_likelihood)
            if callback is not None:
                callback(i, self, X)
            if i > 0 and abs(self.log_likelihoods[-1] - self.log_likelihoods[-2]) < self.tol:
                break

    def getMembership(self):
        return self.gamma

    def getLikelihood(self, X):
        return self.compute_log_likelihood(X)
    
    def drawLikelihood(self):
        plt.figure(figsize=(8, 5))
        plt.plot(self.log_likelihoods, marker="o", linestyle="--", color="b")
        plt.xlabel("Iteration")
        plt.ylabel("Log Likelihood")
        plt.title(f"GMM Training - Log Likelihood vs Iteration")
        plt.grid(True)
        plt.text(0.5, 0.5, 'Rithvika',
         transform=plt.gcf().transFigure,
         fontsize=40, color='gray', alpha=0.3,
         ha='center', va='center', rotation=30)
        plt.show()
