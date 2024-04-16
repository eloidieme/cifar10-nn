import numpy as np
from nnClassifier import logger

def compute_grads_num_slow(self, X, Y, Ws, bs, h=1e-6):
        grads_W = [np.zeros(W.shape) for W in Ws]
        grads_b = [np.zeros(b.shape) for b in bs]

        for layer in range(len(Ws)):
            for i in range(len(bs[layer])):
                b_try = np.array(bs[layer])
                b_try[i] -= h
                bs_try = bs[:]
                bs_try[layer] = b_try
                c1 = self._compute_cost(X, Y, Ws, bs_try)

                b_try[i] += 2*h
                bs_try[layer] = b_try
                c2 = self._compute_cost(X, Y, Ws, bs_try)

                grads_b[layer][i] = (c2 - c1) / (2 * h)

            for i in range(Ws[layer].shape[0]):
                for j in range(Ws[layer].shape[1]):
                    W_try = np.array(Ws[layer])
                    W_try[i, j] -= h
                    Ws_try = Ws[:]
                    Ws_try[layer] = W_try
                    c1 = self._compute_cost(X, Y, Ws_try, bs)

                    W_try[i, j] += 2*h
                    Ws_try[layer] = W_try
                    c2 = self._compute_cost(X, Y, Ws_try, bs)

                    grads_W[layer][i, j] = (c2 - c1) / (2 * h)

        return grads_W, grads_b

def validate_gradient(self, X, Y, h=1e-6, eps=1e-10):
        n_features = X.shape[0]
        reduced_Ws = []
        for W in self.Ws:
            reduced_Ws.append(W[:, :n_features])
        ga_W = self._compute_gradients(X, Y, reduced_Ws, self.bs)[0]
        gn_W = self.compute_grads_num_slow(X, Y, reduced_Ws, self.bs, h)[0]
        rel_err_W = np.zeros((len(ga_W), ga_W[0].shape[0], ga_W[0].shape[1]))
        for k in range(2):
            for i in range(ga_W[k].shape[0]):
                for j in range(ga_W[k].shape[1]):
                    rel_err_W[k, i, j] = (np.abs(ga_W[k][i, j] - gn_W[k][i, j])) / \
                        (max(eps, np.abs(ga_W[k][i, j]) +
                         np.abs(gn_W[k][i, j])))
        ga_b = self._compute_gradients(X, Y, reduced_Ws, self.bs)[1]
        gn_b = self.compute_grads_num_slow(X, Y, reduced_Ws, self.bs, h)[1]
        rel_err_b = np.zeros((len(ga_b), ga_b[0].shape[0]))
        for k in range(2):
            for i in range(ga_b[k].shape[0]):
                rel_err_b[k, i] = (np.abs(ga_b[k][i] - gn_b[k][i])) / \
                    (max(eps, np.abs(ga_b[k][i]) + np.abs(gn_b[k][i])))
        max_diff = max(np.max(rel_err_W), np.max(rel_err_b))
        logger.info("Maximum difference between numerical and analytical gradients: %.2e", max_diff)
        return max_diff