import numpy as np
import matplotlib.pyplot as plt
import tqdm
import json
import os
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn.functional as F

from nnClassifier import logger

N_CLASSES = 10
N_FEATURES = 3072
N_HIDDEN_NODES = 1500


class Model:
    def __init__(self, X_train, Y_train, gd_params, dropout_rate=None, lamda=0.0, validation=None, cyclical_lr=False, seed=None) -> None:
        if seed:
            np.random.seed(seed)
        self.device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu")
        self.X_train = torch.tensor(X_train, dtype=torch.float).to(self.device)
        self.Y_train = torch.tensor(Y_train, dtype=torch.float).to(self.device)
        self.gd_params = gd_params
        self.dropout_rate = dropout_rate
        self.lamda = lamda
        self.validation = validation
        if validation:
            self.X_val = torch.tensor(
                validation[0], dtype=torch.float).to(self.device)
            self.Y_val = torch.tensor(
                validation[1], dtype=torch.float).to(self.device)
            self.y_val = torch.tensor(
                validation[2], dtype=torch.float).to(self.device)
        self.cyclical_lr = cyclical_lr
        self.Ws, self.bs = self._init_params()

    def _init_params(self, means=[0.0, 0.0], stds=[1/np.sqrt(N_FEATURES), 1/np.sqrt(N_HIDDEN_NODES)]):
        W1 = torch.normal(means[0], stds[0], size=(
            N_HIDDEN_NODES, N_FEATURES), dtype=torch.float)
        b1 = torch.zeros((N_HIDDEN_NODES, 1), dtype=torch.float)
        W2 = torch.normal(means[1], stds[1], size=(
            N_CLASSES, N_HIDDEN_NODES), dtype=torch.float)
        b2 = torch.zeros((N_CLASSES, 1), dtype=torch.float)
        return [W1, W2], [b1, b2]

    def manual_dropout(self, x, dropout_rate=0.5):
        mask = (torch.rand(x.shape) > dropout_rate).float().to(x.device)
        return mask * x / (1.0 - dropout_rate)

    def _evaluate_classifier(self, X, Ws, bs, training):
        s1 = torch.matmul(Ws[0], X) + bs[0]
        H = F.relu(s1)
        if training and self.dropout_rate:
            H = self.manual_dropout(H, self.dropout_rate)
        s = torch.matmul(Ws[1], H) + bs[1]
        P = F.softmax(s, dim=0)

        return [H, P]

    def _compute_gradients(self, X_batch, Y_batch, Ws, bs):
        nb = X_batch.shape[1]
        fact = 1.0 / nb
        H, P = self._evaluate_classifier(X_batch, Ws, bs, True)
        G = -(Y_batch - P)

        grad_W2 = fact * torch.matmul(G, H.T)
        grad_b2 = fact * \
            torch.matmul(G, torch.ones(
                nb, 1, device=self.device, dtype=torch.float))

        G = torch.matmul(Ws[1].T, G)
        G = G * (H > 0).float()

        grad_W1 = fact * torch.matmul(G, X_batch.T)
        grad_b1 = fact * \
            torch.matmul(G, torch.ones(
                nb, 1, device=self.device, dtype=torch.float))

        grad_W1 += 2 * self.lamda * Ws[0]
        grad_W2 += 2 * self.lamda * Ws[1]

        return [grad_W1, grad_W2], [grad_b1, grad_b2]

    def _compute_cost(self, X, Y, Ws, bs):
        P = self._evaluate_classifier(X, Ws, bs, False)[-1]
        fact = 1/X.shape[1]
        log_probs = torch.log(P)
        lcross_sum = -torch.sum(Y * log_probs)
        reg_sum = 0
        for W in Ws:
            reg_sum += torch.sum(torch.flatten(W**2))
        return fact*lcross_sum + self.lamda*reg_sum

    def mini_batch_gd_adam(self, gd_params, grid_search=False):
        train_costs = [self._compute_cost(
            self.X_train, self.Y_train, self.Ws, self.bs)]
        val_costs = [self._compute_cost(
            self.X_val, self.Y_val, self.Ws, self.bs)] if self.validation else None
        n_batch, n_epochs, eta = gd_params["n_batch"], gd_params["n_epochs"], gd_params["eta"]
        beta_1, beta_2, epsilon = gd_params["beta_1"], gd_params["beta_2"], gd_params["epsilon"]
        dataset = TensorDataset(self.X_train.t(), self.Y_train.t())
        dataloader = DataLoader(dataset, batch_size=n_batch, shuffle=True)
        m_Ws = []
        v_Ws = []
        m_bs = []
        v_bs = []
        for idx in range(len(self.Ws)):
            m_Ws.append(torch.zeros_like(self.Ws[idx]))
            v_Ws.append(torch.zeros_like(self.Ws[idx]))
        for idx in range(len(self.bs)):
            m_bs.append(torch.zeros_like(self.bs[idx]))
            v_bs.append(torch.zeros_like(self.bs[idx]))
        t = 1
        for i in range(n_epochs):
            print(f"Epoch {i+1}/{n_epochs}")
            for X_batch, Y_batch in tqdm.tqdm(dataloader, desc="Processing batches"):
                grads = self._compute_gradients(
                    X_batch.t(), Y_batch.t(), self.Ws, self.bs)
                for idx, W_grad in enumerate(grads[0]):
                    m = beta_1*m_Ws[idx] + (1 - beta_1)*W_grad
                    v = beta_2*v_Ws[idx] + (1 - beta_2)*(W_grad**2)
                    m_Ws[idx] = m
                    v_Ws[idx] = v
                    m_hat = m/(1 - beta_1**t)
                    v_hat = v/(1 - beta_2**t)
                    self.Ws[idx] -= (eta/(np.sqrt(v_hat) + epsilon))*m_hat
                for idx, b_grad in enumerate(grads[1]):
                    m = beta_1*m_bs[idx] + (1 - beta_1)*b_grad
                    v = beta_2*v_bs[idx] + (1 - beta_2)*(b_grad**2)
                    m_bs[idx] = m
                    v_bs[idx] = v
                    m_hat = m/(1 - beta_1**t)
                    v_hat = v/(1 - beta_2**t)
                    self.bs[idx] -= (eta/(np.sqrt(v_hat) + epsilon))*m_hat
                t += 1
            current_train_loss = self._compute_cost(
                self.X_train, self.Y_train, self.Ws, self.bs)
            print(f"\t * Train loss: {current_train_loss}")
            train_costs.append(current_train_loss)
            if self.validation:
                current_val_loss = self._compute_cost(
                    self.X_val, self.Y_val, self.Ws, self.bs)
                print(f"\t * Validation loss: {current_val_loss}")
                val_costs.append(current_val_loss)
        return self.Ws, self.bs, train_costs, val_costs

    def compute_accuracy(self, X, y, Ws, bs):
        P = self._evaluate_classifier(X, Ws, bs, False)[-1]
        y_pred = torch.argmax(P, axis=0)
        correct = y_pred[y == y_pred].shape[0]
        return correct / y_pred.shape[0]

    def _grid_search(self, grid, n_epochs=20):
        '''
        grid = {
            "n_batch": [...],
            "eta": [...],
            "lamda": [...]
        }
        '''
        train_losses = {}
        val_losses = {}
        accuracies = {}
        for n_batch in grid["n_batch"]:
            for eta in grid["eta"]:
                for lamda in grid["lamda"]:
                    gd_params = {"n_batch": n_batch,
                                 "n_epochs": n_epochs, "eta": eta}
                    Ws_train, bs_train, train_costs, val_costs = self.mini_batch_gd(
                        gd_params)
                    train_losses[f"{n_batch}_{eta}_{lamda}"] = train_costs
                    val_losses[f"{n_batch}_{eta}_{lamda}"] = val_costs
                    accuracies[f"{n_batch}_{eta}_{lamda}"] = self.compute_accuracy(
                        self.X_val, self.y_val, Ws_train, bs_train)
        return max(accuracies, key=accuracies.get), [train_losses, val_losses, accuracies]

    def run_grid_search(self, grid, n_epochs, output_path):
        grid_res = self._grid_search(grid, n_epochs)
        with open(output_path, 'w+') as f:
            json.dump(grid_res[1], f)
        print(f"Best accuracy: {grid_res[1][-1][grid_res[0]]}")

    def run_training(self, gd_params, figure_savepath=None, test_data=None, model_savepath=None):
        Ws_train, bs_train, train_costs, val_costs = self.mini_batch_gd(
            gd_params)
        logger.info("Training completed.")

        if model_savepath:
            os.makedirs(f"{model_savepath}/Ws", exist_ok=True)
            os.makedirs(f"{model_savepath}/bs", exist_ok=True)
            for idx, W in enumerate(Ws_train):
                torch.save(W, f"{model_savepath}/Ws/W_{idx}")
            for idx, b in enumerate(bs_train):
                torch.save(b, f"{model_savepath}/bs/b_{idx}")

        if test_data:
            (X_test, y_test) = test_data
            X_test = torch.tensor(X_test, dtype=torch.float).to(self.device)
            y_test = torch.tensor(y_test, dtype=torch.float).to(self.device)
            accuracy = self.compute_accuracy(
                X_test, y_test, Ws_train, bs_train)
            print(f"Accuracy on test data: {accuracy}")
            logger.info("Accuracy on test data: %.3f", accuracy)

        plt.plot(np.arange(0, gd_params["n_epochs"] + 1),
                 train_costs, label="training loss")
        plt.plot(np.arange(0, gd_params["n_epochs"] + 1),
                 val_costs, label="validation loss")
        plt.xlabel("epoch")
        plt.ylabel("loss")
        plt.legend()
        if self.cyclical_lr:
            eta = "Cyclical"
        else:
            eta = gd_params["eta"]
        plt.title(
            f"Training curves (n_batch = {gd_params['n_batch']}, n_epochs = {gd_params['n_epochs']}, eta = {eta}, lambda = {self.lamda})")
        plt.grid()
        plt.xlim(0, gd_params["n_epochs"])
        if figure_savepath:
            plt.savefig(figure_savepath, bbox_inches='tight')
        plt.clf()
        logger.info("Figure saved.")
