import os

import numpy as np
from scipy.stats import norm

from numba import njit
from numpy import newaxis


@njit()
def calc_alpha(alpha, matrix, obs_prob, c):
    for t in range(1, alpha.shape[2]):
        alpha[..., t] = alpha[..., t - 1] @ matrix
        alpha[..., t] *= obs_prob[..., t]
        c[:, t] = alpha[..., t].sum(1)
        alpha[..., t] /= np.expand_dims(c[:, t], 0)


@njit()
def calc_beta(beta, matrix, obs_prob, c):
    for t in range(-1, -beta.shape[2], -1):
        beta[..., t - 1] = (beta[..., t] * obs_prob[..., t]) @ matrix
        beta[..., t - 1] /= np.expand_dims(c[:, t - 1], 0)


class RcaHmm:
    """
    An hidden Markov model to determine the country-product martix in
    the context of Economic Fitness model
    """

    def __init__(self, series, n_states):
        self.n_states = n_states
        nonzero = np.log(series[series.nonzero()]).flatten()
        means = np.quantile(
            nonzero,
            np.linspace(0, 1, num=n_states, endpoint=False) + 0.5 / n_states,
        )
        dev = 0.2 * (nonzero.max() - nonzero.min()) / n_states
        self.distr = norm(
            loc=means[:, newaxis], scale=np.full(n_states, dev)[:, newaxis],
        )
        self.matrix = np.full((n_states, n_states), 1 / n_states)
        self.zero_distr = np.logspace(1, n_states, num=n_states)[::-1]
        self.zero_distr /= self.zero_distr.sum()
        self.zero_distr *= 1 - np.count_nonzero(series) / series.size
        self.init_distr = np.full(n_states, 1 / n_states)

    @property
    def distr_params(self):
        return np.column_stack(self.distr.stats(moments="mv"))

    def states(self, series):
        n_series, series_lenght = series.shape
        ser_msk = series != 0
        series = np.piecewise(series, [ser_msk, ~ser_msk], [np.log, -99])
        obs_prob = self.distr.pdf(series[:, newaxis])
        np.copyto(
            obs_prob, self.zero_distr[:, newaxis], where=~ser_msk[:, newaxis],
        )

        alpha = np.empty((n_series, self.n_states, series_lenght))
        beta = np.empty_like(alpha)
        c = np.empty((n_series, series_lenght))

        alpha[..., 0] = self.init_distr * obs_prob[..., 0]
        c[:, 0] = alpha[..., 0].sum(1)
        alpha[..., 0] /= c[:, newaxis, 0]
        for t in range(1, series_lenght):
            np.einsum(
                "ri,ij,rj->rj",
                alpha[..., t - 1],
                self.matrix,
                obs_prob[..., t],
                out=alpha[..., t],
                optimize=True,
            )
            c[:, t] = alpha[..., t].sum(1)
            alpha[..., t] /= c[:, newaxis, t]

        beta[..., -1] = 1 / c[:, newaxis, -1]
        for t in range(1, series_lenght):
            np.einsum(
                "ij,rj,rj->ri",
                self.matrix,
                beta[..., -t],
                obs_prob[..., -t],
                out=beta[..., -t - 1],
                optimize=True,
            )
            beta[..., -t - 1] /= c[:, newaxis, -t - 1]

        gamma = alpha * beta * c[:, newaxis]
        gamma /= gamma.sum(1, keepdims=True)

        return gamma.argmax(1)

    def viterbi(self, series):
        n_series, series_lenght = series.shape
        ser_msk = series != 0
        series = np.piecewise(series, [ser_msk, ~ser_msk], [np.log, -99])
        with np.errstate(divide="ignore"):
            log_obs_prob = np.log(self.distr.pdf(series[:, newaxis]))
        np.copyto(
            log_obs_prob,
            np.log(self.zero_distr)[:, newaxis],
            where=~ser_msk[:, newaxis],
        )

        phi = np.empty((n_series, self.n_states, series_lenght))
        psi = np.empty((n_series, self.n_states, series_lenght))
        phi[..., 0] = np.log10(self.init_distr) + log_obs_prob[..., 0]
        psi[..., 0] = 0
        log_matrix = np.log10(self.matrix)
        for t in range(1, series_lenght):
            phi[..., t] = np.amax(
                phi[..., t - 1, newaxis] + log_matrix, axis=1,
            )
            phi[..., t] += log_obs_prob[..., t]
            psi[..., t] = np.argmax(
                phi[..., t - 1, newaxis] + log_matrix, axis=1,
            )

        states = np.empty((n_series, series_lenght), dtype=int)
        states[:, -1] = np.argmax(phi[..., -1], axis=1)
        for t in range(series_lenght - 1, 0, -1):
            for r in range(n_series):
                states[r, t - 1] = psi[r, states[r, t], t]
        # likelihood = phi[..., -1].max(1)
        return states

    def baum_welch(self, series, eps=5):
        n_series, series_lenght = series.shape

        obs_prob = np.empty((n_series, self.n_states, series_lenght))
        alpha, beta = np.empty_like(obs_prob), np.empty_like(obs_prob)
        c, gamma = np.empty((n_series, series_lenght)), np.empty_like(obs_prob)
        gamma_mix = np.empty((n_series, self.n_states, 2, series_lenght))
        zero_distr = np.empty((self.n_states, 2))

        ser_msk = series == 0
        series = np.piecewise(series, [ser_msk], [-99, np.log])
        mix_hack = np.dstack(
            (~ser_msk[:, newaxis, newaxis], ser_msk[:, newaxis, newaxis],)
        )
        likelihood = []

        matrix_opt, _ = np.einsum_path(
            "rit,ij,rjt,rjt->ij",
            alpha[..., :-1],
            self.matrix,
            beta[..., 1:],
            obs_prob[..., 1:],
            optimize="optimal",
        )

        while True:
            obs_prob[...] = self.distr.pdf(series[:, newaxis])
            np.copyto(
                obs_prob,
                self.zero_distr[:, newaxis],
                where=ser_msk[:, newaxis],
            )

            alpha[..., 0] = self.init_distr * obs_prob[..., 0]
            c[:, 0] = alpha[..., 0].sum(1)
            alpha[..., 0] /= c[:, newaxis, 0]

            # calc_alpha(alpha, self.matrix, obs_prob, c)
            for t in range(1, series_lenght):
                alpha[..., t] = alpha[..., t - 1] @ self.matrix
                alpha[..., t] *= obs_prob[..., t]
                c[:, t] = alpha[..., t].sum(1)
                alpha[..., t] /= c[:, newaxis, t]

            likelihood.append(np.log10(c).sum() / series.size)
            if (
                len(likelihood) == 1
                or np.abs(likelihood[-2] - likelihood[-1]) > 10 ** -eps
            ):
                print(len(likelihood), likelihood[-1], end="\r")
            else:
                order = np.argsort(self.distr.mean().flatten())
                if (order != np.arange(self.n_states)).any():
                    print("Sorted")
                    self.distr = norm(
                        loc=self.distr.mean()[order],
                        scale=self.distr.std()[order],
                    )
                    self.matrix = self.matrix[np.ix_(order, order)]
                    self.zero_distr = self.zero_distr[order]
                    self.init_distr = self.init_distr[order]

                print(
                    f"Traninig completed in {len(likelihood)} iterations",
                    f"Model likelihood : {likelihood[-1]}",
                    sep="\n",
                )
                break

            beta[..., -1] = 1 / c[:, newaxis, -1]

            # calc_beta(beta, self.matrix, obs_prob, c)
            for t in range(1, series_lenght):
                beta[..., -t - 1] = (
                    beta[..., -t] * obs_prob[..., -t]
                ) @ self.matrix.T
                beta[..., -t - 1] /= c[:, newaxis, -t - 1]

            np.einsum(
                "rit,ij,rjt,rjt->ij",
                alpha[..., :-1],
                self.matrix,
                beta[..., 1:],
                obs_prob[..., 1:],
                out=self.matrix,
                optimize=matrix_opt,
            )
            self.matrix /= self.matrix.sum(axis=1, keepdims=True)

            gamma[...] = alpha * beta * c[:, newaxis]
            gamma /= gamma.sum(axis=1, keepdims=True)
            gamma_mix[...] = gamma[..., newaxis, :] * mix_hack

            gamma_mix.sum(axis=(0, 3), out=zero_distr)
            zero_distr /= zero_distr.sum(axis=1, keepdims=True)
            self.zero_distr[:] = zero_distr[:, 1]

            gamma[..., 0].sum(axis=0, out=self.init_distr)
            self.init_distr /= self.init_distr.sum()
            means = (gamma_mix[..., 0, :] * series[:, newaxis]).sum(
                axis=(0, 2)
            )
            means /= gamma_mix[..., 0, :].sum(axis=(0, 2))
            devs = (
                (series[:, newaxis] - means[:, newaxis]) ** 2
                * gamma_mix[..., 0, :]
            ).sum(axis=(0, 2))
            devs /= gamma_mix[..., 0, :].sum((0, 2))
            self.distr = norm(
                loc=means[:, newaxis], scale=np.sqrt(devs)[:, newaxis]
            )


def import_rca_data(data_path):
    first_year, sample_lenght = 1995, 21
    n_countries, n_products = np.genfromtxt(
        f"{data_path}{first_year}/RCAmatrix{first_year}.txt"
    ).shape

    data = np.empty((n_countries, n_products, sample_lenght))
    for i, year in enumerate(range(first_year, first_year + sample_lenght)):
        data[..., i] = np.genfromtxt(f"{data_path}{year}/RCAmatrix{year}.txt")

    with open(
        f"{data_path}{first_year}/countries{first_year}.txt", "r"
    ) as countries_file:
        countries = {
            country.rstrip(): i
            for i, country in enumerate(countries_file.readlines())
        }

    return data, countries


def save_country_model(country, country_model, country_series):
    save_path = f"./results/{country}/"
    if not os.path.exists("./results/"):
        os.mkdir("./results/")
    if not os.path.exists(save_path):
        os.mkdir(save_path)
    np.savetxt(f"{save_path}matrix.txt", country_model.matrix)
    np.savetxt(f"{save_path}means and devs.txt", country_model.distr_params)
    np.savetxt(
        f"{save_path}starting distribution.txt", country_model.init_distr
    )
    np.savetxt(f"{save_path}zeros distribution.txt", country_model.zero_distr)
    np.savetxt(
        f"{save_path}states.txt",
        country_model.states(country_series),
        fmt="%d",
    )

    # for param in ["matrix", "distr_params", "init_distr", "zero_distr"]:
    # np.savetxt(f"{save_path}{param}.txt", country_model[param])


if __name__ == "__main__":
    data, countries = import_rca_data("./data/")

    for country_index, country in enumerate(countries.keys()):
        print(country)
        country_series = data[country_index, ...]
        country_model = RcaHmm(country_series, 4)
        country_model.baum_welch(country_series)

        print(country_model.matrix)
        print(country_model.zero_distr)
        print(country_model.distr_params)

        # print(
        #     country_model[param]
        #     for param in ["matrix", "zero_distr", "distr_params"]
        # )

        # save_country_model(
        #     country, country_model, country_model.states(country_series)
        # )
