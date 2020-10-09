# %%
import holoviews as hv
import numpy as np
from holoviews import opts
from scipy.stats import norm

from guess import RcaHmm, import_rca_data, save_country_model

hv.extension("bokeh", "matplotlib")
opts.defaults(opts.Distribution(width=650), opts.Curve(width=650))

a, countries = import_rca_data("./mcps export fixed size/")

# %%
country = "Canada"
country_series = a[countries[country], ...]
n_nonzero = 1 - np.count_nonzero(country_series) / np.prod(
    country_series.shape)
print(f"Null values: {100 * n_nonzero:.2f}%")

country_model = RcaHmm(country_series, 4)
country_model.baum_welch(country_series, eps=8)

print(
    "Matrix:",
    country_model.matrix,
    "Means and std devs:",
    country_model.distr_params,
    "Zero distribution:",
    country_model.zero_distr,
    "Starting distribution:",
    country_model.init_distr,
    sep="\n",
)

hv.Curve(country_model.lk, "iterations", "likelihood")
# %%
country_states = country_model.states(country_series)

support = np.linspace(
    np.log(country_series[country_series.nonzero()].min()),
    np.log(country_series.max()),
    250,
)

occ = (np.count_nonzero(
    np.bitwise_and(
        country_states == np.arange(4)[:, np.newaxis, np.newaxis],
        country_series != 0),
    axis=1) / country_series.shape[0])  # yapf: disable
occ /= occ.sum(0)

dists = norm.pdf(support[:, np.newaxis],
                 loc=country_model.means,
                 scale=country_model.devs)[..., np.newaxis] * occ

agg_plot = hv.HoloMap(
    {
        t + 1995: hv.Distribution(
            np.log(country_series[country_series[:, t].nonzero()[0], t])) *
        hv.Overlay([hv.Curve((support, dists[:, i, t])) for i in range(4)])
        for t in range(country_series.shape[1])
    },
    "year",
).redim(Density="density", Value="log(RCA)")

agg_plot

# %%
save_country_model(country, country_model, country_states)
