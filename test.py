# %%
import cProfile

import holoviews as hv
import numpy as np
from holoviews import opts

from guess import RcaHmm

hv.extension("bokeh", "matplotlib")
opts.defaults(opts.Distribution(width=650))

# %%
series = np.genfromtxt("gen_param/values.txt")
gen_states = np.genfromtxt("gen_param/states.txt")

model = RcaHmm(series, 4)
hv.Distribution(np.log(series[series.nonzero()]))

# %%
model.baum_welch(series, 7)
hv.Curve(model.lk, "iterations", "likelihood")

# %%
right_states = 100 * np.count_nonzero(
    gen_states == model.viterbi(series)) / series.size
right_viterbi = 100 * np.count_nonzero(
    gen_states == model.states(series)) / series.size
print(f"Right states with Viterbi:      {right_states:.2f}%\n"
      f"Right states with gamma argmax: {right_viterbi:.2f}%")

mt = hv.Scatter((model.matrix.flat, np.genfromtxt("gen_param/A.txt").flat))
avg = hv.Scatter(
    (model.distr.mean, np.log(np.genfromtxt("gen_param/b.txt")[:, 0])))
std = hv.Scatter((model.distr.std, np.genfromtxt("gen_param/b.txt")[:, 1]))
pi = hv.Scatter((model.init_distr, np.genfromtxt("gen_param/pi.txt")))
ze = hv.Scatter(
    (model.zero_distr, np.genfromtxt("gen_param/zero_distr.txt")[:, 0]))

param_max = (
    np.array([model.matrix.max(), model.init_distr.max(), model.zero_distr.max()]).max()
    + 0.02
)  # yapf: disable
fit = hv.Curve((*[[0, param_max]] * 2, ))

layout = (mt * std * pi * ze * fit).redim(x="Inferred", y="Simulated")
layout.opts(opts.Scatter(size=7, tools=["hover"]), opts.Curve(color="green"))

# %%
cProfile.run("model.baum_welch(series, 5)", sort="cumtime")

# %%
