import numpy as np
from scipy.stats import lognorm, norm
import os

n_states, max_jump = 4, 2
series_lenght, n_series = 20, 3000

A = np.random.ranf((n_states, n_states))
A /= A.sum(1, keepdims=True)

means = np.exp(
    np.linspace(1, n_states + 1, num=n_states) +
    norm.rvs(scale=0.3, size=n_states))
devs = norm.rvs(loc=0.3, scale=0.2, size=n_states)

init_distr = np.random.ranf(n_states)
init_distr /= init_distr.sum()

zero_distr = np.random.ranf((n_states, 2))
zero_distr /= zero_distr.sum(1, keepdims=True)
sw = zero_distr.sum(0).argmax()
zero_distr = zero_distr[np.ix_(
    np.argsort(zero_distr, axis=0)[:, sw], [1 - sw, sw])]

states = np.empty((n_series, series_lenght), dtype=int)
states[:, 0] = np.random.choice(n_states, n_series, p=init_distr)
for t in np.arange(1, series_lenght):
    for r in np.arange(n_series):
        states[r, t] = np.random.choice(n_states, p=A[states[r, t - 1], :])

values = np.empty((n_series, series_lenght))
for r, t in np.ndindex(values.shape):
    if np.random.choice(2, p=zero_distr[states[r, t], :]):
        values[r, t] = lognorm.rvs(devs[states[r, t]],
                                   scale=means[states[r, t]])
    else:
        values[r, t] = 0

if not os.path.exists('./gen_param'):
    os.mkdir('gen_param')
np.savetxt('./gen_param/A.txt', A)
np.savetxt('./gen_param/b.txt', np.column_stack((means, devs)))
np.savetxt('./gen_param/pi.txt', init_distr)
np.savetxt('./gen_param/zero_distr.txt', zero_distr)
np.savetxt('./gen_param/states.txt', states, fmt='%d')
np.savetxt('./gen_param/values.txt', values, fmt='%.5e')
