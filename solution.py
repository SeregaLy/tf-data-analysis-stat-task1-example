import numpy
import pandas as pd
import numpy as np
import scipy.stats as st

chat_id = 876290128


def solution(x: np.ndarray) -> float:
    v_mean = np.mean(x)
    t = 10
    n = len(x)
    t_alpha2 = st.t.ppf(1 - 0.05 / 2, df=n - 1)
    s = np.std(x, ddof=1)
    delta = t_alpha2 * s / np.sqrt(n)
    a = (v_mean - 0) / t * (1 - st.t.cdf(delta, df=n - 1))
    return a + delta


