from moss.spec_model import SpecModel
from moss.spec_model_chunked import SpecModelChunked

import numpy as np

import matplotlib.pyplot as plt

hyper_hs = []
tau0 = 0.01
c2 = 4
sig2_alp = 10
N_delta = 30
N_theta = 30
degree_fluctuate = N_delta / 2  # the smaller tends to be smoother
hyper_hs.extend([tau0, c2, sig2_alp, degree_fluctuate])


def _setupModel(mdl, x, hyper_hs, nchunks=None):
    if nchunks is not None:
        model = mdl(x, hyper_hs, nchunks=nchunks, sparse_op=False)
    else:
        model = SpecModel(x, hyper_hs, sparse_op=False)
    # comput fft
    model.sc_fft()
    # compute array of design matrix Z, 3d
    model.Zmtrix()
    # compute X matrix related to basis function on ffreq
    model.Xmtrix(N_delta, N_theta)
    # convert all above to tensorflow object
    model.toTensor()
    # create tranable variables
    model.createModelVariables_hs()
    return model


def test_lnl(Simulation):
    x, freq, spec_true = Simulation

    model = _setupModel(SpecModel, x, hyper_hs)
    val = model.loglik(params=model.trainable_vars)
    assert val.shape == (1,)
    original_val = val[0].numpy()
    assert val[0].numpy() > -2000

    chunked_lnls = []
    # TODO: test nchunk=1
    n_chunks = [1, 2,  4, 8, 16]
    for i in n_chunks:
        model = _setupModel(SpecModelChunked, x, hyper_hs, nchunks=i)
        val = model.loglik(params=model.trainable_vars)
        chunked_lnls.append(val.numpy())

    # assert original_val is close to chunked_lnls[0] by 10
    assert np.abs(original_val - chunked_lnls[0]) < 10

    plt.plot(n_chunks, chunked_lnls, label="chunked")
    # horizontal line at original value
    plt.axhline(original_val, color='red', label="original")
    plt.legend()
    plt.xlabel("Number of chunks")
    plt.ylabel("Log likelihood")
    plt.show()



def test_model_creation(Simulation):
    x, _, _ = Simulation

    model = _setupModel(SpecModel, x, hyper_hs)
    new_model = _setupModel(SpecModelChunked, x, hyper_hs, nchunks=2)

