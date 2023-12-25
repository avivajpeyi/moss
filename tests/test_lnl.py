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


def _setupModel(mdl, x, hyper_hs, nchunks=None) -> SpecModel:
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
    print(model)
    original_val = val.numpy()
    assert original_val > -2000

    chunked_lnls = []
    # TODO: test nchunk=1
    n_chunks = [1, 2, 4, 8, 16]
    for i in n_chunks:
        m = _setupModel(SpecModelChunked, x, hyper_hs, nchunks=i)
        val = m.loglik(params=m.trainable_vars)
        print(m)
        chunked_lnls.append(val.numpy())

    # assert original_val is close to chunked_lnls[0] by 10
    assert np.abs(original_val - chunked_lnls[0]) < 10

    plt.plot(n_chunks, chunked_lnls, label="chunked", color="tab:green")
    plt.scatter(n_chunks, chunked_lnls, color="tab:green")
    plt.xlim(left=1)
    # horizontal line at original value
    plt.axhline(original_val, color='red', label="original")
    plt.legend()
    plt.xlabel("Number of chunks")
    plt.ylabel("Log likelihood")
    plt.show()


def test_lnl_chunked_manual(Simulation):
    x, freq, spec_true = Simulation

    model = _setupModel(SpecModel, x, hyper_hs)
    val = model.loglik(params=model.trainable_vars)

    lnls = []
    chunks = [1, 2, 4, 8, 16]
    for i in range(len(chunks)):
        num_of_chunks = chunks[i]
        # reshape data to be split into num_of_chunks
        chunked_data = np.array_split(x, num_of_chunks)
        lnl_for_chunks = []
        for j in range(num_of_chunks):
            assert chunked_data[j].shape == (x.shape[0] // num_of_chunks, x.shape[1])
            m = _setupModel(SpecModel, chunked_data[j], hyper_hs)
            m_lnl = m.loglik(params=model.trainable_vars).numpy()
            lnl_for_chunks.append(m_lnl)
        lnls.append(np.sum(lnl_for_chunks))

    plt.plot(chunks, lnls, label="chunked", color="tab:blue")
    plt.scatter(chunks, lnls, color="tab:blue")
    plt.xlim(left=1)
    # horizontal line at original value
    plt.axhline(val, color='red', label="original")
    plt.legend()
    plt.show()


def test_lnl_chunked_manual_vs_jianan(Simulation):
    x, freq, spec_true = Simulation

    model = _setupModel(SpecModel, x, hyper_hs)
    orig_val = model.loglik(params=model.trainable_vars)

    manual_lnls = []
    jianan_lnls = []


    chunks = [1, 2, 4, 8, 16]
    for i in range(len(chunks)):
        num_of_chunks = chunks[i]

        # KATE VERSION
        manual_lnl = _compute_manual_chunked_lnl(x, hyper_hs, num_of_chunks)
        manual_lnls.append(np.sum(manual_lnl))

        # JIANAN VERSION
        m = _setupModel(SpecModelChunked, x, hyper_hs, nchunks=num_of_chunks)
        val = m.loglik(params=m.trainable_vars)
        jianan_lnls.append(val.numpy())



    plt.plot(chunks, manual_lnls, label="Manual Chunked", color="tab:blue")
    plt.scatter(chunks, manual_lnls, color="tab:blue")

    plt.plot(chunks, jianan_lnls, label="Automatic Chunked", color="tab:green")
    plt.scatter(chunks, jianan_lnls, color="tab:green")

    plt.xlim(left=1)
    # horizontal line at original value
    plt.axhline(orig_val, color='red', label="original")
    plt.legend()
    plt.show()


def _compute_manual_chunked_lnl(x, hyper_hs, num_of_chunks):
    chunked_data = np.array_split(x, num_of_chunks)
    lnl_for_chunks = []
    for j in range(num_of_chunks):
        assert chunked_data[j].shape == (x.shape[0] // num_of_chunks, x.shape[1])
        m = _setupModel(SpecModel, chunked_data[j], hyper_hs)
        m_lnl = m.loglik(params=m.trainable_vars).numpy()
        lnl_for_chunks.append(m_lnl)
    return np.sum(lnl_for_chunks)


def test_model_creation(Simulation):
    x, _, _ = Simulation

    model = _setupModel(SpecModel, x, hyper_hs)
    new_model = _setupModel(SpecModelChunked, x, hyper_hs, nchunks=2)
