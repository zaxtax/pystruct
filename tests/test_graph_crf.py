import numpy as np
from numpy.testing import assert_array_equal, assert_array_almost_equal
##from nose.tools import assert_equal, assert_almost_equal, assert_raises
from nose.tools import assert_almost_equal, assert_equal

from pystruct.models import GraphCRF, EdgeTypeGraphCRF

w = np.array([1, 0,  # unary
              0, 1,
              .22,  # pairwise
              0, .22])

# for directional CRF with non-symmetric weights
w_sym = np.array([1, 0,    # unary
                  0, 1,
                  .22, 0,  # pairwise
                  0, .22])

# triangle
x_1 = np.array([[0, 1], [1, 0], [.4, .6]])
g_1 = np.array([[0, 1], [1, 2], [0, 2]])
# expected result
y_1 = np.array([1, 0, 1])

# chain
x_2 = np.array([[0, 1], [1, 0], [.4, .6]])
g_2 = np.array([[0, 1], [1, 2]])
# expected result
y_2 = np.array([1, 0, 0])


def test_graph_crf_inference():
    # create two samples with different graphs
    # two states only, pairwise smoothing
    for inference_method in ['qpbo', 'lp', 'ad3', 'dai', 'ogm']:
        crf = GraphCRF(n_states=2, inference_method=inference_method)
        assert_array_equal(crf.inference((x_1, g_1), w), y_1)
        assert_array_equal(crf.inference((x_2, g_2), w), y_2)


def test_edge_type_graph_crf():
    # create two samples with different graphs
    # two states only, pairwise smoothing

    # all edges are of the first type. should do the same as GraphCRF
    # if we make w symmetric
    for inference_method in ['qpbo', 'lp', 'ad3', 'dai', 'ogm']:
        crf = EdgeTypeGraphCRF(n_states=2, inference_method=inference_method,
                               n_edge_types=1)
        assert_array_equal(crf.inference((x_1, [g_1]), w_sym), y_1)
        assert_array_equal(crf.inference((x_2, [g_2]), w_sym), y_2)

    # same, only with two edge types and no edges of second type
    w_sym_ = np.array([1, 0,    # unary
                      0, 1,
                      .22, 0,  # pairwise
                      0, .22,
                      2, -1,   # second edge type, doesn't exist
                      -1, 3])
    for inference_method in ['qpbo', 'lp', 'ad3', 'dai', 'ogm']:
        crf = EdgeTypeGraphCRF(n_states=2, inference_method=inference_method,
                               n_edge_types=2)
        assert_array_equal(crf.inference((x_1, [g_1, np.zeros((0, 2),
                                                              dtype=np.int)]),
                                         w_sym_), y_1)
        assert_array_equal(crf.inference((x_2, [g_2, np.zeros((0, 2),
                                                              dtype=np.int)]),
                                         w_sym_), y_2)


def test_graph_crf_continuous_inference():
    for inference_method in ['lp', 'ad3']:
        crf = GraphCRF(n_states=2, inference_method=inference_method)
        assert_array_equal(np.argmax(crf.inference((x_1, g_1), w,
                                                   relaxed=True)[0], axis=-1),
                           y_1)
        assert_array_equal(np.argmax(crf.inference((x_2, g_2), w,
                                                   relaxed=True)[0], axis=-1),
                           y_2)


def test_graph_crf_energy_lp_integral():
    crf = GraphCRF(n_states=2, inference_method='lp')
    inf_res, energy_lp = crf.inference((x_1, g_1), w, relaxed=True,
                                       return_energy=True)
    # integral solution
    assert_array_almost_equal(np.max(inf_res[0], axis=-1), 1)
    y = np.argmax(inf_res[0], axis=-1)
    # energy and psi check out
    assert_almost_equal(energy_lp, -np.dot(w, crf.psi((x_1, g_1), y)))


def test_graph_crf_energy_lp_relaxed():
    crf = GraphCRF(n_states=2, inference_method='lp')
    for i in xrange(10):
        w_ = np.random.uniform(size=w.shape)
        inf_res, energy_lp = crf.inference((x_1, g_1), w_, relaxed=True,
                                           return_energy=True)
        assert_almost_equal(energy_lp,
                            -np.dot(w_, crf.psi((x_1, g_1), inf_res)))

    # now with fractional solution
    x = np.array([[0, 0], [0, 0], [0, 0]])
    inf_res, energy_lp = crf.inference((x, g_1), w, relaxed=True,
                                       return_energy=True)
    assert_almost_equal(energy_lp, -np.dot(w, crf.psi((x, g_1), inf_res)))


def test_graph_crf_loss_augment():
    x = (x_1, g_1)
    y = y_1
    crf = GraphCRF(n_states=2, inference_method='lp')
    y_hat, energy = crf.loss_augmented_inference(x, y, w, return_energy=True)
    # check that y_hat fulfills energy + loss condition
    assert_almost_equal(np.dot(w, crf.psi(x, y_hat)) + crf.loss(y, y_hat),
                        -energy)


def test_edge_type_graph_crf_energy_lp_integral():
    # same test as for graph crf above, using single edge type
    crf = EdgeTypeGraphCRF(n_states=2, inference_method='lp', n_edge_types=1)
    inf_res, energy_lp = crf.inference((x_1, [g_1]), w_sym, relaxed=True,
                                       return_energy=True)
    # integral solution
    assert_array_almost_equal(np.max(inf_res[0], axis=-1), 1)
    y = np.argmax(inf_res[0], axis=-1)
    # energy and psi check out
    assert_almost_equal(energy_lp, -np.dot(w_sym, crf.psi((x_1, [g_1]), y)))


def test_edge_type_graph_crf_energy_lp_relaxed():
    # same test as for graph crf above, using single edge type
    crf = EdgeTypeGraphCRF(n_states=2, inference_method='lp',
                           n_edge_types=1)
    for i in xrange(10):
        w_ = np.random.uniform(size=w_sym.shape)
        inf_res, energy_lp = crf.inference((x_1, [g_1]), w_, relaxed=True,
                                           return_energy=True)
        assert_almost_equal(energy_lp,
                            -np.dot(w_, crf.psi((x_1, [g_1]), inf_res)))

    # now with fractional solution
    x = np.array([[0, 0], [0, 0], [0, 0]])
    inf_res, energy_lp = crf.inference((x, [g_1]), w_sym, relaxed=True,
                                       return_energy=True)
    assert_almost_equal(energy_lp,
                        -np.dot(w_sym, crf.psi((x, [g_1]), inf_res)))


def test_graph_crf_class_weights():
    # no edges
    crf = GraphCRF(n_states=3, n_features=3, inference_method='dai')
    w = np.array([1, 0, 0,  # unary
                  0, 1, 0,
                  0, 0, 1,
                  0,        # pairwise
                  0, 0,
                  0, 0, 0])
    x = (np.array([[1, 1.5, 1.1]]), np.empty((0, 2)))
    assert_equal(crf.inference(x, w), 1)
    # loss augmented inference picks last
    assert_equal(crf.loss_augmented_inference(x, [1], w), 2)

    # with class-weights, loss for class 1 is smaller, loss-augmented inference
    # will find it
    crf = GraphCRF(n_states=3, n_features=3, inference_method='dai',
                   class_weight=[1, .1, 1])
    assert_equal(crf.loss_augmented_inference(x, [1], w), 1)

    # except if we do C rescaling (I think)
    crf = GraphCRF(n_states=3, n_features=3, inference_method='dai',
                   class_weight=[1, .1, 1], rescale_C=True)
    #assert_equal(crf.loss_augmented_inference(x, [1], w), 1)
    # smoketest only :-(
    crf.loss_augmented_inference(x, [1], w),


def test_class_weights_rescale_C_psi_inference():
    # check consistency of crammer-singer svm and crf if rescale_C=True
    from sklearn.datasets import make_blobs
    from pystruct.problems import CrammerSingerSVMProblem
    X, Y = make_blobs(n_samples=210, centers=4, random_state=1, cluster_std=3,
                      shuffle=False)
    X = np.hstack([X, np.ones((X.shape[0], 1))])
    X, Y = X[:170], Y[:170]

    weights = 1. / np.bincount(Y)
    weights *= len(weights) / np.sum(weights)

    X_graphs = [(x[np.newaxis, :], np.empty((0, 2), dtype=np.int)) for x in X]

    pbl = GraphCRF(n_features=3, n_states=4, class_weight=weights,
                   rescale_C=True, inference_method='dai')

    pbl_cs = CrammerSingerSVMProblem(n_features=3, n_classes=4,
                                     class_weight=weights, rescale_C=True)

    for x, y in zip(X_graphs, Y[:, np.newaxis]):
        assert_array_almost_equal(pbl.psi(x, y, y_true=y)[:12],
                                  pbl_cs.psi(x[0].ravel(), y[0], y_true=y[0]))
        y_true = np.random.randint(3)
        assert_array_almost_equal(pbl.psi(x, y, y_true=[y_true])[:12],
                                  pbl_cs.psi(x[0].ravel(), y[0],
                                             y_true=y_true))
        w = np.random.normal(size=pbl.size_psi)
        assert_array_equal(pbl.inference(x, w),
                           [pbl_cs.inference(x[0].ravel(), w[:12])])

        assert_array_equal(pbl.loss_augmented_inference(x, y, w),
                           [pbl_cs.loss_augmented_inference(
                               x[0].ravel(), y[0], w[:12])])

        y_hat, energy = pbl.loss_augmented_inference(x, y, w,
                                                     return_energy=True)
        assert_almost_equal(energy,
                            pbl.loss(y, y_hat)
                            + np.dot(w, pbl.psi(x, y_hat, y_true=y)))

        pot = pbl.get_unary_potentials(x, w, y_true=y)
        for i in xrange(4):
            assert_almost_equal(pot[0, i], np.dot(w, pbl.psi(x, np.array([i]),
                                                             y_true=y)))


def test_class_weights_rescale_C_edges():
    # check consistency of psi and inference when edges are present
    crf = GraphCRF(n_states=2, inference_method='dai', class_weight=[.3, .8],
                   rescale_C=True)
    for i in xrange(10):
        if i == 0:
            w_ = w
        else:
            w_ = np.random.normal(size=crf.size_psi)

    y, energy = crf.loss_augmented_inference((x_1, g_1), y_1, w_,
                                             return_energy=True)
    # energy and psi check out
    assert_almost_equal(energy, crf.loss(y_1, y)
                        + np.dot(w_, crf.psi((x_1, g_1), y, y_true=y_1)))
