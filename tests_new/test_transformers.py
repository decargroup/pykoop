"""Test lifting functions."""

class TestLiftingFunctions:

    pass

@pytest.mark.parametrize('lf, X, Xt_exp, n_inputs, episode_feature',
                         episode_indep_test_cases)
def test_lifting_fn_transform(lf, X, Xt_exp, n_inputs, episode_feature):
    lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
    Xt = lf.transform(X)
    np.testing.assert_allclose(Xt, Xt_exp)


@pytest.mark.parametrize('lf, X, Xt_exp, n_inputs, episode_feature',
                         episode_indep_test_cases)
def test_lifting_fn_inverse(lf, X, Xt_exp, n_inputs, episode_feature):
    lf.fit(X, n_inputs=n_inputs, episode_feature=episode_feature)
    Xt = lf.transform(X)
    Xi = lf.inverse_transform(Xt)
    np.testing.assert_allclose(Xi, X)
