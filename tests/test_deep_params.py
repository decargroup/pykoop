import pykoop


def test_get_set():
    kp = pykoop.KoopmanPipeline(lifting_functions=[
        ('p', pykoop.PolynomialLiftingFn(order=1)),
    ])
    assert kp.get_params()['p__order'] == 1
    kp.set_params(p__order=2)
    assert kp.get_params()['p__order'] == 2


def test_nested_get_set():
    kp = pykoop.KoopmanPipeline(lifting_functions=[
        ('s',
         pykoop.SplitPipeline(
             lifting_functions_state=[
                ('p_state', pykoop.PolynomialLiftingFn(order=1)),
             ],
             lifting_functions_input=[
                ('p_input', pykoop.PolynomialLiftingFn(order=2)),
             ],
         )),
    ])
    assert kp.get_params()['s__p_state__order'] == 1
    kp.set_params(s__p_state__order=3)
    assert kp.get_params()['s__p_state__order'] == 3


def test_invalid_set():
    kp = pykoop.KoopmanPipeline(lifting_functions=[
        ('p', pykoop.PolynomialLiftingFn(order=1)),
    ])
    kp.set_params(p=5)  # Wrong type for ``p``
    assert kp.get_params()['p'] == 5
    kp.set_params(p=pykoop.PolynomialLiftingFn(order=2))
    assert kp.get_params()['p__order'] == 2
