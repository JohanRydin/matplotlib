import pytest
from numpy.testing import assert_allclose, assert_array_equal
import numpy as np
import logging

from matplotlib.sankey import Sankey
from matplotlib.testing.decorators import check_figures_equal


def test_sankey():
    # lets just create a sankey instance and check the code runs
    sankey = Sankey()
    sankey.add()


def test_label():
    s = Sankey(flows=[0.25], labels=['First'], orientations=[-1])
    assert s.diagrams[0].texts[0].get_text() == 'First\n0.25'


def test_format_using_callable():
    # test using callable by slightly incrementing above label example

    def show_three_decimal_places(value):
        return f'{value:.3f}'

    s = Sankey(flows=[0.25], labels=['First'], orientations=[-1],
               format=show_three_decimal_places)

    assert s.diagrams[0].texts[0].get_text() == 'First\n0.250'


@pytest.mark.parametrize('kwargs, msg', (
    ({'gap': -1}, "'gap' is negative"),
    ({'gap': 1, 'radius': 2}, "'radius' is greater than 'gap'"),
    ({'head_angle': -1}, "'head_angle' is negative"),
    ({'tolerance': -1}, "'tolerance' is negative"),
    ({'flows': [1, -1], 'orientations': [-1, 0, 1]},
     r"The shapes of 'flows' \(2,\) and 'orientations'"),
    ({'flows': [1, -1], 'labels': ['a', 'b', 'c']},
     r"The shapes of 'flows' \(2,\) and 'labels'"),
    ))
def test_sankey_errors(kwargs, msg):
    with pytest.raises(ValueError, match=msg):
        Sankey(**kwargs)


@pytest.mark.parametrize('kwargs, msg', (
    ({'trunklength': -1}, "'trunklength' is negative"),
    ({'flows': [0.2, 0.3], 'prior': 0}, "The scaled sum of the connected"),
    ({'prior': -1}, "The index of the prior diagram is negative"),
    ({'prior': 1}, "The index of the prior diagram is 1"),
    ({'connect': (-1, 1), 'prior': 0}, "At least one of the connection"),
    ({'connect': (2, 1), 'prior': 0}, "The connection index to the source"),
    ({'connect': (1, 3), 'prior': 0}, "The connection index to this dia"),
    ({'connect': (1, 1), 'prior': 0, 'flows': [-0.2, 0.2],
      'orientations': [2]}, "The value of orientations"),
    ({'connect': (1, 1), 'prior': 0, 'flows': [-0.2, 0.2],
      'pathlengths': [2]}, "The lengths of 'flows'"),
    ))
def test_sankey_add_errors(kwargs, msg):
    sankey = Sankey()
    with pytest.raises(ValueError, match=msg):
        sankey.add(flows=[0.2, -0.2])
        sankey.add(**kwargs)


def test_sankey2():
    s = Sankey(flows=[0.25, -0.25, 0.5, -0.5], labels=['Foo'],
               orientations=[-1], unit='Bar')
    sf = s.finish()
    assert_array_equal(sf[0].flows, [0.25, -0.25, 0.5, -0.5])
    assert sf[0].angles == [1, 3, 1, 3]
    assert all([text.get_text()[0:3] == 'Foo' for text in sf[0].texts])
    assert all([text.get_text()[-3:] == 'Bar' for text in sf[0].texts])
    assert sf[0].text.get_text() == ''
    assert_allclose(sf[0].tips,
                    [(-1.375, -0.52011255),
                     (1.375, -0.75506044),
                     (-0.75, -0.41522509),
                     (0.75, -0.8599479)])

    s = Sankey(flows=[0.25, -0.25, 0, 0.5, -0.5], labels=['Foo'],
               orientations=[-1], unit='Bar')
    sf = s.finish()
    assert_array_equal(sf[0].flows, [0.25, -0.25, 0, 0.5, -0.5])
    assert sf[0].angles == [1, 3, None, 1, 3]
    assert_allclose(sf[0].tips,
                    [(-1.375, -0.52011255),
                     (1.375, -0.75506044),
                     (0, 0),
                     (-0.75, -0.41522509),
                     (0.75, -0.8599479)])


@check_figures_equal(extensions=['png'])
def test_sankey3(fig_test, fig_ref):
    ax_test = fig_test.gca()
    s_test = Sankey(ax=ax_test, flows=[0.25, -0.25, -0.25, 0.25, 0.5, -0.5],
                    orientations=[1, -1, 1, -1, 0, 0])
    s_test.finish()

    ax_ref = fig_ref.gca()
    s_ref = Sankey(ax=ax_ref)
    s_ref.add(flows=[0.25, -0.25, -0.25, 0.25, 0.5, -0.5],
              orientations=[1, -1, 1, -1, 0, 0])
    s_ref.finish()


def test__preprocess_flows():
    sankey = Sankey()
    flows = [100.0, 200.0, -50.0, -250.0]
    empty_array = []
    assert_array_equal(sankey._preprocess_flows(None), np.array([1.0, -1.0])),
    "Failed for None input"
    assert_array_equal(sankey._preprocess_flows(np.array(flows)), np.array(flows)),
    "Failed for list input"
    assert_array_equal(sankey._preprocess_flows(np.array(empty_array)),
                       np.array(empty_array)),
    "Failed for empty list input"


def test__preprocess_rotation():
    sankey = Sankey()
    assert sankey._preprocess_rotation(None) == 0
    assert sankey._preprocess_rotation(90) == 1
    assert sankey._preprocess_rotation(270) == 3


def test__preprocess_orientations():
    sankey = Sankey()
    assert_array_equal(sankey._preprocess_orientations(
        None, 3, np.broadcast_to(0, 3)), np.broadcast_to(0, 3))
    orientations = [1, 2]
    flows = np.array([3, 4])
    with pytest.raises(ValueError):
        sankey._preprocess_orientations(orientations, 1, flows)


# TODO: only indirectly calls _preprocess_labels
# since it is quite embedded in other functions/codesnippet
def test__preprocess_labels():
    sankey = Sankey()

    # Test with None labels
    flows = np.array([1.0, -1.0, 2.0])
    result = sankey.add(flows=flows, labels=None, trunklength=1.0)
    assert isinstance(result, Sankey)

    # Case 2: Test add() with a valid list of labels
    labels = ['Input', 'Output', 'Min']
    result = sankey.add(flows=flows, labels=labels, trunklength=1.0)
    assert isinstance(result, Sankey)

    # Case 3: Test with less labels than flows
    incompatible_labels = ['Input', 'Output']  # Only 2 labels for 3 flows
    with pytest.raises(ValueError):
        sankey.add(flows=flows, labels=incompatible_labels, trunklength=1.0)

    # Case 4: Test with more labels than flows
    more_labels = ['Input', 'Output', 'Banana', 'Apple']
    with pytest.raises(ValueError):
        result = sankey.add(flows=flows, labels=more_labels, trunklength=1.0)


def test__check_trunklength():
    s = Sankey(flows=[0.25], labels=['First'], orientations=[-1])
    s._check_trunklength(5)
    s._check_trunklength(0)
    with pytest.raises(ValueError,
                       match="trunklength' is negative, which is not allowed because it"
                       " would cause poor layout"):
        s._check_trunklength(-3)


# TODO: Does the caplog really reset between the tests?
def test__check_flows(caplog):
    sankey = Sankey(tolerance=3)

    # Test for zero flow sum
    flows = [100.0, -100.0]
    patchlabel = "Example Patchlabel"

    with caplog.at_level(logging.INFO):
        sankey._check_flows(flows, patchlabel)

    assert "The sum of the flows is nonzero" not in caplog.text

    # Test for flow with empty array
    flows = []

    with caplog.at_level(logging.INFO):
        sankey._check_flows(flows, patchlabel)

    assert "The sum of the flows is nonzero" not in caplog.text

    # Test for non-zero flow sum
    flows = [100.0, -50.0]

    with caplog.at_level(logging.INFO):
        sankey._check_flows(flows, patchlabel)

    assert "The sum of the flows is nonzero" in caplog.text

    # Test for flow sum equal to tolerance
    flows = [5.02, -2.02]

    with caplog.at_level(logging.INFO):
        sankey._check_flows(flows, patchlabel)

    assert "The sum of the flows is nonzero" in caplog.text
