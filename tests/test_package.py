import harness
import trueodds


def test_packages_import():
    assert trueodds.__version__
    assert harness.__doc__
