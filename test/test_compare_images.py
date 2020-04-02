import pytest

from metcalcpy.compare_images import CompareImages


@pytest.fixture
def settings():
    compare_diff = CompareImages('data/img_1.png', 'data/img_2.png')
    compare_same = CompareImages('data/img_1.png', 'data/img_1.png')
    settings_dict = dict()
    settings_dict['compare_diff'] = compare_diff
    settings_dict['compare_same'] = compare_same
    return settings_dict


def test_get_ssim(settings):
    assert settings['compare_diff'].get_mssim() != 1.0
    assert settings['compare_same'].get_mssim() == 1.0


def test_save_difference_image(settings):
    settings['compare_diff'].save_difference_image('data/img_diff.png')
