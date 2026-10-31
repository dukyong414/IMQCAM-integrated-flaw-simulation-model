import sys
import os
import pytest

# Add project root to sys.path so we can import download.fetch
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from download import fetch

TEST_DOWNLOAD_DIR = 'test/test_downloads'
TEST_MERGED_DIR = 'test/test_merged'


def test_getdata_valid_ids():
    # Should run without exception
    fetch.getdata(['28152209', '26304175'], TEST_DOWNLOAD_DIR, TEST_MERGED_DIR)


def test_downloaddata_invalid_id():
    # downloaddata should not raise but print error, so no exception expected
    # We can just run to check no crash
    fetch.downloaddata("invalid_id_123456", os.path.join(TEST_DOWNLOAD_DIR, "invalid"))


def test_getdata_invalid_itemids_type():
    # Should raise ValueError because itemids is not a list
    with pytest.raises(ValueError):
        fetch.getdata("not_a_list", TEST_DOWNLOAD_DIR)


def test_mergefiles_nonexistent_src():
    # Should not raise but print error and return
    fetch.mergefiles("nonexistent_folder_999", TEST_MERGED_DIR)
