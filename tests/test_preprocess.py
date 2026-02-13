import pytest
import pandas as pd
import numpy as np
from src.preprocess import DataProcessor

@pytest.fixture
def sample_data(tmp_path):
    """
    Create a temporary mock MovieLens dataset for testing.
    
    Args:
        tmp_path: Pytest fixture for temporary directory.
        
    Returns:
        str: Path to the temporary file.
    """
    d = tmp_path / "data"
    d.mkdir()
    file_path = d / "u.data"
    # Mock data: user_id, movie_id, rating, timestamp
    content = "1\t101\t5\t881250949\n2\t102\t3\t881250949"
    file_path.write_text(content)
    return str(file_path)

def test_load_and_clean(sample_data):
    """
    Test if the DataProcessor correctly loads data and creates mappings.
    """
    processor = DataProcessor(sample_data)
    df = processor.load_and_clean()
    
    assert isinstance(df, pd.DataFrame)
    assert 'user_idx' in df.columns
    assert 'movie_idx' in df.columns
    assert len(processor.user_map) == 2