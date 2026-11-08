"""Synthetic verification of original-row alignment across CSV chunk boundaries."""
import unittest
from unittest.mock import patch
import numpy as np
import pandas as pd
from src.data_loader import CHANNELS, TARGETS
from src.holdout_evaluation import load_raw_holdout


class HoldoutAlignmentTests(unittest.TestCase):
    def test_unsorted_positions_restored_across_chunks(self):
        values=np.arange(6*230,dtype=np.float32).reshape(6,230)
        frame=pd.DataFrame(values,columns=TARGETS+CHANNELS)
        chunks=[frame.iloc[:2],frame.iloc[2:4],frame.iloc[4:]]
        with patch('src.holdout_evaluation._load_labeled_chunks',return_value=iter(chunks)):
            x,y=load_raw_holdout(np.array([5,0,3]))
        np.testing.assert_array_equal(x,values[[5,0,3],4:])
        np.testing.assert_array_equal(y,values[[5,0,3],:4])

    def test_missing_original_row_rejected(self):
        frame=pd.DataFrame(np.ones((2,230),dtype=np.float32),columns=TARGETS+CHANNELS)
        with patch('src.holdout_evaluation._load_labeled_chunks',return_value=iter([frame])):
            with self.assertRaises(ValueError):
                load_raw_holdout(np.array([3]))


if __name__=='__main__':
    unittest.main()
