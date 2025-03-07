import unittest
import os
from src.data_loader import download_data, load_data_from_csv

class TestDataLoader(unittest.TestCase):
    def test_download_data(self):
        filename = "data/raw/test_sp500.csv"
        download_data(period="1mo", filename=filename)
        self.assertTrue(os.path.exists(filename))
        os.remove(filename)

    def test_load_data_from_csv(self):
        # Create a dummy CSV file for testing
        filename = "data/raw/dummy.csv"
        with open(filename, "w") as f:
            f.write("Date,Open,High,Low,Close,Adj Close,Volume\n")
            f.write("2020-01-01,100,110,90,105,105,1000000\n")
        df = load_data_from_csv(filename)
        self.assertFalse(df.empty)
        os.remove(filename)

if __name__ == "__main__":
    unittest.main()