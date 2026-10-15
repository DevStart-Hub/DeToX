"""Run with: python tests/test_hdf5_settings.py (numpy, pandas, tables required)."""

from pathlib import Path
import runpy
import tempfile
import unittest

import numpy as np
import pandas as pd
import tables


# Load settings without importing the package's tracker/PsychoPy dependencies.
SETTINGS = runpy.run_path(str(Path(__file__).resolve().parents[1] / 'DeToX' / 'ETSettings.py'))
COLUMN_CLASSES = (SETTINGS['RawDataColumns'], SETTINGS['SimplifiedDataColumns'])


class HDF5SettingsTests(unittest.TestCase):
    def test_schema_matches_declared_columns_and_types(self):
        for columns in COLUMN_CLASSES:
            with self.subTest(format=columns.__name__):
                dtype = columns.get_hdf5_dtype()
                self.assertEqual(dtype.names, tuple(columns.ORDER))

                for name in columns.ORDER:
                    expected = (
                        f"S{SETTINGS['HDF5_EVENT_BYTES']}"
                        if columns.DTYPES[name] == 'string'
                        else columns.DTYPES[name]
                    )
                    self.assertEqual(dtype[name], np.dtype(expected))

    def test_schema_generation_does_not_change_settings(self):
        for columns in COLUMN_CLASSES:
            with self.subTest(format=columns.__name__):
                order = list(columns.ORDER)
                types = dict(columns.DTYPES)
                columns.get_hdf5_dtype()
                self.assertEqual(columns.ORDER, order)
                self.assertEqual(columns.DTYPES, types)
                self.assertEqual(columns.DTYPES['Events'], 'string')

    def test_schema_uses_current_class_definitions(self):
        for columns in COLUMN_CLASSES:
            with self.subTest(format=columns.__name__):
                class CustomColumns(columns):
                    ORDER = ['measurement', 'Events']
                    DTYPES = {'measurement': 'float32', 'Events': 'string'}

                dtype = CustomColumns.get_hdf5_dtype()
                self.assertEqual(dtype.names, ('measurement', 'Events'))
                self.assertEqual(dtype['measurement'], np.dtype('float32'))

    def test_empty_table_append_and_dataframe_read(self):
        with tempfile.TemporaryDirectory(prefix='detox-settings-') as directory:
            for columns in COLUMN_CLASSES:
                with self.subTest(format=columns.__name__):
                    path = Path(directory) / f'{columns.__name__}.h5'
                    dtype = columns.get_hdf5_dtype()

                    with tables.open_file(str(path), mode='w') as file:
                        table = file.create_table('/', 'gaze', description=dtype)
                        self.assertEqual(table.nrows, 0)

                    empty = pd.read_hdf(path, key='gaze')
                    self.assertEqual(empty.shape, (0, len(columns.ORDER)))
                    self.assertEqual(list(empty.columns), columns.ORDER)

                    rows = np.zeros(2, dtype=dtype)
                    rows['Events'] = [b'trial_start', b'trial_end']
                    for row in rows:
                        with tables.open_file(str(path), mode='r+') as file:
                            file.root.gaze.append(np.array([row], dtype=dtype))

                    saved = pd.read_hdf(path, key='gaze')
                    self.assertEqual(len(saved), 2)
                    self.assertEqual(list(saved.columns), columns.ORDER)
                    self.assertEqual(saved['Events'].tolist(), ['trial_start', 'trial_end'])


if __name__ == '__main__':
    unittest.main(verbosity=2)
