from unittest import TestCase
from pkg_resources import resource_filename, get_distribution

from mtbox import mtbox, prep


class TestMtbox(TestCase):        
    def test_read_conf_file(self):
        conf = mtbox.read_conf_file(resource_filename('mtbox.tests', 'conf.json'))
        self.assertEqual(conf['version'], get_distribution('mtbox').version)

    def test_check_version(self):
        with self.assertRaises(ValueError):
            mtbox.check_version(get_distribution('mtbox').version + '1')

    def test_read_input_data(self):
        path_to_input = resource_filename('mtbox.tests', 'input')
        df = prep.read_input_data(path_to_input)
        self.assertEqual(df.shape, (1000, 8))

    def test_extract_indiv_data(self):
        path_to_input = resource_filename('mtbox.tests', 'input')
        df = prep.read_input_data(path_to_input)
        
        target = u'IND_Y1'
        df_indiv = prep.extract_indiv_data(df, target, path_to_input) 
        self.assertEqual(df_indiv.shape, (900, 4))

        target = u'IND_Y2'
        df_indiv = prep.extract_indiv_data(df, target, path_to_input) 
        self.assertEqual(df_indiv.shape, (800, 4))

    def test_encode_categotical_columns(self):
        path_to_input = resource_filename('mtbox.tests', 'input')
        df = prep.read_input_data(path_to_input)

        label_encoders = prep.encode_categotical_columns(df)

        self.assertEqual(label_encoders['CAT_X5'].classes_.tolist(), ['A', 'B', 'C'])
        self.assertEqual(df.dtypes['CAT_X5'].name, 'int64')
