import unittest

from compute_empirical_complexity import get_data


class MyTestCase(unittest.TestCase):
    def test_one_hot(self):
        classes, max_attr_count, features_count = get_data("./datasets/dummy_data_for_test.csv",
                                                           label_location="last",
                                                           encoding="one-hot",
                                                           columns_to_remove=[0])

        self.assertEqual(max_attr_count, 3, 'incorrect max attribute count')
        self.assertEqual(features_count, 3, 'incorrect feature count')
        self.assertEqual(len(classes['label1']), 1, 'label1: incorrect observations counts')
        self.assertEqual(len(classes['label2']), 3, 'label2: incorrect observations counts')
        self.assertEqual(len(classes['label1'][0]), 9, 'label2: incorrect string length')
        self.assertEqual(len(classes['label2'][0]), 9, 'label2: incorrect string length')

    def test_label(self):
        classes, max_attr_count, features_count = get_data("./datasets/dummy_data_for_test.csv",
                                                           label_location="last",
                                                           encoding="label",
                                                           columns_to_remove=[0])

        self.assertEqual(max_attr_count, 3, 'incorrect max attribute count')
        self.assertEqual(features_count, 3, 'incorrect feature count')
        self.assertEqual(len(classes['label1']), 1, 'label1: incorrect observations counts')
        self.assertEqual(len(classes['label2']), 3, 'label2: incorrect observations counts')
        self.assertEqual(len(classes['label1'][0]), 3, 'label2: incorrect string length')
        self.assertEqual(len(classes['label2'][0]), 3, 'label2: incorrect string length')


if __name__ == '__main__':
    unittest.main()
