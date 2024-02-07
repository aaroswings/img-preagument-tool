import unittest

from augment import geometry

class TestPickBucketDims(unittest.TestCase):
    def test_aspect_ratio(self):
        im_width = 650
        im_height = 500
        bucket_dim_options = [(256, 256),(256, 512), (512, 384), (512, 256), (512, 512), (512, 768), (768, 512), (768, 768)]
        

        dims = geometry.pick_bucket_dims(im_width, im_height, bucket_dim_options, True)
        self.assertEqual(dims, (512, 384))

if __name__ == '__main__':
    unittest.main()