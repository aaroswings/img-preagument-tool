import unittest
from PIL import Image
from augment import geometry
import json 

class TestPickBucketDims(unittest.TestCase):
    def test_aspect_ratio(self):
        im_width = 650
        im_height = 500
        bucket_dim_options = [(256, 256),(256, 512), (512, 384), (512, 256), (512, 512), (512, 768), (768, 512), (768, 768)]
        

        dims = geometry.pick_bucket_dims(im_width, im_height, bucket_dim_options, True)
        self.assertEqual(dims, (512, 384))

    def test_geo_transforms_on_sample(self):
        bucket_dim_options = [(256, 256), (384, 384), (384, 512), (512, 512), (500, 500), (500, 1000)]
        im = Image.open('tests/images/Rainier20200906.jpg').convert('RGB')
        im_bundle = {'sample': im}
        transformed_bundle, transform_params = geometry.geo_transforms(im_bundle, bucket_dim_options)
        transformed_bundle['sample'].save(f'tests/images/out/Rainier20200906.jpg', "JPEG")
        print(json.dumps(transform_params))



if __name__ == '__main__':
    unittest.main()