"""Extract features"""
#! /usr/bin/env python
import numpy as np
from skimage import io, color

def component_histograms(image):
    """ Computes histograms of each component of each pixel
    Parameters:
    - image : A 3-dimensional array
    Returns:
    List of histograms with 256 bins for each component of last dimension
    """
    shape = image.shape
    assert len(shape) == 3
    _, _, num_comp = shape
    return [np.histogram(image[:, :, [idx]].flatten(), 256)
            for idx in xrange(0, num_comp)]

def main():
    """Main"""
    filename = "../sample/sample/10_left.jpeg"
    image = io.imread(filename)
    rgb_hists = component_histograms(hsv)
    
    gray = color.rgb2gray(image)
    gray_hist = np.historgram(gray.flatten(), 256)
    
    hsv = color.rgb2hsv(image)
    hsv_hists = component_histograms(hsv)

    print hist.shape

if __name__ == "__main__":
    main()
