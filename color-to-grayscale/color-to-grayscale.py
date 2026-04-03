import numpy as np
def color_to_grayscale(image):
    image = np.array(image) 
    weight = np.array([0.299, 0.587, 0.114])
    return np.dot(image, weight).tolist()
    
    """
    Convert an RGB image to grayscale using luminance weights.
    """
    
    # Write code here