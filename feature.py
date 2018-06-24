import cv2
import glob
import numpy as np
import matplotlib.image as mpimg

from skimage.feature import hog

from joblib import Parallel, delayed

def list_files(data_dir):
    cars = glob.glob('{}/vehicles/*/*.png'.format(data_dir))
    notcars = glob.glob('{}/non-vehicles/*/*.png'.format(data_dir))

    return cars, notcars

def read_image(file):
    image = mpimg.imread(file)
    
    if file.endswith(".png"):
        image = (image*255).astype(np.uint8)
    
    return image

def convert_color_space(image, color_space):
    """
    Assumes that the input image is in RGB color space
    """
    if color_space != 'RGB':
        if color_space == 'HSV':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2HSV)
        elif color_space == 'LUV':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2LUV)
        elif color_space == 'HLS':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2HLS)
        elif color_space == 'YUV':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2YUV)
        elif color_space == 'YCrCb':
            converted = cv2.cvtColor(image, cv2.COLOR_RGB2YCrCb)
    else:
        converted = np.copy(image)

    return converted


class Params:
    """
    color_space: Can be RGB, HSV, LUV, HLS, YUV, YCrCb
    orient: HOG orientations
    pix_per_cell: HOG pixels per cell
    cell_per_block: HOG cells per block
    hog_channel: Can be 0, 1, 2, or "ALL"
    spatial_size: Spatial binning dimensions
    hist_bins: Number of histogram bins
    spatial_feat: Spatial features on or off
    hist_feat: Histogram features on or off
    hog_feat: HOG features on or off
    """
    def __init__(self, color_space='RGB',
                 orient=9, pix_per_cell=8, cell_per_block=2, hog_channel='ALL',
                 spatial_size=(32, 32), hist_bins=32,
                 spatial_feat=True, hist_feat=True, hog_feat=True):
        self.color_space = color_space
        self.orient = orient
        self.pix_per_cell = pix_per_cell
        self.cell_per_block = cell_per_block
        self.hog_channel = hog_channel
        self.spatial_size = spatial_size
        self.hist_bins = hist_bins
        self.spatial_feat = spatial_feat
        self.hist_feat = hist_feat
        self.hog_feat = hog_feat

# Define a function to return HOG features and visualization
def get_hog_features(img, orient, pix_per_cell, cell_per_block, 
                        vis=False, feature_vec=True):
    # Call with two outputs if vis==True
    if vis == True:
        features, hog_image = hog(img, orientations=orient, 
                                  pixels_per_cell=(pix_per_cell, pix_per_cell),
                                  block_norm= 'L2-Hys',
                                  cells_per_block=(cell_per_block, cell_per_block), 
                                  transform_sqrt=True, 
                                  visualise=vis, feature_vector=feature_vec)
        return features, hog_image
    # Otherwise call with one output
    else:      
        features = hog(img, orientations=orient, 
                       pixels_per_cell=(pix_per_cell, pix_per_cell),
                       cells_per_block=(cell_per_block, cell_per_block), 
                       block_norm= 'L2-Hys',
                       transform_sqrt=True, 
                       visualise=vis, feature_vector=feature_vec)
        return features

# Define a function to compute binned color features  
def bin_spatial(img, size=(32, 32)):
    # Use cv2.resize().ravel() to create the feature vector
    features = cv2.resize(img, size).ravel()
    # Return the feature vector
    return features

# Define a function to compute color histogram features 
def color_hist(img, nbins=32, bins_range=(0, 256)):
    # Compute the histogram of the color channels separately
    channel1_hist = np.histogram(img[:,:,0], bins=nbins, range=bins_range)
    channel2_hist = np.histogram(img[:,:,1], bins=nbins, range=bins_range)
    channel3_hist = np.histogram(img[:,:,2], bins=nbins, range=bins_range)
    # Concatenate the histograms into a single feature vector
    hist_features = np.concatenate((channel1_hist[0], channel2_hist[0], channel3_hist[0]))
    # Return the individual histograms, bin_centers and feature vector
    return hist_features

def _extract_features(img_file, params):
    # Create a list to append feature vectors to
    features = []

    # Read in each one by one
    image = read_image(img_file)        
    
    # apply color conversion if other than 'RGB'
    feature_image = convert_color_space(image, params.color_space)    

    if params.spatial_feat == True:
        spatial_features = bin_spatial(feature_image, size=params.spatial_size)
        features.append(spatial_features)
    
    if params.hist_feat == True:
        # Apply color_hist()
        hist_features = color_hist(feature_image, nbins=params.hist_bins)
        features.append(hist_features)
    
    if params.hog_feat == True:
        # Call get_hog_features() with vis=False, feature_vec=True
        if params.hog_channel == 'ALL':
            hog_features = []
            for channel in range(feature_image.shape[2]):
                hog_features.append(get_hog_features(feature_image[:,:,channel], 
                                    params.orient, params.pix_per_cell, params.cell_per_block, 
                                    vis=False, feature_vec=True))
            hog_features = np.ravel(hog_features)        
        else:
            hog_features = get_hog_features(feature_image[:,:,params.hog_channel], params.orient, 
                        params.pix_per_cell, params.cell_per_block, vis=False, feature_vec=True)
        # Append the new feature vector to the features list
        features.append(hog_features)

    features = np.concatenate(features)

    return features

# Define a function to extract features from a list of images
# Have this function call bin_spatial() and color_hist()
def extract_features(img_files, params):   
    features = Parallel(n_jobs=10, verbose=10)(delayed(_extract_features)(file, params) for file in img_files)

    features = np.vstack(features)
    return features