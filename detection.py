from scipy.ndimage.measurements import label
from skimage.filters import gaussian
from feature import *

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, xstart, ystart, ystop, scale, svc, X_scaler, params):
    img_tosearch = img[ystart:ystop, xstart:,:]
        
    # apply color conversion if other than 'RGB'
    ctrans_tosearch = convert_color_space(img_tosearch, params.color_space)
       
    if scale != 1:
        imshape = ctrans_tosearch.shape
        ctrans_tosearch = cv2.resize(ctrans_tosearch, (np.int(imshape[1]/scale), np.int(imshape[0]/scale)))
       
    # Define blocks and steps as above
    nxblocks = (ctrans_tosearch.shape[1] // params.pix_per_cell) - params.cell_per_block + 1
    nyblocks = (ctrans_tosearch.shape[0] // params.pix_per_cell) - params.cell_per_block + 1 
    nfeat_per_block = params.orient*params.cell_per_block**2
    
    # 64 was the orginal sampling rate, with 8 cells and 8 pix per cell
    window = 64
    nblocks_per_window = (window // params.pix_per_cell) - params.cell_per_block + 1
    cells_per_step = 2  # Instead of overlap, define how many cells to step
    nxsteps = (nxblocks - nblocks_per_window) // cells_per_step + 1
    nysteps = (nyblocks - nblocks_per_window) // cells_per_step + 1
    
    if params.hog_feat == True:
        # select colorspace channel for HOG 
        if params.hog_channel == 'ALL':
            ch1 = ctrans_tosearch[:,:,0]
            ch2 = ctrans_tosearch[:,:,1]
            ch3 = ctrans_tosearch[:,:,2]
        else: 
            ch1 = ctrans_tosearch[:,:,params.hog_channel]

        # Compute individual channel HOG features for the entire image
        hog1 = get_hog_features(ch1, params.orient, params.pix_per_cell, params.cell_per_block, feature_vec=False)

        if params.hog_channel == 'ALL':
            hog2 = get_hog_features(ch2, params.orient, params.pix_per_cell, params.cell_per_block, feature_vec=False)
            hog3 = get_hog_features(ch3, params.orient, params.pix_per_cell, params.cell_per_block, feature_vec=False)

    bbox_list=[]
    
    for xb in range(nxsteps):
        for yb in range(nysteps):
            ypos = yb*cells_per_step
            xpos = xb*cells_per_step
            
            spatial_features = []
            hist_features = []
            hog_features = []
            
            if params.hog_feat == True:
                # Extract HOG for this patch
                hog_feat1 = hog1[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 

                if params.hog_channel == 'ALL':
                    hog_feat2 = hog2[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_feat3 = hog3[ypos:ypos+nblocks_per_window, xpos:xpos+nblocks_per_window].ravel() 
                    hog_features = np.hstack((hog_feat1, hog_feat2, hog_feat3))
                else:
                    hog_features = hog_feat1

            xleft = xpos*params.pix_per_cell
            ytop = ypos*params.pix_per_cell

            # Extract the image patch
            subimg = cv2.resize(ctrans_tosearch[ytop:ytop+window, xleft:xleft+window], (64,64))
          
            # Get color features
            if params.spatial_feat == True:
                spatial_features = bin_spatial(subimg, size=params.spatial_size)
                
            if params.hist_feat == True:
                hist_features = color_hist(subimg, nbins=params.hist_bins)

            # Scale features and make a prediction
            test_features = X_scaler.transform(np.hstack((spatial_features, hist_features, hog_features)).reshape(1, -1))    

            test_prediction = svc.predict(test_features)
            
            if test_prediction == 1:
                xbox_left = np.int(xleft*scale)
                ytop_draw = np.int(ytop*scale)
                win_draw = np.int(window*scale)
                
                bbox_left_top = (xstart + xbox_left, ytop_draw+ystart)
                bbox_right_bottom = (xstart + xbox_left+win_draw,ytop_draw+win_draw+ystart)
                
                bbox_list.append((bbox_left_top, bbox_right_bottom))
    return bbox_list

def add_heat(heatmap, bbox_list):
    # Iterate through list of bboxes
    for box in bbox_list:
        # Add += 1 for all pixels inside each bbox
        # Assuming each "box" takes the form ((x1, y1), (x2, y2))
        heatmap[box[0][1]:box[1][1], box[0][0]:box[1][0]] += 1

    # Return updated heatmap
    return heatmap# Iterate through list of bboxes
    
def apply_threshold(heatmap, threshold):
    # Zero out pixels below the threshold
    heatmap[heatmap <= threshold] = 0
    
    heatmap = np.clip(heatmap, 0, 255)

    # Return thresholded map
    return heatmap

def draw_labeled_bboxes(img, labels):
    # Iterate through all detected cars
    for car_number in range(1, labels[1]+1):
        # Find pixels with each car_number label value
        nonzero = (labels[0] == car_number).nonzero()
        
        # Identify x and y values of those pixels
        nonzeroy = np.array(nonzero[0])
        nonzerox = np.array(nonzero[1])
        
        # Define a bounding box based on min/max x and y
        bbox = ((np.min(nonzerox), np.min(nonzeroy)), (np.max(nonzerox), np.max(nonzeroy)))
        
        # Draw the box on the image
        cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img

class CarDetector:
    def __init__(self, params, svc, X_scaler):
        self.params = params
        self.svc = svc
        self.X_scaler = X_scaler

    def detect_cars(self, image):
        new_img = np.copy(image)
        bbox_list = []
        
        multi_window_setting = [
            (400, 1.0),
            (408, 1.0),
            (400, 1.5),
            (420, 1.5),
            (400, 2.0),
            (432, 2.0)
        ]
        
        for ystart, scale in multi_window_setting:
            ystop = ystart + int(64 * scale)        
            
            bbox_list.extend(find_cars(image, 500, ystart, ystop, scale, self.svc, self.X_scaler, self.params))
        
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        heat = add_heat(heat, bbox_list)
            
        heat = gaussian(heat, 3)
        
        heat = apply_threshold(heat, len(multi_window_setting)//2 + 1)

        # Find final boxes from heatmap using label function
        labels = label(heat)
        new_img = draw_labeled_bboxes(new_img, labels)
        
        return new_img