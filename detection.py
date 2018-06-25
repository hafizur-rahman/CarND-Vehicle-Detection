from scipy.ndimage.measurements import label
from skimage.filters import gaussian
from joblib import Parallel,delayed

from feature import *

# Define a single function that can extract features using hog sub-sampling and make predictions
def find_cars(img, xstart, ystart, ystop, scale, svc, X_scaler, params):
    img_tosearch = img[ystart:ystop, xstart:, :]
        
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
                
                bbox_list.append((
                    (xstart + xbox_left, ytop_draw+ystart),
                    (xstart + xbox_left+win_draw,ytop_draw+win_draw+ystart)
                ))
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

# Define a function to draw bounding boxes
def draw_boxes(img, bboxes, color=(0, 0, 255), thick=6):
    # Make a copy of the image
    imcopy = np.copy(img)
    # Iterate through the bounding boxes
    for bbox in bboxes:
        # Draw a rectangle given bbox coordinates
        cv2.rectangle(imcopy, bbox[0], bbox[1], color, thick)
    # Return the image copy with boxes drawn
    return imcopy

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
        w = bbox[1][0] - bbox[0][0]
        h = bbox[1][1] - bbox[0][1]
        area = w * h
        
        draw = False
        if bbox[1][1] >= 550:
            if area > 12000: draw = True
        elif bbox[1][1] >= 450:
            if area > 3000: draw = True

        if draw == True:
            # Draw the box on the image
            cv2.rectangle(img, bbox[0], bbox[1], (0,0,255), 6)
    # Return the image
    return img


class CarDetector:
    def __init__(self, params, svc, X_scaler, scales):
        self.params = params
        self.svc = svc
        self.X_scaler = X_scaler
        self.scales = scales
        self.heatmap_buffer = []
        self.N_buffer = 3
    
    def search_cars(self, image):
        bbox_list = []            

        # Search all scales in parallel
        tmp = Parallel(n_jobs=len(self.scales))(delayed(find_cars)(image, 500, 400, 656,
                                                                           scale, self.svc, self.X_scaler, self.params) for scale in self.scales)

        [bbox_list.extend(bboxes) for bboxes in tmp]
        
        return bbox_list
        
    def generate_heatmap(self, image):        
        heat = np.zeros_like(image[:,:,0]).astype(np.float)
        
        bbox_list = self.search_cars(image)

        # Add heat to each box in box list
        heat = add_heat(heat, bbox_list)

        # Apply threshold to help remove false positives
        heat = apply_threshold(heat, 2)

        # Visualize the heatmap when displaying    
        heatmap = np.clip(heat, 0, 255)
        
        return heatmap       
        
    def find_labels(self, heatmap):
        # Find final boxes from heatmap using label function
        labels = label(heatmap)        
        
        buffer_weights=[0.1,0.2,0.3,0.4]
        
        if len(labels) > 1:
            self.heatmap_buffer.append(heatmap)
            
            if len(self.heatmap_buffer) > self.N_buffer:
                self.heatmap_buffer.pop(0)

            # weight the heatmap based on current frame and previous N frames
            n = len(self.heatmap_buffer)
            for b, w, idx in zip(self.heatmap_buffer, buffer_weights, range(n)):
                self.heatmap_buffer[idx] = b * w

            heatmap = np.sum(np.array(self.heatmap_buffer), axis=0)
            heatmap = apply_threshold( heatmap, threshold= sum(buffer_weights[0:n])*2)

            # Find final boxes from heatmap using label function
            labels = label(heatmap)
        return labels

    def detect_cars(self, image):
        draw_img = np.copy(image)
        heatmap = self.generate_heatmap(image)        
        labels = self.find_labels(heatmap)
        draw_img = draw_labeled_bboxes(draw_img, labels)
        
        return draw_img