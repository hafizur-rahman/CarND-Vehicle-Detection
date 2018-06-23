from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.preprocessing import StandardScaler
from sklearn.utils import shuffle

from feature import *

def prepare_dataset(cars, notcars, params, ratio=0.2):
    car_features = extract_features(cars, params)
    notcar_features = extract_features(notcars, params)

    # Create an array stack of feature vectors
    X = np.vstack((car_features, notcar_features)).astype(np.float64)

    # Define the labels vector
    y = np.hstack((np.ones(len(car_features)), np.zeros(len(notcar_features))))

    # Split up data into randomized training and test sets
    rand_state = np.random.randint(0, 100)

    X, y = shuffle(X, y, random_state=rand_state)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=ratio, random_state=rand_state)
    
    return X_train, X_test, y_train, y_test


def normalize(X_train, X_test):
    # Fit a per-column scaler
    X_scaler = StandardScaler().fit(X_train)

    # Apply the scaler to X
    X_train = X_scaler.transform(X_train)
    X_test = X_scaler.transform(X_test)

    return X_scaler, X_train, X_test


def train_svc_model(X_train, y_train):
    # Use a linear SVC 
    svc = LinearSVC()
    svc.fit(X_train, y_train)

    return svc
