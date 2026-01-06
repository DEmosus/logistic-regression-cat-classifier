import numpy as np
import h5py

def load_dataset():
    # Load the training dataset from HDF5 file
    train_dataset = h5py.File('datasets/train_catvnoncat.h5', "r")
    
    # Extract training set features (images) as a NumPy array
    train_set_x_orig = np.array(train_dataset["train_set_x"][:])  
    
    # Extract training set labels (0 = non-cat, 1 = cat) as a NumPy array
    train_set_y_orig = np.array(train_dataset["train_set_y"][:])  
    
    # Load the test dataset from HDF5 file
    test_dataset = h5py.File('datasets/test_catvnoncat.h5', "r")
    
    # Extract test set features (images) as a NumPy array
    test_set_x_orig = np.array(test_dataset["test_set_x"][:])  
    
    # Extract test set labels (0 = non-cat, 1 = cat) as a NumPy array
    test_set_y_orig = np.array(test_dataset["test_set_y"][:])  
    
    # Extract the list of class names (e.g., [b'non-cat', b'cat'])
    classes = np.array(test_dataset["list_classes"][:])  
    
    # Reshape labels to be row vectors of shape (1, m_train) and (1, m_test)
    # This ensures consistency with logistic regression implementation
    train_set_y_orig = train_set_y_orig.reshape((1, train_set_y_orig.shape[0]))
    test_set_y_orig = test_set_y_orig.reshape((1, test_set_y_orig.shape[0]))
    
    # Return all datasets and class list
    return train_set_x_orig, train_set_y_orig, test_set_x_orig, test_set_y_orig, classes
