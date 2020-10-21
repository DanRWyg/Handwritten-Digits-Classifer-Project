'''
main project file for handwritten digits classifier testing
author: Daniel Wygant
date: 04/10/2020
'''


import numpy as np
import matplotlib.pyplot as plt
from time import time
from sklearn import ensemble 
from sklearn import svm
from sklearn import neural_network
from sklearn.model_selection import GridSearchCV

def read_dataset(name, verbose=True):

    if verbose:
        print('loading {}...'.format(name), end='', flush=True)

    f = open(name, mode='rb')

    # 4 byte header
    magic_number = f.read(4) 

    # 4th byte in header indicates n of data dimensions
    n_dim = magic_number[3] 

    dim_sizes = []

    # next n_dim 32-bit integers indicate the respective sizes of the dimensions
    for i in range(n_dim):       
        dim_sizes.append(int.from_bytes(f.read(4), byteorder='big')) 

    # read the data
    raw_bytes = f.read() 

    # create a numpy array with the specified dimensions with byte elements 
    data = np.array([ raw_bytes[i] for i in range(len(raw_bytes)) ], dtype='uint8').reshape(dim_sizes) 

    if verbose:
        print('done')

    return data


def dataset_subset(labels, images, percentage=1.0):
    # return a random subset of images and labels of the size determined by percentage    
    indices = np.random.choice(labels.shape[0], size=int(labels.shape[0] * percentage), replace=False)
    return labels[indices], images[indices] 


def save_results(results, file_name, best_index=-1, duration=-1, header=None, print_results=True):
    # save the results of a GridSearchCV call

    res_file = open(file_name, 'w')

    res_file.write((file_name if header == None else header) + '\n')

    for k, v in results.items():
        res_file.write('{:20} : {}\n\n'.format(k, v))

    if best_index >= 0:
        res_file.write('\nBest\n')

        for k, v in results.items():
            res_file.write('{:20} : {}\n\n'.format(k, v[best_index]))

    if duration >= 0:
        res_file.write('\nDuration = {}'.format(duration))

    res_file.close()

    if print_results:
        res_file = open(file_name, 'r')
        print(res_file.read())
        res_file.close()

    
def perform_RF_tuning(train_images, train_labels, params, display_graphs=True):
    # initialize random forest classifier and gridsearch
    rf = ensemble.RandomForestClassifier()
    clf = GridSearchCV(estimator=rf, param_grid=params, cv=5, n_jobs=-1, verbose=2)

    # perform gridsearch of hyperparameters
    start_time = time()
    clf.fit(train_images, train_labels)
    fit_duration = time() - start_time 

    # save the results
    save_results(clf.cv_results_, 
                 file_name='results/RF_results.txt',
                 best_index=clf.best_index_, 
                 duration=fit_duration,
                 header='Random Forest Results')

    # create graphs
    param_indices = { i : [] for i in params['n_estimators'] }

    for i, param in enumerate(clf.cv_results_['param_n_estimators']):
        param_indices[param].append(i)

    # graph versus classification accuracy
    for param, indices in param_indices.items():
        plt.plot(clf.cv_results_['param_max_features'][indices] * 100,
                 clf.cv_results_['mean_test_score'][indices],
                 label='n estimators = {}'.format(param))

    plt.legend()
    plt.title('Random Forest Classifier Score')
    plt.xlabel('max features (% of 784)')
    plt.ylabel('score (% correctly classified)')

    plt.savefig('results/RF_Classifier_Score.png')
    if display_graphs:
        plt.show()
    plt.close()

    # graph versus training time
    for param, indices in param_indices.items():
        plt.plot(clf.cv_results_['param_max_features'][indices] * 100,
                 clf.cv_results_['mean_fit_time'][indices],
                 label='n estimators = {}'.format(param))

    plt.legend()
    plt.title('Random Forest Train Time')
    plt.xlabel('max features (% of 784)')
    plt.ylabel('time (seconds)')

    plt.savefig('results/RF_Train_Time.png')
    if display_graphs:
        plt.show()
    plt.close()


def perform_SVC_tuning(train_images, train_labels, params, display_graphs=True):
    # initialize support vector machine classifier and gridsearch
    svc = svm.SVC()
    clf = GridSearchCV(estimator=svc, param_grid=params, cv=5, n_jobs=-1, verbose=2)
    
    # perform gridsearch of hyperparameters
    start_time = time()
    clf.fit(train_images, train_labels)
    fit_duration = time() - start_time 

    # save the results
    save_results(clf.cv_results_, 
                 file_name='results/SVC_results.txt',
                 best_index=clf.best_index_, 
                 duration=fit_duration,
                 header='Support Vector Machine Results')
    
    # create graphs
    param_indices = { i : [] for i in params['C'] }

    for i, param in enumerate(clf.cv_results_['param_C']):
        param_indices[param].append(i)
    
    # graph versus classificattion accuracy
    for param, indices in param_indices.items():
        plt.plot(clf.cv_results_['param_gamma'][indices],
                 clf.cv_results_['mean_test_score'][indices],
                 label='C = {}'.format(param))

    plt.legend()
    plt.title('Support Vector Machine Classifier Score')
    plt.xlabel('gamma')
    plt.ylabel('score (% correctly classified)')

    plt.savefig('results/SVC_Classifier_Score.png')
    if display_graphs:
        plt.show()
    plt.close()

    # graph versus training time
    for param, indices in param_indices.items():
        plt.plot(clf.cv_results_['param_gamma'][indices],
                 clf.cv_results_['mean_fit_time'][indices],
                 label='C = {}'.format(param))

    plt.legend()
    plt.title('Support Vector Machine Train Time')
    plt.xlabel('gamma')
    plt.ylabel('time (seconds)')

    plt.savefig('results/SVC_Train_Time.png')
    if display_graphs:
        plt.show()
    plt.close()


def perform_NN_tuning(train_images, train_labels, params, display_graphs=True):
    # initialize neural network classifier and gridsearch
    nn = neural_network.MLPClassifier()
    clf = GridSearchCV(estimator=nn, param_grid=params, cv=5, n_jobs=-1, verbose=2)

    # perform gridsearch of hyperparameters
    start_time = time()
    clf.fit(train_images, train_labels)
    fit_duration = time() - start_time 

    # save the results
    save_results(clf.cv_results_, 
                 file_name='results/NN_results.txt',
                 best_index=clf.best_index_, 
                 duration=fit_duration,
                 header='Neural Network Results')

    # create graphs
    param_indices = { i : [] for i in params['hidden_layer_sizes'] }

    for i, param in enumerate(clf.cv_results_['param_hidden_layer_sizes']):
        param_indices[param].append(i)

    # graph versus classification accuracy
    for param, indices in param_indices.items():
        plt.plot(clf.cv_results_['param_alpha'][indices],
                 clf.cv_results_['mean_test_score'][indices],
                 label='hidden layer size = {}'.format(param))

    plt.legend()
    plt.title('Neural Network Classifier Score')
    plt.xlabel('alpha')
    plt.ylabel('score (% correctly classified)')

    plt.savefig('results/NN_Classifier_Score.png')
    if display_graphs:
        plt.show()
    plt.close()

    # graph versus training time
    for param, indices in param_indices.items():
        plt.plot(clf.cv_results_['param_alpha'][indices],
                 clf.cv_results_['mean_fit_time'][indices],
                 label='hidden layer size = {}'.format(param))

    plt.legend()
    plt.title('Neural Network Train Time')
    plt.xlabel('alpha')
    plt.ylabel('time (seconds)')

    plt.savefig('results/NN_Train_Time.png')
    if display_graphs:
        plt.show()
    plt.close()


def test_RF(train_images, train_labels, test_images, test_labels):

    # create random forest classifier with best hyperparameters found in the tuning.
    rf = ensemble.RandomForestClassifier(n_estimators=50, max_features=0.111)

    # train random forest classifier on 100% of MNIST training dataset
    start_time = time()
    rf.fit(train_images, train_labels)
    fit_duration = time() - start_time
  
    # classify 100% of the MNIST test dataset
    start_time = time()
    pred = rf.predict(test_images)
    pred_duration = time() - start_time

    # calculate classification accuracy
    acc = (pred == test_labels).sum() / float(test_images.shape[0])

    # claculate 95% confidence interval
    conf_interv = 1.96 * np.sqrt((acc * (1 - acc)) / test_images.shape[0])

    return acc, conf_interv, fit_duration, pred_duration


def test_SVC(train_images, train_labels, test_images, test_labels):

    # create support vector machine classifier with best hyperparemeters found in the tuning.
    svc = svm.SVC(C=0.1, gamma=0.03333)

    # train support vector machine classifier on 100% of MNIST training dataset
    start_time = time()
    svc.fit(train_images, train_labels)
    fit_duration = time() - start_time
  
    # classify 100% of the MNIST test dataset
    start_time = time()
    pred = svc.predict(test_images)
    pred_duration = time() - start_time

    # calculate classification accuracy
    acc = (pred == test_labels).sum() / float(test_images.shape[0])

    # calculate 95% confidence interval
    conf_interv = 1.96 * np.sqrt((acc * (1 - acc)) / test_images.shape[0])

    return acc, conf_interv, fit_duration, pred_duration


def test_NN(train_images, train_labels, test_images, test_labels):

    # create neural network classifier with best hyperparameters found in the tuning.
    nn = neural_network.MLPClassifier(hidden_layer_sizes=(50, 50), alpha=1e-10, learning_rate='adaptive')

    # train neural network classifier on 100% of MNIST training dataset
    start_time = time()
    nn.fit(train_images, train_labels)
    fit_duration = time() - start_time
  
    # classify 100% of the MNIST test dataset
    start_time = time()
    pred = nn.predict(test_images)
    pred_duration = time() - start_time

    # calculate classification accuracy
    acc = (pred == test_labels).sum() / float(test_images.shape[0])

    # calculate 95% confidence interval
    conf_interv = 1.96 * np.sqrt((acc * (1 - acc)) / test_images.shape[0])

    return acc, conf_interv, fit_duration, pred_duration


def main():

    plt.rcParams['figure.figsize'] = (12, 6) # define the figure size of a saved figure

    train_labels     = read_dataset('MNIST_Train_Labels') # read train labels from file
    train_images_raw = read_dataset('MNIST_Train_Images') # read train images from file
    train_images     = train_images_raw.reshape((-1, train_images_raw.shape[1] * train_images_raw.shape[2])) / 255.0 # reshape images from 28x28 to 784 features and normalize

    test_labels      = read_dataset('MNIST_Test_Labels') # read test labels from file
    test_images_raw  = read_dataset('MNIST_Test_Images') # read test images from file
    test_images      = test_images_raw.reshape((-1, test_images_raw.shape[1] * test_images_raw.shape[2])) / 255.0 # reshape images from 28x28 to 784 features and normalize


    # get a 50% subset of the MNIST training set
    train_labels_, train_images_ = dataset_subset(train_labels, train_images, percentage=0.5)

    rf_params = { 'n_estimators' : np.arange(1, 6) * 10, 
                  'max_features' : np.linspace(0.0001, 1.0, num=10) }

    # perform random forest tuning
    #perform_RF_tuning(train_images_, train_labels_, rf_params, display_graphs=False)           # UNCOMMENT THIS LINE TO PERFORM RF TUNING

    svc_params = { 'C'    : np.linspace(1e-10, 0.1, num=5), 
                  'gamma' : np.linspace(0, 0.15, num=10) }

    # perform support vector machine tuning
    #perform_SVC_tuning(train_images_, train_labels_, svc_params, display_graphs=False)         # UNCOMMENT THIS LINE TO PERFORM SVC TUNING

    nn_params = { 'hidden_layer_sizes' : [ (l, l) for l in range(10, 60, 10) ],
                  'alpha' : np.linspace(1e-10, 10, num=10),
                  'learning_rate' : [ 'adaptive' ] }

    # perform neural network tuning
    #perform_NN_tuning(train_images_, train_labels_, nn_params, display_graphs=False)           # UNCOMMENT THIS LINE TO PERFORM NN TUNING

    triv_params = { 'hidden_layer_sizes' : [ () ],
                    'alpha' : [ 1e-7 ],
                    'learning_rate' : [ 'adaptive' ],
                    'max_iter' : [ 1000 ]}
    # perform MNIST triviality (non-linear separable) test with 0 hidden layers.
    #perform_NN_tuning(train_images, train_labels, triv_params)                                 # UNCOMMENT THIS LINE TO PERFORM NN TRIVIALITY TEST

    

    # perform test of algorithms trained on 100% of the MNIST training set
    # print results of prediction on 100% of the MNIST test set
                                                                                                # UNCOMMENT THESE LINES TO PERFORM CLASSIFIERS ON MNIST TEST SET 

    #rf_acc, rf_conf_interv, rf_fit_duration, rf_pred_duration = test_RF(train_images, train_labels, test_images, test_labels)
    #print('rf acc={}, rf conf interv={}, rf fit duration={}, rf pred duration={}'.format(rf_acc, rf_conf_interv, rf_fit_duration, rf_pred_duration))
    
    #svc_acc, svc_conf_interv, svc_fit_duration, svc_pred_duration = test_SVC(train_images, train_labels, test_images, test_labels)
    #print('svc acc={}, svc conf interv={}, svc fit duration={}, svc pred duration={}'.format(svc_acc, svc_conf_interv, svc_fit_duration, svc_pred_duration))
    
    #nn_acc, nn_conf_interv, nn_fit_duration, nn_pred_duration = test_NN(train_images, train_labels, test_images, test_labels)
    #print('nn acc={}, nn conf interv={}, nn fit duration={}, nn pred duration={}'.format(nn_acc, nn_conf_interv, nn_fit_duration, nn_pred_duration))
    
    
if __name__ == '__main__':
    main()
