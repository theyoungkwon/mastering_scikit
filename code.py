import sys, os, random
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import Normalize
from matplotlib.colors import ListedColormap
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import StandardScaler
from sklearn.datasets import make_moons, make_circles, make_classification
from sklearn.linear_model import SGDClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis
from sklearn.metrics import *


# find a current file' directory path.
try:
    dirpath = os.path.dirname(__file__)
except Exception as inst:
    dirpath = ''
    pass
f_name1 = os.path.join(dirpath,"../datasets/breast-cancer.npz")
f_name2 = os.path.join(dirpath,"../datasets/diabetes.npz")
f_name3 = os.path.join(dirpath,"../datasets/digit.npz")
f_name4 = os.path.join(dirpath,"../datasets/iris.npz")
f_name5 = os.path.join(dirpath,"../datasets/wine.npz")


# Utility function to move the midpoint of a colormap to be around
# the values of interest.
class MidpointNormalize(Normalize):

    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))


class ClassModels:

    def __init__(self):
        self.name = ''
        self.grid = ''
        self.param_grid = ''
        self.cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
        self.scoring = 'neg_log_loss' #'accuracy', 'f1', 'precision', 'recall', 'roc_auc'

    def trainModel(self, cname):
        if (cname == "Logistic Regression"):
            self.trainLogisticRegression()
        elif (cname == "Linear SVM"):
            self.trainLinearSVM()
        elif (cname == "RBF SVM"):
            self.trainRBFSVM()
        elif (cname == "Neural Nets"):
            self.trainNeuralNets()
        else:
            print("Please put existing classifier names")
        pass

    # run CV according to params for each classifier
    def trainLogisticRegression(self):
        # TODO: try different scoring rule such as Accuracy (default), F1-measure, AUC
        loss_range = ['log']
        penalty_range = ['l2','l1','none']
        alpha_range = np.geomspace(1.e-07, 1.e+05, num=13)  # 13 params
        self.param_grid = dict(loss=loss_range, penalty=penalty_range, alpha=alpha_range, max_iter=[1000], tol=[1e-3])
        self.grid = GridSearchCV(SGDClassifier(), param_grid=self.param_grid, cv=self.cv,
            n_jobs=-1)
        pass

    def trainLinearSVM(self):
        kernel_range = ['linear']
        C_range = np.geomspace(1.e-07, 1.e+05, num=13)  # 13 params :
        self.param_grid = dict(kernel=kernel_range, C=C_range)
        self.grid = GridSearchCV(SVC(), param_grid=self.param_grid, cv=self.cv,
                    n_jobs=-1)
        pass

    def trainRBFSVM(self):
        # params C / gamma
        kernel_range = ['rbf']
        C_range = np.geomspace(1.e-07, 1.e+05, num=13)  # 13 params :
        gamma_range = np.array([0.001,0.005,0.01,0.05,0.1,0.5,1,2,3])  # 9 params
        self.param_grid = dict(kernel=kernel_range, gamma=gamma_range, C=C_range)
        self.grid = GridSearchCV(SVC(), param_grid=self.param_grid, cv=self.cv,
                    n_jobs=-1)
        pass

    def trainNeuralNets(self):
        # early stopping default False, Momentum default 0.9
        hidden_layer_sizes_range = np.array([1,2,3,4,5,6,7,8,9,10,16,32]) # 12 params
        activation_range = ['logistic']
        solver_range = ['sgd']
        learning_rate_init_range = np.array([1.0e-04,1.0e-03,1.0e-02,1.0e-01]) # 4 params
        self.param_grid = dict(hidden_layer_sizes=hidden_layer_sizes_range,
                               activation=activation_range,solver=solver_range,
                               learning_rate_init=learning_rate_init_range,
                               max_iter=[1000])
        self.grid = GridSearchCV(MLPClassifier(), param_grid=self.param_grid, cv=self.cv,
            n_jobs=-1)
        pass


class Report:
    def __init__(self):
        pass

    # Loss + Accuracy (training + test)
    # auc + confusion matrix
    # cpu computation time
    def showResult(self, model, predicted_test, target_test, predicted_train, target_train):
        print("The best parameters are %s with a score of %0.3f"
              % (model.grid.best_params_, model.grid.best_score_))
        print("The Train Log Loss %0.3f  Zero one loss %f"
              % (log_loss(target_train, predicted_train), zero_one_loss(target_train, predicted_train)))
        print("The test Log Loss %0.3f  Zero one loss %f"
              % (log_loss(target_test, predicted_test), zero_one_loss(target_test, predicted_test)))
        print("The train Accuracy %0.3f"
              % (accuracy_score(target_train, predicted_train)))
        print("The test Accuracy %0.3f"
              % (accuracy_score(target_test, predicted_test) ))
        print("The test AUC of %0.3f"
              % (roc_auc_score(target_test, predicted_test) ))
        print("The mean training time of %f"
              % (np.mean(model.grid.cv_results_['mean_fit_time'], axis=0)) )
        print("The mean test time of %f"
              % (np.mean(model.grid.cv_results_['mean_score_time'], axis=0)) )
        # confusion matrix
        print("confusion matrix / precision recall scores")
        print ( confusion_matrix(target_test, predicted_test) )
        print ( classification_report(target_test, predicted_test) )
        pass

    def showPlot(self, model, clfname):

        if (clfname == "Logistic Regression"):
            self.showLogisticRegression(model, clfname)
        elif (clfname == "Linear SVM"):
            self.showLinearSVM(model, clfname)
        elif (clfname == "RBF SVM"):
            self.showRBFSVM(model, clfname)
        elif (clfname == "Neural Nets"):
            self.showNeuralNets(model, clfname)
        else:
            print("Please put existing classifier names")
        pass

    def showLogisticRegression(self, model, clfname):
        penalty_range = model.param_grid['penalty']
        alpha_range = model.param_grid['alpha']  # 13 params

        scores = np.array(model.grid.cv_results_['mean_test_score'])
        min_score = scores.min()
        max_score = scores.max()
        mean_score = np.mean(scores, axis=0)
        scores = scores.reshape(len(alpha_range),len(penalty_range))

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                   norm=MidpointNormalize(vmin=min_score, vmax=max_score, midpoint=mean_score))
        plt.xlabel('penalty')
        plt.ylabel('alpha (regularization)')
        plt.colorbar()
        plt.xticks(np.arange(len(penalty_range)), penalty_range, rotation=45)
        plt.yticks(np.arange(len(alpha_range)), alpha_range)
        plt.title('Validation accuracy')
        # plt.show()
        pass

    def showLinearSVM(self, model, clfname):
        C_range = model.param_grid['C']
        scores = np.array(model.grid.cv_results_['mean_test_score'])
        min_score = scores.min()
        max_score = scores.max()
        mean_score = np.mean(scores, axis=0)
        scores = scores.reshape(len(C_range),1)

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                   norm=MidpointNormalize(vmin=min_score, vmax=max_score, midpoint=mean_score))
        plt.ylabel('C')
        plt.colorbar()
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
        # plt.show()
        pass

    def showRBFSVM(self, model, clfname):
        C_range = model.param_grid['C']
        gamma_range = model.param_grid['gamma']
        # scores = model.grid.cv_results_['mean_test_score'].reshape(len(C_range), len(gamma_range))
        scores = np.array(model.grid.cv_results_['mean_test_score'])
        min_score = scores.min()
        max_score = scores.max()
        mean_score = np.mean(scores, axis=0)
        scores = scores.reshape(len(C_range), len(gamma_range))

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        # plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
        #            norm=MidpointNormalize(vmin=0.2, midpoint=0.92))
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                   norm=MidpointNormalize(vmin=min_score,vmax=max_score, midpoint=mean_score))
        plt.xlabel('gamma')
        plt.ylabel('C')
        plt.colorbar()
        plt.xticks(np.arange(len(gamma_range)), gamma_range, rotation=45)
        plt.yticks(np.arange(len(C_range)), C_range)
        plt.title('Validation accuracy')
        # plt.show()
        pass

    def showNeuralNets(self, model, clfname):
        hidden_layer_sizes_range = model.param_grid['hidden_layer_sizes']
        learning_rate_init_range = model.param_grid['learning_rate_init']

        scores = np.array(model.grid.cv_results_['mean_test_score'])
        min_score = scores.min()
        max_score = scores.max()
        mean_score = np.mean(scores, axis=0)
        scores = scores.reshape(len(learning_rate_init_range), len(hidden_layer_sizes_range))

        plt.figure(figsize=(8, 6))
        plt.subplots_adjust(left=.2, right=0.95, bottom=0.15, top=0.95)
        plt.imshow(scores, interpolation='nearest', cmap=plt.cm.hot,
                   norm=MidpointNormalize(vmin=min_score,vmax=max_score, midpoint=mean_score))
        plt.xlabel('hidden_layer_sizes')
        plt.ylabel('learning_rate_init')
        plt.colorbar()
        plt.xticks(np.arange(len(hidden_layer_sizes_range)), hidden_layer_sizes_range, rotation=45)
        plt.yticks(np.arange(len(learning_rate_init_range)), learning_rate_init_range)
        plt.title('Validation accuracy')
        # plt.show()
        pass

def plotLROverTime(data_x, loss_y, acc_y, idx):
    # Set the style globally
    # Alternatives include bmh, fivethirtyeight, ggplot,
    # dark_background, seaborn-deep, etc
    plt.style.use('ggplot')

    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.serif'] = 'Ubuntu'
    plt.rcParams['font.monospace'] = 'Ubuntu Mono'
    plt.rcParams['font.size'] = 10
    plt.rcParams['axes.labelsize'] = 10
    plt.rcParams['axes.labelweight'] = 'bold'
    plt.rcParams['axes.titlesize'] = 10
    plt.rcParams['xtick.labelsize'] = 8
    plt.rcParams['ytick.labelsize'] = 8
    plt.rcParams['legend.fontsize'] = 10
    plt.rcParams['figure.titlesize'] = 12

    # Set an aspect ratio
    width, height = plt.figaspect(1.68)
    fig = plt.figure(figsize=(width, height), dpi=400)

    plt.plot(data_x, loss_y, linewidth=0.5, linestyle=':', marker='o',
             markersize=2, label='loss')
    plt.plot(data_x, acc_y, linewidth=0.5, linestyle='--', marker='v',
             markersize=2, label='accuracy')
    plt.xlabel('Data Points')
    plt.ylabel('Score')

    # Axes alteration to put zero values inside the figure Axes
    # Avoids axis white lines cutting through zero values - fivethirtyeight style
    xmin, xmax, ymin, ymax = plt.axis()
    plt.axis([xmin - 0.1, xmax + 0.1, ymin, ymax])
    plt.title('LR performance over time', fontstyle='italic')
    plt.legend(loc='best', numpoints=1, fancybox=True)

    # Space plots a bit
    plt.subplots_adjust(hspace=0.25, wspace=0.40)

    plt.savefig('./LR_overtime_'+str(idx)+'.png', bbox_inches='tight')
    pass

def batches(l, n):
    for i in range(0, len(l), n):
        yield l[i:i+n]

def runLROverTime(train_X, train_y, test_X, test_y, idx):
    clf = SGDClassifier(loss='log')  # shuffle=True is useless here
    shuffledRange = range(train_X.shape[0])
    n_iter = 10
    data_point = 0
    f_loss = open('./LR_overtime_loss_'+str(idx)+'.txt', 'w')
    f_acc = open('./LR_overtime_acc_'+str(idx)+'.txt', 'w')
    data_x = []
    loss_y = []
    acc_y = []
    # temp_loss = zero_one_loss(train_y, clf.predict(train_X))
    # temp_acc = accuracy_score(train_y, clf.predict(train_X))
    # f_loss.write("data_point= " + str(data_point) + " zero_one_loss= " + str(temp_loss) + " \n")
    # f_acc.write("data_point= " + str(data_point) + " accuracy= " + str(temp_acc) + " \n")
    # data_x.append(data_point)
    # loss_y.append(temp_loss)
    # acc_y.append(temp_acc)
    for n in range(n_iter):
        shuffledRange = list(shuffledRange)
        random.shuffle(shuffledRange)
        shuffledX = [train_X[i] for i in shuffledRange]
        shuffledY = [train_y[i] for i in shuffledRange]
        for batch in batches(range(len(shuffledX)), 10):
            clf.partial_fit(shuffledX[batch[0]:batch[-1] + 1], shuffledY[batch[0]:batch[-1] + 1],
                             classes=np.unique(train_y))
            data_point += len(batch)
            temp_loss = zero_one_loss(train_y, clf.predict(train_X))
            temp_acc = accuracy_score(train_y, clf.predict(train_X))
            f_loss.write("data_point= " + str(data_point) + " zero_one_loss= " + str(temp_loss) + " \n")
            f_acc.write("data_point= " + str(data_point) + " accuracy= " + str(temp_acc) + " \n")
            data_x.append(data_point)
            loss_y.append(temp_loss)
            acc_y.append(temp_acc)

    f_loss.write("\n===== End of Training / Test Set Results =====\n")
    f_loss.write("data_point= %d , zero_one_loss= %f\n" % (data_point, zero_one_loss(test_y, clf.predict(test_X))))
    f_acc.write("\n===== End of Training / Test Set Results =====\n")
    f_acc.write("data_point= %d , accuracy= %f\n" % (data_point, accuracy_score(test_y, clf.predict(test_X))))
    f_loss.close()
    f_acc.close()
    plotLROverTime(data_x, loss_y, acc_y, idx)
    pass


class RunEval:

    def __init__(self):
        self.dnames = [f_name1, f_name2, f_name3, f_name4, f_name5]
        self.train_X = []
        self.train_y = []
        self.test_X = []
        self.test_y = []

    def run(self):
        report = Report()
        for idx, dname in enumerate(self.dnames):
            # load data
            if len(sys.argv) > 1 and int(sys.argv[1]) != idx:
                continue
            data = np.load(dname)
            self.train_y = data['train_Y']
            self.test_y = data['test_Y']
            # standardize data (mean=0, std=1)
            self.train_X = StandardScaler().fit_transform(data['train_X'])
            self.test_X = StandardScaler().fit_transform(data['test_X'])
            print ("shape of data set  ", self.train_X.shape, self.train_y.shape, self.test_X.shape, self.test_y.shape)

            if len(sys.argv) > 2 and int(sys.argv[2]) == 1:
                runLROverTime(self.train_X, self.train_y, self.test_X, self.test_y, idx)
                continue

            clfnames = ["Logistic Regression", "Linear SVM", "RBF SVM", "Neural Nets"]
            # clfnames = ["RBF SVM"]
            # clfnames = ["Linear SVM"]

            for idx2, clfname in enumerate(clfnames):
                print("===== %s " %(dname))
                print("===== %s" %(clfname))
                # (1) train model with CV                model = ClassModels()
                model = ClassModels()
                model.trainModel(clfname)
                model.grid.fit(self.train_X, self.train_y)

                # (2) show results
                predicted_test = model.grid.predict(self.test_X)
                predicted_train = model.grid.predict(self.train_X)
                # Loss + Accuracy (training + test)
                # auc + confusion matrix
                # cpu computation time
                report.showResult(model, predicted_test, self.test_y, predicted_train, self.train_y)
                report.showPlot(model, clfname)
                plt.savefig('./'+clfname+'_'+str(idx)+'.png', bbox_inches = 'tight')


if __name__ == '__main__':
    eval = RunEval()
    eval.run()
    exit()