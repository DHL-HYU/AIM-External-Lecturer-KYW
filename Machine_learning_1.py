import os
import numpy as np
import scipy
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import confusion_matrix
import matplotlib
import matplotlib.pyplot as plt

defPath = 'C:\\workshop_221130\\' # change to the path where the data is located
fileName = 'MMN_workshop_data.mat'

filePath = os.path.join(defPath, fileName)

loadedMat = scipy.io.loadmat(filePath)

MMN = loadedMat['MMN']

#%% Feature selection
x = MMN[0:30, 0:9]
y = MMN[30:60, 0:9]

h, p = scipy.stats.ttest_ind(x, y)
B = np.sort(p)
I = np.argsort(p)

#%% 분류를 위한 data 준비
data = MMN[:,[5,8]]
group = MMN[:,-1]

#%% Cross-validation
number_kfold = 10
indices = StratifiedKFold(number_kfold)

#%% LDA (linear discriminant analysis)
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis

accuracy_LDA = np.zeros(number_kfold)
sensitivity_LDA = np.zeros(number_kfold)
specificity_LDA = np.zeros(number_kfold)
cnt = 0

for trainInd, testInd in indices.split(data, group):
    train = data[trainInd]
    test = data[testInd]
    trainGroup = group[trainInd]
    testGroup = group[testInd]
    
    LDAmodel = LinearDiscriminantAnalysis()
    LDAmodel.fit(train, trainGroup)
    
    predictedLabel = LDAmodel.predict(test)
    confusionMatrix = confusion_matrix(testGroup, predictedLabel)
    
    TP = confusionMatrix[0,0]
    TN = confusionMatrix[1,1]
    FP = confusionMatrix[1,0]
    FN = confusionMatrix[0,1]
    
    accuracy_LDA[cnt] = (TP+TN)/(TP+FP+FN+TN)
    sensitivity_LDA[cnt] = TP/(TP+FN)
    specificity_LDA[cnt] = TN/(TN+FP)
    
    cnt += 1

final_accuracy_LDA = np.mean(accuracy_LDA)*100
final_sensitivity_LDA = np.mean(sensitivity_LDA)*100
final_specificity_LDA = np.mean(specificity_LDA)*100

#%% LDA plot
X0, X1 = data[group==1], data[group==2]

plt.scatter(X0[:,0], X0[:, 1], color="red")
plt.scatter(X1[:,0], X1[:, 1], color="blue")

xlimit = [-8.2, 1.3]
ylimit = [-8.2, 1.3]

x_min, x_max = xlimit[0], xlimit[1]
y_min, y_max = ylimit[0], ylimit[1]

plt.xlim(xlimit); plt.ylim(ylimit)

xx, yy = np.meshgrid(np.linspace(x_min, x_max, 2000), np.linspace(y_min, y_max, 2000))
Z = LDAmodel.predict_proba(np.c_[xx.ravel(), yy.ravel()])
Z = Z[:, 1].reshape(xx.shape) >= 0.5

cmap = matplotlib.colormaps['Pastel1']
plt.pcolormesh(xx, yy, Z, cmap=cmap , zorder=0)
plt.contour(xx, yy, Z, [0.5], linewidths=2.0)


#%% SVM (Support vector machine)
from sklearn import svm

accuracy_SVM = np.zeros(number_kfold)
sensitivity_SVM = np.zeros(number_kfold)
specificity_SVM = np.zeros(number_kfold)
cnt = 0

for trainInd, testInd in indices.split(data, group):
    train = data[trainInd]
    test = data[testInd]
    trainGroup = group[trainInd]
    testGroup = group[testInd]
    
    # SVMmodel = svm.LinearSVC(C=1, max_iter=10000)
    # SVMmodel = svm.SVC(C=1, kernel="poly", degree=2)
    # SVMmodel = svm.SVC(C=1, kernel="poly", degree=3)
    SVMmodel = svm.SVC(C=1, kernel="rbf")
    SVMmodel.fit(train, trainGroup)
    
    predictedLabel = SVMmodel.predict(test)
    confusionMatrix = confusion_matrix(testGroup, predictedLabel)
    
    TP = confusionMatrix[0,0]
    TN = confusionMatrix[1,1]
    FP = confusionMatrix[1,0]
    FN = confusionMatrix[0,1]
    
    accuracy_SVM[cnt] = (TP+TN)/(TP+FP+FN+TN)
    sensitivity_SVM[cnt] = TP/(TP+FN)
    specificity_SVM[cnt] = TN/(TN+FP)
    
    cnt += 1

final_accuracy_SVM = np.mean(accuracy_SVM)*100
final_sensitivity_SVM = np.mean(sensitivity_SVM)*100
final_specificity_SVM = np.mean(specificity_SVM)*100

#%% SVM plot
from sklearn.inspection import DecisionBoundaryDisplay
X0, X1 = data[group==1], data[group==2]

DecisionBoundaryDisplay.from_estimator(SVMmodel, data, ax=plt.gca(),
        grid_resolution=2000, response_method="predict", cmap=plt.cm.Pastel1,
        alpha=0.8)

plt.scatter(X0[:,0], X0[:, 1], color="red")
plt.scatter(X1[:,0], X1[:, 1], color="blue")

xlimit = [-8.2, 1.3]
ylimit = [-8.2, 1.3]

plt.xlim(xlimit); plt.ylim(ylimit)

decision_function = SVMmodel.decision_function(data)
support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
support_vectors = data[support_vector_indices]

plt.scatter(support_vectors[:,0], support_vectors[:,1],
            s=100, facecolor="none", edgecolors="k")

DecisionBoundaryDisplay.from_estimator(SVMmodel, data, ax=plt.gca(),    
        grid_resolution=50, plot_method="contour", levels=[-1, 0, 1],        
        colors="k", alpha=0.5, linestyles=["--", "-", "--"])

#%% define SVM plot (additional)
def SVMplot(SVMmodel, data, group):
    X0, X1 = data[group==1], data[group==2]
    
    DecisionBoundaryDisplay.from_estimator(SVMmodel, data, ax=plt.gca(),
            grid_resolution=2000, response_method="predict", cmap=plt.cm.Pastel1,
            alpha=0.8)
    
    plt.scatter(X0[:,0], X0[:, 1], color="red")
    plt.scatter(X1[:,0], X1[:, 1], color="blue")
    
    xlimit = [-8.2, 1.3]
    ylimit = [-8.2, 1.3]
    
    plt.xlim(xlimit); plt.ylim(ylimit)
    
    decision_function = SVMmodel.decision_function(data)
    support_vector_indices = np.where(np.abs(decision_function) <= 1 + 1e-15)[0]
    support_vectors = data[support_vector_indices]
    
    plt.scatter(support_vectors[:,0], support_vectors[:,1],
                s=100, facecolor="none", edgecolors="k")
    
    DecisionBoundaryDisplay.from_estimator(SVMmodel, data, ax=plt.gca(),
            grid_resolution=50, plot_method="contour", levels=[-1, 0, 1],
            colors="k", alpha=0.5, linestyles=["--", "-", "--"])

SVMplot(SVMmodel, data, group)





