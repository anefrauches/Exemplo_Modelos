from sklearn.cluster import KMeans
from yellowbrick.cluster import KElbowVisualizer
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_curve, auc
from itertools import cycle
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from yellowbrick.regressor import ResidualsPlot

#FIND BEST K
model = KMeans()
visualizer = KElbowVisualizer(model, k=(1,8))
visualizer.fit(var_teste1_op2)        # Fit the data to the visualizer
visualizer.show()

#KNN
n_clusters = 5
model = KMeans(n_clusters)

#PAIRPLOT
var_teste2_op2['pred'] = model.fit_predict(var_teste2_op2)
g = sns.pairplot(var_teste2_op2,hue = "pred", height=5)
base_teste_teste2_op2['pred'] = model.predict(base_teste_teste2_op2)


#REGRESSÃO LOGÍSTICA
treino_mod = dict()
teste_mod = dict()

fpr_treino = dict()
tpr_treino = dict()
roc_auc_treino = dict()

fpr_teste = dict()
tpr_teste = dict()
roc_auc_teste = dict()

colors = cycle(['aqua', 'darkorange', 'cornflowerblue', 'pink','red'])
plt.figure(figsize=[15,5])
for i, color in zip(range(n_clusters), colors):
    treino_mod[i] = var_teste2_op2[var_teste2_op2['pred']==i].join(treino[target_class])
    teste_mod[i] = base_teste_teste2_op2[base_teste_teste2_op2['pred']==i].join(teste[target_class])
    
    X = treino_mod[i][var]
    y = treino_mod[i][target_class]
    pred = LogisticRegression(multi_class ='ovr', solver = 'lbfgs').fit(X, y)

    fpr_treino[i], tpr_treino[i], _ = roc_curve(treino_mod[i][target_class],pred.decision_function(treino_mod[i][var]))
    roc_auc_treino[i] = auc(fpr_treino[i], tpr_treino[i])
    
    fpr_teste[i], tpr_teste[i], _ = roc_curve(teste_mod[i][target_class],pred.decision_function(teste_mod[i][var]))
    roc_auc_teste[i] = auc(fpr_teste[i], tpr_teste[i])
    
    plt.subplot(1,2,1)
    plt.plot(fpr_treino[i], tpr_treino[i], color=color,
             label='ROC curve of model {0} (area = {1:0.2f})'
             ''.format(i, roc_auc_treino[i]))
    
    
    plt.subplot(1,2,2)
    plt.plot(fpr_teste[i], tpr_teste[i], color=color,
             label='ROC curve of model {0} (area = {1:0.2f})'
             ''.format(i, roc_auc_teste[i]))

    
plt.subplot(1,2,1)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC_treino')
plt.legend(loc="lower right")

plt.subplot(1,2,2)
plt.plot([0, 1], [0, 1], color='black', linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC_teste')
plt.legend(loc="lower right")

#REGRESSÃO LINEAR
treino_mod = dict()
teste_mod = dict()
pred = dict()

colors = cycle(['aqua', 'darkorange', 'cornflowerblue'])

for i, color in zip(range(n_clusters), colors):
    treino_mod[i] = var_teste2_op2[var_teste2_op2['pred']==i].join(treino[target_reg])
    teste_mod[i] = base_teste_teste2_op2[base_teste_teste2_op2['pred']==i].join(teste[target_reg])
    
    X = treino_mod[i][var]
    y = treino_mod[i][target_reg]
    model = LinearRegression().fit(X, y)
    pred[i] = model.predict(teste_mod[i][var])
    plt.figure()
    plt.figure(figsize=[15,5])
    plt.subplot(1,2,1)
    visualizer = ResidualsPlot(model, hist=False)
    visualizer.fit(X, y)
    visualizer.score(teste_mod[i][var],teste_mod[i][target_reg])
    
    plt.subplot(1,2,2)
    plt.scatter(pred[i],teste_mod[i][target_reg], color='darkorange')
    plt.title('Target x Predict')
    plt.xlabel('Predict')
    plt.ylabel('True value')
    visualizer.show()
