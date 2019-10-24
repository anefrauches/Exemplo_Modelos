from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from sklearn.preprocessing import StandardScaler


X_train, X_test, y_train, y_test = train_test_split(base[var], base[target_class], test_size=0.33, random_state=42)
scaler = StandardScaler()
scaler.fit(X_train)
X_treino = scaler.transform(X_train)
y_treino = y_train

scaler.fit(X_test)
X_teste = scaler.transform(X_test)
y_teste = y_test

#construção da rede
model = Sequential()
model.add(Dense(12, input_dim=len(var), activation='relu', kernel_initializer='he_uniform'))
model.add(Dense(1, activation='sigmoid'))
opt = SGD(lr=0.01, momentum=0.9)
model.compile(loss='binary_crossentropy', optimizer=opt, metrics=['accuracy', 'mean_squared_error'])
history = model.fit(X_treino, y_treino,validation_data=(X_teste, y_teste), epochs=100, verbose=0)
# plot loss during training
plt.figure(figsize=[15,5])
plt.subplot(121)
plt.title('Loss')
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.legend()
# plot accuracy during training
plt.subplot(122)
plt.title('Accuracy')
plt.plot(history.history['acc'], label='train')
plt.plot(history.history['val_acc'], label='test')
plt.legend()
plt.show()
y_pred_treino = model.predict(X_treino).ravel()
fpr_treino, tpr_treino, _ = roc_curve(y_treino, y_pred_treino)
auc_treino = auc(fpr_treino, tpr_treino)

y_pred_teste = model.predict(X_teste).ravel()
fpr_teste, tpr_teste, _ = roc_curve(y_teste, y_pred_teste)
auc_teste = auc(fpr_teste, tpr_teste)

plt.figure(figsize=[15,5])
plt.subplot(121)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_treino, tpr_treino, label='Treino (area = {:.3f})'.format(auc_treino))

plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_teste, tpr_teste, label='Teste (area = {:.3f})'.format(auc_teste))
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()
