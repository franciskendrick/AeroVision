from sklearn.metrics import multilabel_confusion_matrix, accuracy_score
# from keras.models import load_model
from keras._tf_keras.keras.models import load_model
from preprocess_data import preprocess_data
import numpy as np

x_train, x_test, y_train, y_test = preprocess_data()

model = load_model("action.h5")

#
yhat = model.predict(x_test)
ytrue = np.argmax(y_test, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat)) 
print(accuracy_score(ytrue, yhat))

#
yhat = model.predict(x_train)
ytrue = np.argmax(y_train, axis=1).tolist()
yhat = np.argmax(yhat, axis=1).tolist()

print(multilabel_confusion_matrix(ytrue, yhat)) 
print(accuracy_score(ytrue, yhat))