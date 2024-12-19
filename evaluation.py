import numpy as np
from matplotlib import pyplot as plt
from sklearn import metrics

from CNN2 import model2
from fine_tuning import model3
from train import history, test_x, test_y
from sklearn.metrics import confusion_matrix, classification_report

plt.plot(history.history['accuracy'], label='train acc')
plt.plot(history.history['val_accuracy'], label='val acc')
plt.legend()
plt.show()
plt.plot(history.history['loss'], label='train loss')
plt.plot(history.history['val_loss'], label='val loss')
plt.plot(history.history['accuracy'], label='accuracy')
plt.plot(history.history['val_accuracy'], label='val-accuracy')
plt.legend()
plt.show()
model3.evaluate(test_x,test_y,batch_size=32)
y_pred=model3.predict(test_x)
y_pred=np.argmax(y_pred,axis=1)

#matrice de confusion
cm = confusion_matrix(y_pred, test_y)
cm_display = metrics.ConfusionMatrixDisplay(confusion_matrix=cm ,
                                            display_labels=["airplanes" , "cars" , "ships"])
cm_display.plot(cmap='Blues', values_format='d')
plt.xlabel('Predicted Labels')
plt.ylabel('True Labels')
plt.title('Confusion Matrix : ')
plt.show()

print(classification_report(y_pred,test_y))


