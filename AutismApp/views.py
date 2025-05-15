from django.shortcuts import render
from django.template import RequestContext
from django.contrib import messages
from django.http import HttpResponse
import os
from django.core.files.storage import FileSystemStorage
import cv2
import io
import base64
import numpy as np
import matplotlib.pyplot as plt
from keras.utils.np_utils import to_categorical
from keras.layers import  MaxPooling2D
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Convolution2D, Conv2D, BatchNormalization
from keras.models import Sequential, load_model, Model
import pickle
from keras.callbacks import ModelCheckpoint
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import confusion_matrix
import seaborn as sns
import os.path
import os
import tensorflow as tf
import time
from GANModel import GAN
import sys
sys.path.append('./tools/')
from utils import save_images, save_source
from data_generator import ImageDataGenerator
import warnings
warnings.filterwarnings('ignore')

global uname, X_train, X_test, y_train, y_test, X, Y, cnn_model, gan_model, sess

path = "Dataset"
labels = []
X = []
Y = []

config = tf.ConfigProto()
config.gpu_options.allow_growth = True
generator = ImageDataGenerator(batch_size = 32, height = 128, width = 128, z_dim = 256, scale_size=(128, 128), shuffle=False, mode='train')
val_generator = ImageDataGenerator(batch_size = 32, height = 128, width = 128, z_dim = 256, scale_size=(128, 128), shuffle=False, mode='test')


for root, dirs, directory in os.walk(path):
    for j in range(len(directory)):
        name = os.path.basename(root)
        if name not in labels:
            labels.append(name.strip())

def getLabel(name):
    index = -1
    for i in range(len(labels)):
        if labels[i] == name:
            index = i
            break
    return index

def calculateMetrics(algorithm, predict, y_test):
    global accuracy, precision, recall, fscore
    a = accuracy_score(y_test,predict)*100
    p = precision_score(y_test, predict,average='macro') * 100
    r = recall_score(y_test, predict,average='macro') * 100
    f = f1_score(y_test, predict,average='macro') * 100
    conf_matrix = confusion_matrix(y_test, predict)
    return conf_matrix, a, p, r, f 

def LoadDatasetAction(request):
    if request.method == 'POST':
        global X, Y, labels, X_train, X_test, y_train, y_test
        if os.path.exists("model/X.txt.npy"):
            X = np.load('model/X.txt.npy')
            Y = np.load('model/Y.txt.npy')
        else:
            for root, dirs, directory in os.walk(path):
                for j in range(len(directory)):        
                    name = os.path.basename(root)
                    if 'Thumbs.db' not in directory[j]:
                        img = cv2.imread(root+"/"+directory[j])
                        img = cv2.resize(img, (32, 32))
                        X.append(img)
                        label = getLabel(name)
                        Y.append(label)
            X = np.asarray(X)
            Y = np.asarray(Y)
            np.save('model/X.txt',X)
            np.save('model/Y.txt',Y)
        X = X.astype('float32')
        X = X/255
        indices = np.arange(X.shape[0])
        np.random.shuffle(indices)
        X = X[indices]
        Y = Y[indices]
        Y = to_categorical(Y)
        X_train, X_test, y_train, y_test = train_test_split(X, Y, test_size=0.2) #split dataset into train and test    
        output = "Dataset Images Loading Completed<br/>Total Images Found in Dataset = "+str(X.shape[0])+"<br/>"
        output += "Features available in each image = "+str(X.shape[1] * X.shape[2] * X.shape[3])+"<br/>"
        output += "Class Labels found in Dataset = "+str(labels)+"<br/><br/>Dataset Train & Test Split Details<br/>"
        output += "80% Dataset Images used for training = "+str(X_train.shape[0])+"<br/>"
        output += "20% Dataset Images used for testing = "+str(X_test.shape[0])+"<br/><br/><br/><br/>"
        context= {'data': output}
        return render(request, 'UserScreen.html', context)


def TrainModel(request):
    if request.method == 'GET':
        global cnn_model, gan_model, X_train, X_test, y_train, y_test, labels, sess
        with tf.Graph().as_default():
            sess = tf.Session(config=config)
            gan_model = GAN(sess=sess, lr = 0.001, keep_prob = 1., model_num = None, batch_size = 32, age_loss_weight = None, gan_loss_weight = None,
                            fea_loss_weight = None, tv_loss_weight = None)
            gan_model.imgs = tf.placeholder(tf.float32, [32, 128, 128, 3])
            gan_model.true_label_features_128 = tf.placeholder(tf.float32, [32, 128, 128, 5])
            gan_model.ge_samples = gan_model.generate_images(gan_model.imgs, gan_model.true_label_features_128, stable_bn=False, mode='train')
            gan_model.get_vars()
            gan_model.saver = tf.train.Saver(gan_model.save_g_vars)
            # Start running operations on the Graph.
            sess.run(tf.global_variables_initializer())
            if gan_model.load(gan_model.saver, 'acgan', 399999):
                print("GAN model successfully loaded")
        cnn_model = Sequential()
        cnn_model.add(Convolution2D(32, (3 , 3), input_shape = (X_train.shape[1], X_train.shape[2], X_train.shape[3]), activation = 'relu'))
        cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
        cnn_model.add(Convolution2D(32, (3, 3), activation = 'relu'))
        cnn_model.add(MaxPooling2D(pool_size = (2, 2)))
        cnn_model.add(Flatten())
        cnn_model.add(Dense(units = 256, activation = 'relu'))
        cnn_model.add(Dropout(0.3))
        cnn_model.add(Dense(units = y_train.shape[1], activation = 'softmax'))
        cnn_model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
        if os.path.exists("model/cnn_weights.hdf5") == False:
            model_check_point = ModelCheckpoint(filepath='model/cnn_weights.hdf5', verbose = 1, save_best_only = True)
            hist = cnn_model.fit(X_train, y_train, batch_size = 32, epochs = 30, validation_data=(X_test, y_test), callbacks=[model_check_point], verbose=1)
            f = open('model/cnn_history.pckl', 'wb')
            pickle.dump(hist.history, f)
            f.close()    
        else:
            cnn_model.load_weights("model/cnn_weights.hdf5")
        predict = cnn_model.predict(X_test)
        predict = np.argmax(predict, axis=1)
        y_test1 = np.argmax(y_test, axis=1)
        cm, a, p, r, f  = calculateMetrics("GAN + CNN", predict, y_test1)
        plt.figure(figsize =(5, 3)) 
        ax = sns.heatmap(cm, xticklabels = labels, yticklabels = labels, annot = True, cmap="viridis" ,fmt ="g");
        ax.set_ylim([0,len(labels)])
        plt.title("GAN + CNN Confusion matrix") 
        plt.xticks(rotation=90)
        plt.ylabel('True class') 
        plt.xlabel('Predicted class')
        cols = ['Algorithm Name', 'Accuracy', 'Precision', 'Recall', 'Fscore']
        output = '<table border="1" align="center" width="100%"><tr>'
        font = '<font size="" color="black">'
        for i in range(len(cols)):
            output += "<td>"+font+cols[i]+"</font></td>"
        output += "</tr>"
        output += "<tr><td>"+font+"GAN + CNN</font></td>"
        output += "<td>"+font+str(a)+"</font></td>"
        output += "<td>"+font+str(p)+"</font></td>"
        output += "<td>"+font+str(r)+"</font></td>"
        output += "<td>"+font+str(f)+"</font></td></tr>"
        output += "</table><br/>"    
        buf = io.BytesIO()
        plt.savefig(buf, format='png', bbox_inches='tight')
        plt.close()
        img_b64 = base64.b64encode(buf.getvalue()).decode()    
        context= {'data':output, 'img': img_b64}
        return render(request, 'UserScreen.html', context)

def GANAugment(filename, predict):
    global gan_model, sess
    arr = ['original','GAN1', 'GAN2', 'GAN3', 'GAN4']
    img_list = []
    source = val_generator.load_imgs(filename, 128)
    train_imgs = generator.load_train_imgs("tools/train", 128)
    temp = np.reshape(source, (1, 128, 128, 3))
    save_source(temp, [1, 1], "output/"+arr[0]+".jpg")
    images = np.concatenate((temp, train_imgs), axis=0)
    for j in range(1, generator.n_classes):
        true_label_fea = generator.label_features_128[j]
        dict = {
            gan_model.imgs: images,
            gan_model.true_label_features_128: true_label_fea,
            }
        samples = sess.run(gan_model.ge_samples, feed_dict=dict)
        image = np.reshape(samples[0, :, :, :], (1, 128, 128, 3))
        save_images(image, [1, 1], "output/"+arr[j]+".jpg")
    for i in range(len(arr)):
        img = cv2.imread("output/"+arr[i]+".jpg")
        img = cv2.resize(img, (280,200))
        cv2.putText(img, arr[i]+" "+predict, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        img_list.append(img)            
    images = cv2.hconcat(img_list)
    return images
        
def DetectAutismAction(request):
    if request.method == 'POST':
        global uname, labels, gan_model, cnn_model
        filename = request.FILES['t1'].name
        image = request.FILES['t1'].read() #reading uploaded file from user
        if os.path.exists("AutismApp/static/"+filename):
            os.remove("AutismApp/static/"+filename)
        with open("AutismApp/static/"+filename, "wb") as file:
            file.write(image)
        file.close()
        cnn_model = load_model("model/cnn_weights.hdf5")
        test_img = cv2.imread("AutismApp/static/"+filename)
        test_img = cv2.resize(test_img, (32, 32))
        print(str(test_img.shape)+"=================")
        test_img = test_img.reshape(1,32,32,3)
        test_img = test_img.astype('float32')
        test_img = test_img/255
        predict = cnn_model.predict(test_img)
        predict = np.argmax(predict)
        predict = labels[predict]
        gan_image = GANAugment("AutismApp/static/"+filename, predict)
        #gan_image = cv2.resize(gan_image, (800, 400))
        #gan_image = cv2.cvtColor(gan_image, cv2.COLOR_BGR2RGB)
        #cv2.putText(gan_image, 'CNN Features Classification = '+predict, (10, 25),  cv2.FONT_HERSHEY_SIMPLEX,0.7, (0, 255, 0), 2)
        cv2.imshow("GAN + CNN Features Classification = "+predict, gan_image)
        cv2.waitKey(0)
        return render(request, 'DetectAutism.html', {})   

def DetectAutism(request):
    if request.method == 'GET':
        return render(request, 'DetectAutism.html', {}) 

def LoadDataset(request):
    if request.method == 'GET':
        return render(request, 'LoadDataset.html', {})     

def UserLogin(request):
    if request.method == 'GET':
       return render(request, 'UserLogin.html', {})  

def index(request):
    if request.method == 'GET':
       return render(request, 'index.html', {})

def UserLoginAction(request):
    if request.method == 'POST':
        global uname
        username = request.POST.get('username', False)
        password = request.POST.get('password', False)
        page = "UserLogin.html"
        status = "Invalid Login" 
        if "admin" == username and "admin" == password:
            page = "UserScreen.html"
            status = "Welcome Admin"
        context= {'data': status}
        return render(request, page, context)



