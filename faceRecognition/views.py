import os
from cv2 import cv2
import numpy as np
import scipy.linalg as s_linalg

from . import dataset
from . import imageToMatrix
from . import pca_Algo
from django.shortcuts import redirect, render
from django.http.response import StreamingHttpResponse
import base64
from django.http import HttpResponseServerError
from django.views.decorators import gzip

from PIL import Image
from accounts.models import Profile,Parameter
from Client.models import Client,Training
from django.contrib import messages

from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
import matplotlib.cm as cm
#image reco type =0 
#Group image reco type =1
#vedio reco type =2
from numpy import pi
from numpy import sin
from numpy import zeros
from numpy import r_
from scipy import signal
import scipy.misc # pip install Pillow
import matplotlib.pylab as pylab

reco_type = 2
no_of_imgs_of_one_person = 4
dataset_obj = dataset.datasetClass(4)

    # Dataset tarin
images_names_for_training = dataset_obj.images_name_for_train
labels_for_training = dataset_obj.labels_for_train
no_of_elements_for_training = dataset_obj.no_of_elements_for_train
target_names = dataset_obj.target_name_as_array

    # Dataset test

images_name_for_testing = dataset_obj.images_name_for_test
labels_for_testing = dataset_obj.labels_for_test
no_of_elements_for_testing = dataset_obj.no_of_elements_for_test

    #images_targets = dataset_obj.images_targets

    # ImageToMatrix
#img_width, img_height = 20, 20

#imageTOoMatrix_obj = imageToMatrix.imageTOoMatrix(images_names_for_training, img_width, img_height)
#scaled_face = imageTOoMatrix_obj.getMatrix()


    #if algo_type == "pca":
    #cv2.imshow("Original Image" , cv2.resize(np.array(np.reshape(scaled_face[:,1],[img_height, img_width]), dtype = np.uint8),(200, 200)))
    #cv2.waitKey()

    # PCA_Algo
#pca_Algo_obj = pca_Algo.Pca_Algo(scaled_face, labels_for_training, target_names, no_of_elements_for_training, 90)
#new_coordinates = pca_Algo_obj.reduce_dim()
    ##pca_Algo_obj.show_eigen_face(img_width, img_height, 50, 150, 0)
    # Recognition




def get_frame():
    droidcam = 'http://192.168.43.1:4747/mjpegfeed'
    user = Client.objects.values_list('username', flat=True)

    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
    camera =cv2.VideoCapture(0)
    
    para = Parameter.objects.filter(para_name='Parameter')
    for p in para: 
        scaleFactor = p.scaleFactor
        minNeighbors = int(p.minNeighbors)
        rec_color = p.rec_color
        rec_stroke = p.rec_stroke
        if(rec_color == "Green"):
             rec_color = (36,255,12)
        if(rec_color == "Blue"):
            rec_color = (255,0,0)
        if(rec_color == "Red"):
            rec_color = (0,0,255)
    while True:
        _, cap = camera.read()
        gray =cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=scaleFactor, minNeighbors=minNeighbors)
        
        #img = cv2.imread(gray, cv2.IMREAD_GRAYSCALE) # get grayscale image
        img_resized = cv2.resize(gray,(200,200))        
        imgBlurred = cv2.GaussianBlur(img_resized, (5,5), 0)  # blur

        imgThresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #threshold
    
        #cv2.imshow('Test Image',imgThresh)
        npaFlattenedImage = imgThresh.reshape((1, 200 * 200)) 
        m=[]   
        usernames=user
        rec  =''
        for v in usernames: 

            f= 'media/trained/'+v+'.png'  

            img2 = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
            img_resized2 = cv2.resize(img2,(200,200))
            imgBlurred2 = cv2.GaussianBlur(img_resized2, (5,5), 0)      
            imgThresh2 = cv2.adaptiveThreshold(imgBlurred2, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
            npaFlattenedImage2= imgThresh2.reshape((1, 200 * 200))  
    
            distance = dist(npaFlattenedImage, npaFlattenedImage2)
            #distance vector
            m.append( distance )
            pos=m.index(min(m))
            rec =usernames[pos]
       


        for(x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            scaled = cv2.resize(roi ,(40, 40))
            rec_color = rec_color
            rec_stroke=rec_stroke
            cv2.rectangle(cap, (x,y),(x+w, y+h), rec_color,rec_stroke)
            
            
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_color = rec_color
            font_stroke = rec_stroke
            if(rec != ""):
             cv2.putText(cap, rec, (x,y-10), font, 0.9, font_color, font_stroke)
            else:
             cv2.putText(cap, "Unknown", (x,y-10), font, 0.9, font_color, font_stroke)
            
        imgencode=cv2.imencode('.jpg',cap)[1]
        stringData=imgencode.tostring()
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
    del(camera)
    
        
    
def Recognition_Video(request): 
    para = Parameter.objects.filter(para_name='Parameter')
    context = {
       'para' : para,   
    }

    return render(request,'stream.html',context)

@gzip.gzip_page
def dynamic_stream(request,stream_path="video"):
    try :
        return StreamingHttpResponse(get_frame(),content_type="multipart/x-mixed-replace;boundary=frame")
    except :
        return "error"





 

#euclidean distance
def dist(x,y):   
    return np.sqrt(np.sum((x-y)**2))
 
def To_rec_image(request):
    clientimg= request.FILES['clientimg']
    im = Image.open(clientimg)
    #im.show()
    imgPath = 'media/uploadedImages/'+str(clientimg)
    im.save(imgPath, 'JPEG')
    context = {
       'clientimg' : imgPath,   
    }

    return render(request,'RecoIamge.html',context)

     

def Recognition_image(request): 
    
    user = Client.objects.values_list('username', flat=True)
    #user = Client._meta.get_field('username')
    #print(user)
    # inputImageToPredict= 'media/UserImage/3.jpg'

    clientimg= request.POST['clientimg']
    print(clientimg)
    #im = Image.open(clientimg)
    #im.show()
    imgPath = 'media/uploadedImages/'+str(clientimg)
    #im.save(imgPath, 'JPEG')
    inputImageToPredict = clientimg  
    img = cv2.imread(inputImageToPredict, cv2.IMREAD_GRAYSCALE) # get grayscale image
    img_resized = cv2.resize(img,(200,200))        
    imgBlurred = cv2.GaussianBlur(img_resized, (5,5), 0)  # blur

    imgThresh = cv2.adaptiveThreshold(imgBlurred,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2) #threshold
 
 
    #cv2.imshow('Test Image',imgThresh)
 
    npaFlattenedImage = imgThresh.reshape((1, 200 * 200)) 
    #print (npaFlattenedImage.shape)
    mean_to= npaFlattenedImage.mean()
 
    m=[]
 
    vowels=user
    cnt=0
    for v in vowels:
        print ('Read ' + v + ' mean image from directory !')
        f= 'media/trained/'+v+'.png'
        print(f)
        
        img2 = cv2.imread(f, cv2.IMREAD_GRAYSCALE)
        img_resized2 = cv2.resize(img2,(200,200))
        imgBlurred2 = cv2.GaussianBlur(img_resized2, (5,5), 0)      
        imgThresh2 = cv2.adaptiveThreshold(imgBlurred2, 255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
        npaFlattenedImage2= imgThresh2.reshape((1, 200 * 200))  
 
        distance = dist(npaFlattenedImage, npaFlattenedImage2)
        #distance vector
        m.append( distance )
    
    print ('Euclidean Distance Array: ')
    print (m)
    #Min Distance
    print ('Min Distance: ')
    print (min(m))
    #Array Position
    print ('Array Position: ')
    pos=m.index(min(m))
    print (pos)
    #Vowel Recognized
    print ('The Face Recognized Is : ')
    print (vowels[pos])
    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
    frame = cv2.imread(inputImageToPredict)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=3)


    for(x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        scaled = cv2.resize(roi_gray, (40, 40))
        rec_color = (36,255,12)
        rec_stroke=1
        cv2.rectangle(frame, (x,y),(x+w, y+h), rec_color,rec_stroke)

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_color = (36,255,12)
        font_stroke = 1
        cv2.putText(frame, vowels[pos], (x,y-10), font, 4, font_color, font_stroke)
       
        # cv2.imshow('Face', scaled)
        # cv2.waitKey()


    #frame = cv2.resize(frame, (780, 520))
    #cv2.imshow('Client Frame', frame)
    #Distance array
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    client = Client.objects.get(username=vowels[pos])
    context = {
       'client' : client,
        'clientimg':clientimg
    }
    return render(request,'RecoIamge.html',context)
    
#FaceDetect

def get_frame_for_face():
    
    face_cascade = cv2.CascadeClassifier("data/haarcascade_frontalface_alt2.xml")
    cam =cv2.VideoCapture(0) 
    while True:
        _, cap = cam.read()
        gray =cv2.cvtColor(cap, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4)
        
        #img = cv2.imread(gray, cv2.IMREAD_GRAYSCALE) # get grayscale image
        
        for(x, y, w, h) in faces:
            roi = gray[y:y+h, x:x+w]
            scaled = cv2.resize(roi ,(40, 40))
            rec_color = (30,144,255)
            rec_stroke=2
            cv2.rectangle(cap, (x,y),(x+w, y+h), rec_color,rec_stroke)
            
        imgcode=cv2.imencode('.jpg',cap)[1]
        stringData=imgcode.tostring()
        yield (b'--frame\r\n'b'Content-Type: text/plain\r\n\r\n'+stringData+b'\r\n')
    del(cam)
  
        
    
def detect_face(request): 
    return render(request,"detectFace.html")

@gzip.gzip_page
def face_dynamic_stream(request,stream_path_detect="videoface"):
    try :
        return StreamingHttpResponse(get_frame_for_face(),content_type="multipart/x-mixed-replace;boundary=frame")
    except :
        return "error"




def index(request):
    user = Profile.objects.get(user = request.user)
    return render(request, "index.html",{'user':user})


def unFlatten(vector,rows, cols):
    img = []
    cutter = 0
    while(cutter+cols<=rows*cols):
        try:
            img.append(vector[cutter:cutter+cols])
        except:
            img = vector[cutter:cutter+cols]
        cutter+=cols
    img = np.array(img)
    return img
 

# Construct the input matrix
 


def dct2(a):
    return scipy.fftpack.dct( scipy.fftpack.dct( a, axis=0, norm='ortho' ), axis=1, norm='ortho' )

def idct2(a):
    return scipy.fftpack.idct( scipy.fftpack.idct( a, axis=0 , norm='ortho'), axis=1 , norm='ortho')


def computeDCT2(img):
 
    im = cv2.imread(img).astype(float)
    imsize = im.shape
    dct = np.zeros(imsize)

    # Do 8x8 DCT on image (in-place)
    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            dct[i:(i+8),j:(j+8)] = dct2( im[i:(i+8),j:(j+8)] )
    
    # Threshold
    thresh = 0.012
    dct_thresh = dct * (abs(dct) > (thresh*np.max(dct)))
    im_dct = np.zeros(imsize)

    for i in r_[:imsize[0]:8]:
        for j in r_[:imsize[1]:8]:
            im_dct[i:(i+8),j:(j+8)] = idct2( dct_thresh[i:(i+8),j:(j+8)] )

    cv2.imwrite("media/DCTImages/dct.jpg",im_dct)
    return dct

def computeDCT(img):
    #clientimg= request.FILES['clientimg']
    para = Parameter.objects.filter(para_name='Parameter')
    for p in para: 
        blocksize = p.BlockSize

    imgPath = img
    #print(imgPath)
    B=blocksize #blocksize
    fn3= imgPath
    img1 = cv2.imread(fn3, cv2.IMREAD_GRAYSCALE)
    h,w=np.array(img1.shape[:2])/B * B
    img1=img1[:int(h),:int(w)]
    blocksV=h/B
    blocksH=w/B
    
    vis0 = np.zeros((int(h),int(w)), np.float32)
    Trans = np.zeros((int(h),int(w)), np.float32)
    vis0[:int(h), :int(w)] = img1
    for row in range(int(blocksV)):
            for col in range(int(blocksH)):
                    currentblock = cv2.dct(vis0[row*B:(row+1)*B,col*B:(col+1)*B])
                    Trans[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
    #cv2.imwrite("media/DCTImages/1.jpg",np.asarray(Trans))

    back0 = np.zeros((int(h),int(w)), np.float32)
    for row in range(int(blocksV)):
            for col in range(int(blocksH)):
                    currentblock = cv2.idct(Trans[row*B:(row+1)*B,col*B:(col+1)*B])
                    back0[row*B:(row+1)*B,col*B:(col+1)*B]=currentblock
    #cv2.imwrite('media/DCTImages/BackTransformed.jpg',np.asarray(back0))
    return np.asarray(back0)

def training(request):
    client_notrain = Client.objects.filter(is_trained = False)
    #dataset_with_client = Dataset.objects.all()
    train_client = Training.objects.all()
    context = {
       'train_client' : train_client,
       'client_notrain':client_notrain,
       #'dataset_with_client':dataset_with_client
    }

    return render(request,'train.html',context)   

def train(request):
    Clients = Client.objects.values_list('username', flat=True).filter(is_created_ds=True,is_trained=False) 
    face=Clients  
    for v in face:
        in_matrix = None 
        imgcnt=0
       
        for f in os.listdir(os.path.join('media/dataset/',v)):
            imgcnt+=1
            print(f)
            # Read the image in as a gray level image. 
            #imgp = cv2.imread(os.path.join('media/dataset/',v, f), cv2.IMREAD_GRAYSCALE)
            img = computeDCT(os.path.join('media/dataset/',v, f))
            img_resized = cv2.resize(img,(200,200))
            # let's resize them to w * h 
            vec = img_resized.reshape(200 * 200)
            # stack them up to form the matrix
           
            #resize image to 32x32
            # convert back
            try:
              in_matrix = np.vstack((in_matrix, vec))
            #
            except:
                in_matrix = vec
            
            
            
            # PCA 
        if in_matrix is not None:
            mean, eigenvectors = cv2.PCACompute(in_matrix, np.mean(in_matrix, axis=0).reshape(1,-1))
 
        img = unFlatten(mean.transpose(),200,200)
        pathimg='media/trained/'+v+'.png'
        cv2.imwrite('media/trained/'+v+'.png',img)
        tr = Training(clientname=v,TrainImg=pathimg)
        tr.save() 
        Client.objects.filter(username=v).update(is_trained=True)
 
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    messages.success(request, 'Your profile is updated successfully!')
    return redirect('training')


from scipy.fftpack import dct, idct
from skimage import io,color 
import numpy as np
import matplotlib.pylab as plt

# implement 2D IDCT
def idct2(a):
    return idct(idct(a.T, norm='ortho').T, norm='ortho') 




def update_pca_para(request):
    if request.method == 'POST':
       width_pca = request.POST.get('Width_pca')
       height_pca = request.POST.get('Height_pca')
       scaleFactor = request.POST.get('scaleFactor')
       minNeighbors = request.POST.get('minNeighbors')
       blockSize = request.POST.get('BlockSize')
       

       Parameter.objects.filter(para_name="Parameter").update(width_pca=width_pca,height_pca=height_pca,scaleFactor=scaleFactor,minNeighbors=minNeighbors,BlockSize=blockSize)
      
    return redirect('settings')
def update_frame_para(request):
    if request.method == 'POST':
       width_frame = request.POST.get('width_frame')
       height_frame = request.POST.get('height_frame')
       color_rec = request.POST.get('color_rec')
       stroke_rec = request.POST.get('stroke_rec')

       Parameter.objects.filter(para_name="Parameter").update(width_frame=width_frame,height_frame=height_frame,rec_color=color_rec,rec_stroke=stroke_rec)
      
    return redirect('settings')
def get_para(request):
    profiles = Profile.objects.all()
    para_pca = Parameter.objects.all()
    
    context = {
       'para_pca' : para_pca,
       'profiles':profiles
       
    }
    return render(request,'accounts/settings.html',context)