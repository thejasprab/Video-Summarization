import cv2
import numpy as np
import os
import glob
from matplotlib import pyplot as plt
import pandas as pd
from  sklearn . cluster  import  KMeans
from  sklearn . metrics  import  pairwise_distances_argmin_min
def  elbowMethod ( X ):
    num_iterations  =  20

    Nc  =  range ( 1 , num_iterations )
    kmeans  = [ KMeans ( n_clusters = i ) for  i  in  Nc ]
    kmeans
    score  = [ kmeans [ i ]. fit ( X ). score ( X ) for  i  in  range ( len ( kmeans ))]
    score
    plt . plot ( Nc , score )
    plt . xlabel ( 'Number of clusters' )
    plt . ylabel ( 'K score' )
    plt . title ( 'Bend of the elbow' )
    plt . show ()
def  summary_color ( histr ):
    result  =  0
    colorSum  =  0
    index  =  0
    for  idx , value  in  enumerate ( histr ):
        colorSum   =  colorSum+value
        index  =  idx

    return  colorSum / 256

def imgtohist(noofframes):
    file_list = glob.glob('./data1/*.*')
    img_number=1
    my_list=[] 
    path = "./data1/*.*"
    histrRecord=[]
    for file in glob.glob(path):
        #print(file)
        color  = ( 'b' , 'g' , 'r' )
        a= cv2.imread(file)  
        #c = cv2.cvtColor(a, cv2.COLOR_BGR2RGB)
        #for  i , col  in  enumerate ( color ):
        histr  =  cv2 . calcHist ([ a ], [ 0 ], None , [ 256 ], [ 0 , 255 ])
        plt.plot(histr)
        SummaryOfColor  =  summary_color ( histr )
        histrRecord . append ( SummaryOfColor )
        #cv2.imwrite("images/test_images/Color_image"+str(img_number)+".jpg", c)
        img_number +=1 
        #print ( 'processed image:'  +  image )
    plt . rcParams [ 'figure.figsize' ] = ( 16 , 9 )
    plt . style . use ( 'ggplot' )

    data  =  pd . DataFrame ({
        'x' : [ i  for  i  in  range ( 1 , len ( histrRecord ))],
        'y' : [ float(histrRecord[ i ])  for  i  in  range ( 1 , len ( histrRecord ))]
        })
    #print(data.head())
    f1  =  data [ 'x' ]. values
    f2  =  data [ 'y' ]. values 
    X  =  np . array ( list ( zip ( f1 , f2 )))
    #print(X)
    colors  = [ 'b' , 'g' , 'r' , 'c' , 'm' , 'y' , 'k' , 'w' ]

    elbowMethod ( X )

    print ( 'Indicate the number of K in reference to the previous graph' )
    
    k_value  =  input ()

    kmeans  =  KMeans ( n_clusters = int ( k_value )). fit ( X )
    centroids  =  kmeans . cluster_centers_
    labels  =  kmeans . predict ( X )
    assign  = []
    for  row  in  labels :
        assign . append ( colors [ row ])
    
    C  =  centroids
    plt . scatter ( X [:, 0 ], X [:, 1 ], c = assign , s = 20 )
    plt . scatter ( C [:, 0 ], C [:, 1 ], marker  =  '*' , c = colors , s = 100 )
    
    plt . show ()
    
    closest , _  =  pairwise_distances_argmin_min ( centroids , X )
     
    listKeyFrames  = []
    for  captureNumber  in  closest :
        listKeyFrames . append ( 'capture-'  +  str ( captureNumber ) + '.jpg' )
    print(listKeyFrames)

def videotoframe():
    print('video to frame')
# Playing video from file:
    cap = cv2.VideoCapture('test.mp4')
    try:
        if not os.path.exists('data'):
            os.makedirs('data')
    except OSError:
                print ('Error: Creating directory of data')
    currentFrame = 0
    noofframes=0
    while(True):
    # Capture frame-by-frame
        ret, frame = cap.read()
        if frame is None:
            break
    # Saves image of the current frame in jpg file
        name = './data/' + str(currentFrame) + '.jpg'
        print ('Creating...' + name)
        cv2.imwrite(name, frame)
    # To stop duplicate images
        currentFrame += 1
        noofframes=noofframes+1
# When everything done, release the capture
    cap.release()
    cv2.destroyAllWindows()
    return noofframes

def main():
    noofframes=0
    inp=int(input("Enter 1 to make frames else 0:"))
    if(inp==1):
        noofframes=videotoframe()
    imgtohist(noofframes)
main()