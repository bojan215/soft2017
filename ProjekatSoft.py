# -*- coding: utf-8 -*-
"""
Created on Tue Jan 23 22:19:47 2018

@author: bojan
"""
import math
import numpy as np
import cv2
import glob, os
from scipy import ndimage
from sklearn.datasets import fetch_mldata


from skimage import color

from skimage.measure import label
from skimage.measure import regionprops



def ispisiPostojeceVID():
    #svi avi video zapisi se nalaze u videos folderu
    direktorijum="videos/"
    
    print("############### Video fajlovi ##################")
    for VID in glob.glob(os.path.join(direktorijum,'*.avi')):
        nazivVid=os.path.splitext(VID)[0]
        print(os.path.basename(nazivVid))
        


ispisiPostojeceVID()
izabranVid=input("Uneti broj video snimka iz liste: video-")
video="videos/video-"+izabranVid+".avi"
print("Ucitavanje...")
suma=0
#video="videos/video-9.avi"
video0=cv2.VideoCapture(video)


def dot(v,w): 
    x,y=v
    X,Y=w
    return x*X+y*Y

def length(v):
    x,y=v
    return math.sqrt(x*x+y*y)

def vector(b,e):
    x,y=b
    X,Y=e
    return (X-x,Y-y)

def unit(v):
    x,y=v
    mag=length(v)
    return (x/mag,y/mag)

def scale(v,sc):
    x,y=v
    return (x*sc,y*sc)

def distance(p0,p1):
    return length(vector(p0,p1))


def add(v,w):
    x,y=v
    X,Y=w
    return (x+X,y+Y)

def pnt2line2(pnt, start, end):
    line_vec = vector(start, end)
    pnt_vec = vector(start, pnt)
    line_len = length(line_vec)
    line_unitvec = unit(line_vec)
    pnt_vec_scaled = scale(pnt_vec, 1.0 / line_len)
    t = dot(line_unitvec, pnt_vec_scaled)
    r = 1
    if t < 0.0:
        t = 0.0
        r = -1
    elif t > 1.0:
        t = 1.0
        r = -1
    nearest = scale(line_vec, t)
    dist = distance(nearest, pnt_vec)
    nearest = add(nearest, start)
    return (dist, (int(nearest[0]), int(nearest[1])), r)
    




def dodajBroj(br):
    global suma
    suma+=br







def houghTransformation(frejm,siva):
    x0=1000
    x1=-10
    y0=1000
    y1=-10
    minVal=50
    maxVal=150
    aperture_size=3
    threshold=40;
    minDuzina=100
    maxGap=8
    ivice=cv2.Canny(siva,minVal,maxVal,aperture_size)

    
    linije=cv2.HoughLinesP(ivice,1,np.pi/180,threshold,minDuzina,maxGap)

    lnl=len(linije)
    for i in range(lnl): 
        x01 = linije[i][0][0]
        y01 = linije[i][0][1]
        x02 = linije[i][0][2]
        y02 = linije[i][0][3]
        
        if  x01 < x0:
            y0 = y01
            x0 = x01
        if  x02 > x1:
            x1 = x02
            y1 = y02
                

    return x0,y0,x1,y1



def hough(video):
    kernel = np.ones((2,2),np.uint8)
    cap = cv2.VideoCapture(video)
    
    while(cap.isOpened()):
        ret, frejm = cap.read()
        siva = cv2.cvtColor(frejm, cv2.COLOR_BGR2GRAY)
        
        
        siva = cv2.dilate(siva,kernel)
        frejmTemp=frejm
        cap.release()
        cv2.destroyAllWindows()
        return houghTransformation(frejmTemp,siva)


     
x1,y1,x2,y2=hough(video)

ivice = [(x1, y1), (x2, y2)]


mnist=fetch_mldata('MNIST original',data_home='dta2') 
mnist_brojevi=[]
id = -1    
def sledeciID():
    global id
    id += 1
    return id

dozvDistanca=15    
def ddistanca(broj,brojevi) :
    rez = []
   
    for br in brojevi:
         
          if (distance(broj['centar'],br['centar']) < dozvDistanca) :

            rez.append(br)

    return rez
        

def pozicioniranjeSlike(slikaCB) :
    
    slika = np.zeros((28,28),np.uint8)
    xV  = -10
    xM  = 1000
    yV  = -10
    yM  = 1000
    
    sirina = 0
    visina = 0
    z = 0

    try :
        labelaSlika  = label(slikaCB)
        regioni = regionprops(labelaSlika)        
        while (z < len(regioni)):
            bbox = regioni[z].bbox
            if bbox[0] < xM:
                xM = bbox[0]
            if bbox[1] < yM:
                yM = bbox[1]
            if bbox[2] > xV:
                xV = bbox[2]
            if bbox[3] > yV:
                yV = bbox[3]
            z += 1
            
        visina = yV - yM
        sirina = xV - xM    
        slika [0 : sirina, 0 : visina] = slika [0 : sirina,0 : visina] + slikaCB[ xM : xV, yM : yV]
        return  slika
    
    except  ValueError: 
        print ("catch")    
        pass
        
def ucitajMnist(mnist):
    
    for i in range(70000):
        slika  =  mnist.data[i].reshape(28,28)
        bin_slika  =  ((color.rgb2gray(slika)/255.0)>0.80).astype('uint8')
        
        bin_slika  =  pozicioniranjeSlike(bin_slika)
        
        mnist_brojevi.append(bin_slika)
   
        

def nadjiNajbliziEl (lista,elem):
    nulti = lista[0]
    vr=nulti
    dList0IEl = distance(vr['centar'],elem['centar'])
    for el in lista:
        dListIEl = distance(el['centar'],elem['centar'])
        if dListIEl < dList0IEl:
            vr = el
    return vr




def detekcijaBroja(slika) :
    

    slikaCB=((color.rgb2gray(slika)/255.0)>0.80).astype('uint8')
    slika  =  pozicioniranjeSlike(slikaCB)
 
    minSuma  =  10000
    rez  =  -1
    for i in range(len(mnist_brojevi)) : 
        suma  =  0
        mnist_slika  =  mnist_brojevi[i]
        suma  =  np.sum(mnist_slika != slika)
       
        if suma  <  minSuma :
            minSuma  =  suma
            rez  =  mnist.target[i]
        i+=1
    return  rez
 
    

    
def main():
    
    brojevi = []
    frejm = 0
    
    ucitajMnist(mnist)

    donja = np.array([160 , 160 , 160],dtype = "uint8")
    gornja = np.array([255 , 255 , 255],dtype = "uint8")
    kernel = np.ones((2,2),np.uint8)
    while(1): 
        ret, slika = video0.read()
        if not ret: break
        
        maska = cv2.inRange(slika, donja, gornja)#uzmi samo bele 
        
        slikaCB = maska * 1.0
        slikaCB2 = slikaCB
        
        slikaCB = cv2.dilate(slikaCB,kernel)
        
        
        #niz br nadjenih objekata
        slikaCBLabel,niz = ndimage.label(slikaCB)
        objekti = ndimage.find_objects(slikaCBLabel)
        #print("BR NADJENIH"+format(niz))
               
        for i in range(niz): 
            duzina = []
            centarObjekta = []
            lokacija = objekti[i]
           
           
            duzina.append(lokacija[1].stop - lokacija[1].start)           
            duzina.append(lokacija[0].stop - lokacija[0].start)
            
            centarObjekta.append((lokacija[1].stop + lokacija[1].start) /2)
            centarObjekta.append((lokacija[0].stop + lokacija[0].start) /2)

            if duzina[0] > 10 or duzina[1] > 10 : 
                broj = {'centar' : centarObjekta, 'duzina' : duzina, 'frejm' : frejm}
               
                rezultat = ddistanca(broj,brojevi)
                
                lnrez=len(rezultat)
               
                if lnrez == 0 :
                   
                    x11 = centarObjekta[0] - 14
                    y11 = centarObjekta[1] - 14
                    x22 = centarObjekta[0] + 14
                    y22 = centarObjekta[1] + 14
                    broj['id'] = sledeciID()
                    broj['prosao'] = False                      
                    broj['vrednost'] = detekcijaBroja(slikaCB2[int(y11):int(y22),int(x11):int(x22)])
                    broj['slika'] = slikaCB2[int(y11):int(y22),int(x11):int(x22)]
                    try:
                        cv2.imshow('br',broj['slika'])
                        cv2.imshow('frejmUhv',slika)
                        cv2.waitKey()
                    except cv2.error:
                        pass
                    print ("Pojavio se broj: " + format(int(broj['vrednost'])))
                    brojevi.append(broj)
                else:
                    
                    br = nadjiNajbliziEl(rezultat,broj)
                    br['centar'] = broj['centar']
                    br['frejm'] = broj['frejm']
                    
        for br in brojevi :
            razlikaF  =  frejm - br['frejm']
            #print("razlikaF JE"+format(frejm)+"--"+format(br['frejm'])+"=="+format(razlikaF))
            if ( razlikaF < 3 ): 
                dist, pnt,r  =  pnt2line2(br['centar'],ivice[0],ivice[1])
                if r  >  0 :
                    if dist < 10 :
                        if br['prosao'] == False:
                            br['prosao'] = True
                            (x,y) = br['centar']
                            print (">Prosao je broj: " + format(int(br['vrednost'])))
   
                            dodajBroj(br['vrednost'])
                            
      
        frejm += 1
      
    print("#####################################################")
    print("Video: "+video)      
    print ("Suma svih brojeva koji su prosli ispod linije: " + format(int(suma)))
    print("#####################################################")
    video0.release()
    cv2.destroyAllWindows()
main()





    
    
    
    
   


