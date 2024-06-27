import numpy as np
import matplotlib.pyplot as plt
import tkinter as tk
import os
from tkinter import *
from tkinter import filedialog
from scipy import optimize
import pandas as pd
import matplotlib.figure 
from matplotlib.backends.backend_tkagg import (FigureCanvasTkAgg, NavigationToolbar2Tk)

np.set_printoptions(suppress=True)

window=tk.Tk()
window.title("Collinearaity Analyzer")

isPoint0 = True
isPoint1 = True
isPoint2 = True
isPoint3 = True
isPoint4 = True
isPoint5 = True
isPoint6 = True
isPoint7 = True

def browseFolder(): #user selects test folder, gets test files and name/type of stave
    global folder_selected
    folder_selected = filedialog.askdirectory(initialdir=r"c:\Users\Admin\Desktop\ATLAS\Optical Test Results")
    if folder_selected:
        files = [os.path.join(folder_selected,file) for file in os.listdir(folder_selected)]
        global file0
        global file1
        global file2
        for file in files:
            if file.split("_")[-1]=="7-13.txt" or file.split("_")[-1]=="13-7.txt":
                file0=file
            if file.split("_")[-1]=="3-8.txt" or file.split("_")[-1]=="8-3.txt":
                file1=file
            if file.split("_")[-1]=="0-5.txt" or file.split("_")[-1]=="5-0.txt":
                file2=file
        file0_lbl.config(text=f"{folder_selected}")
        global staveType
        fileName = os.path.basename(folder_selected)
        if not fileName[0]=="O":
            staveType = fileName.split('_')[0]+"_"+ fileName.split('_')[-1]
        else:
            staveType = fileName.split('_')[-1]
fileFrame = tk.Frame(master=window, relief=tk.SUNKEN, borderwidth=2)
fileFrame.pack()

file713_btn = tk.Button(master=fileFrame, text="Select Measurement Folder",command=browseFolder)
file713_btn.grid(row=0, column=0, pady=10)
file0_lbl = tk.Label(master = fileFrame, text="No File Selected")
file0_lbl.grid(row=0, column=1, pady=10)

def passFail(array): #determines if stave passes or fails
    if (np.max(array)<0.1 and np.min(array) > -0.1):
        test_lbl.config(text="The Stave Core has PASSED", bg="green")
    else:
        test_lbl.config(text="The Stave Core has FAILED", bg="red")



def runScript(): #main analyzing script

    if not os.path.exists(folder_selected + "\\Stave" +staveType +"_Test Results"):
        resultFolder = folder_selected + "\\Stave" +staveType +"_Test Results"
        os.mkdir(resultFolder)

    f0 = np.loadtxt(file0, skiprows=7, usecols=(2,3))
    f0 = np.concatenate([[[0,0]],f0], axis=0)

    f1 = np.loadtxt(file1, skiprows=7, usecols=(2,3))
    f1 = np.concatenate([[[0,0]],f1], axis=0)

    f2 = np.loadtxt(file2, skiprows=7, usecols=(2,3))
    f2 = np.concatenate([[[0,0]],f2], axis=0)

    near_fiducial_idxs0 = np.array([0,2,4,6,8,10,12])
    near_fiducial_idxs1 = np.array([0,2,4,6,8,10])
    near_fiducial_idxs2 = np.array([0,2,4,6,8,10])

    common_fiducials01 = np.array([
    [10,   0],
    [12,   2],
    [30, 30],
    ])

    common_lockpoints01 = np.array([
    [15, 25],
    [18, 28],
    ])

    common_fiducials12 = np.array([
    [6,   0],
    [8,   2],
    [10,   4],    
    [34, 30],
    [36, 32],
    ])

    common_lockpoints12 = np.array([
    [13,  19],
    [16,  22],
    [19,  25],
    [22,  28],
    ])
    fiducials00 = f0[common_fiducials01[:,0]]
    fiducials01 = f1[common_fiducials01[:,1]]

    lockpoints00 = f0[common_lockpoints01[:,0]]
    lockpoints01 = f1[common_lockpoints01[:,1]]

    allpoints00 = np.concatenate([fiducials00, lockpoints00])
    allpoints01 = np.concatenate([fiducials01, lockpoints01])

    fiducials11 = f1[common_fiducials12[:,0]]
    fiducials12 = f2[common_fiducials12[:,1]]

    lockpoints11 = f1[common_lockpoints12[:,0]]
    lockpoints12 = f2[common_lockpoints12[:,1]]

    allpoints11 = np.concatenate([fiducials11, lockpoints11])
    allpoints12 = np.concatenate([fiducials12, lockpoints12])
    def rigid(points, params):
        theta, *d = params
        d = np.array(d)
    
        R = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    
        return np.einsum('ij,nj->ni', R, points) - d
    
    def cost_fn(points0, points1):
        def fn(inputs):
            return np.mean( (points1 - rigid(points0, inputs))**2 )
        return fn
    
    correction01 = optimize.minimize(cost_fn(allpoints00, allpoints01), (0,0,0)).x
    correction12 = optimize.minimize(cost_fn(allpoints12, allpoints11), (0,0,0)).x
    f0_corrected = rigid(f0, correction01)
    f2_corrected = rigid(f2, correction12)

    ax3.clear() #graphs and saves reconstructed stave core after stitching
    ax3.set_title("Reconstruction of Stave " + staveType + " after Stitching")
    ax3.set_xlabel("X position in mm")
    ax3.set_ylabel("Y position in mm")
    ax3.scatter(f0_corrected[:,0], f0_corrected[:,1], s=40, alpha=0.7)
    ax3.scatter(f1[:,0], f1[:,1], s=16, alpha=0.7)
    ax3.scatter(f2_corrected[:,0], f2_corrected[:,1], s=16, alpha=0.7)
    canvas3.draw()
    fig3.savefig(folder_selected + "\\Stave" +staveType +"_Test Results\\"+"Reconstruction After Stitching_"+ staveType+".png")
    

    FiducialsALL= np.concatenate([f0_corrected[near_fiducial_idxs0], f1[near_fiducial_idxs1], f2_corrected[near_fiducial_idxs2]], axis=0)

    weight = np.array([1, 1, 1, 1, 1, 0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5, 0.5, 1, 1, 1])
    a, b = np.polyfit(FiducialsALL[:,0], FiducialsALL[:,1], 1, w = weight)

    def ROTATE(points, angle):
        R = np.array([[np.cos(angle), -np.sin(angle)],[np.sin(angle), np.cos(angle)]])
        return np.einsum('ij,nj->ni', R, points)
    
    alpha = -np.arctan(a)

    f0_rotated = ROTATE(f0_corrected, alpha)


    offxD = f0_rotated[0,0]
    offyD = f0_rotated[0,1]

    LPidxs0 = np.array([15, 18, 21, 24])
    LPidxs1 = np.array([13, 16, 19, 22, 25, 28])
    LPidxs2 = np.array([13, 16, 19, 22, 25, 28])

    LpAll= np.concatenate([f0_corrected[LPidxs0], f1[LPidxs1], f2_corrected[LPidxs2]], axis=0)
    RotatedLpAll = ROTATE(LpAll, alpha)

    LpAll_re = RotatedLpAll.reshape(-1,2,2)
    print(LpAll_re)
    LpAll_ave = np.mean(LpAll_re, axis=1)
    LPweight = np.array([0.5, 1, 0.5, 0.5, 0.5, 1, 0.5, 0.5])
    e, f = np.polyfit(LpAll_ave[:,0], LpAll_ave[:,1], 1, w=LPweight)
    print(LpAll_ave)
    LpAll_ave_1 = LpAll_ave[0:2,:]
    LpAll_ave_2 = LpAll_ave[2:5,:]
    LpAll_ave_3 = LpAll_ave[5:,:]
    def graphRestore(): #graphs Lockpoints of Stave with Fit Line
        ax2.clear()
        ax2.scatter(LpAll_ave_1[:,0]-offxD, LpAll_ave_1[:,1]-offyD);
        ax2.scatter(LpAll_ave_2[:,0]-offxD, LpAll_ave_2[:,1]-offyD);
        ax2.scatter(LpAll_ave_3[:,0]-offxD, LpAll_ave_3[:,1]-offyD);
        ax2.set_ylabel("Y position in mm")
        ax2.set_xlabel("X position in mm")
        ax2.set_title("Lockpoint positions of Stave " + staveType)
        ax2.plot(LpAll_ave[:,0] - offxD, LpAll_ave[:,0]*e + f - offyD, "r-" )
        ax2.grid()
        canvas2.draw()
    graphRestore()
    fig2.savefig(folder_selected + "\\Stave" +staveType +"_Test Results\\"+"Locking Point Positions_"+ staveType+".png")

    #updating table with values of locking points in a nicer order
    tableLpx = np.array([LpAll_ave[1,0],LpAll_ave[0,0],LpAll_ave[4,0],LpAll_ave[3,0],LpAll_ave[7,0],LpAll_ave[2,0],LpAll_ave[6,0],LpAll_ave[5,0]])
    tableLpy = np.array([LpAll_ave[1,1],LpAll_ave[0,1],LpAll_ave[4,1],LpAll_ave[3,1],LpAll_ave[7,1],LpAll_ave[2,1],LpAll_ave[6,1],LpAll_ave[5,1]])
    for point in xLabels:
        point.config(text=f"{round(tableLpx[xLabels.index(point)]-offxD,5)}")
    for point in yLabels:
        point.config(text=f"{round(tableLpy[yLabels.index(point)]-offyD,5)}")

    def graphUpdate(Lp1,Lp2,Lp3,Lp123,weight): #updates Locking Point graph with updated Locking Point Arrays
        ax2.clear()
        ax2.scatter(Lp1[:,0]-offxD, Lp1[:,1]-offyD);
        ax2.scatter(Lp2[:,0]-offxD, Lp2[:,1]-offyD);
        ax2.scatter(Lp3[:,0]-offxD, Lp3[:,1]-offyD);
        ax2.plot(Lp123[:,0] - offxD, Lp123[:,0]*e + f - offyD, "r-" )
        m , b = np.polyfit(Lp123[:,0], Lp123[:,1], 1, w=weight)
        ax2.plot(Lp123[:,0] - offxD, Lp123[:,0]*m + b - offyD, "g-")
        ax2.set_ylabel("Y position in mm")
        ax2.set_xlabel("X position in mm")
        ax2.set_title("Lockpoint positions of Stave " + staveType)
        ax2.grid()
        canvas2.draw()
        passFail(Lp123[:,1] - Lp123[:,0]*e - f)
        resMax_lbl.config(text="Max Residual Value (in mm): " + f"{round(np.max(Lp123[:,1] - Lp123[:,0]*e - f),5)}")
        resMin_lbl.config(text="Min Residual Value (in mm): " + f"{round(np.min(Lp123[:,1] - Lp123[:,0]*e - f),5)}")

    def point0(): #command to remove/add point0 and create new fit line
        global isPoint0
        if (not isPoint1) or (not isPoint2) or (not isPoint3) or (not isPoint4) or (not isPoint5) or (not isPoint6) or (not isPoint7):
            btn0.config(text="Remove Point?")
        else:
            if isPoint0:
                btn0.config(text="Add Point?")            
                LpAll_ave_1_new = np.delete(LpAll_ave_1, 1, 0)
                LpAll_ave_new= np.delete(LpAll_ave, 1, 0)
                LPweight_new = np.array([0.5, 0.5, 0.5, 0.5, 1, 0.5, 0.5])
                graphUpdate(LpAll_ave_1_new,LpAll_ave_2, LpAll_ave_3, LpAll_ave_new, LPweight_new)
                isPoint0 = False
            else:
                btn0.config(text="Remove Point?")
                graphRestore()
                isPoint0 = True
    def point1(): #command to remove/add point1 and create fitline
        global isPoint1
        if (not isPoint0) or (not isPoint3) or (not isPoint4) or (not isPoint5) or (not isPoint6) or (not isPoint7):
            btn1.config(text="Remove Point?")
        else:
            if isPoint1:
                btn1.config(text="Add Point?")            
                LpAll_ave_1_new = np.delete(LpAll_ave_1, 0, 0)
                LpAll_ave_new= np.delete(LpAll_ave, 0, 0)
                if not isPoint2:
                    LpAll_ave_2_new = np.delete(LpAll_ave_2, 2, 0)
                    LpAll_ave_newer = np.delete(LpAll_ave_new, 3, 0)
                    LPweight_new = np.array([1, 0.5, 0.5, 1, 0.5, 0.5])
                    graphUpdate(LpAll_ave_1_new,LpAll_ave_2_new,LpAll_ave_3,LpAll_ave_newer,LPweight_new)
                else:
                    LPweight_new = np.array([1, 0.5, 0.5, 0.5, 1, 0.5, 0.5])
                    graphUpdate(LpAll_ave_1_new,LpAll_ave_2,LpAll_ave_3,LpAll_ave_new,LPweight_new)
                isPoint1 = False
            else:
                btn1.config(text="Remove Point?")
                if not isPoint2:
                    LpAll_ave_2_new = np.delete(LpAll_ave_2, 2, 0)
                    LpAll_ave_newish = np.delete(LpAll_ave, 4, 0)
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 1, 0.5, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2_new,LpAll_ave_3,LpAll_ave_newish,LPweight_new)
                else:
                    graphRestore()
                isPoint1 = True
    def point2(): #command to remove/add point2 and create fitline
        global isPoint2
        if (not isPoint0) or (not isPoint3) or (not isPoint4) or (not isPoint5) or (not isPoint6) or (not isPoint7):
            btn2.config(text="Remove Point?")
        else:
            if isPoint2:
                btn2.config(text="Add Point?")
                LpAll_ave_2_new = np.delete(LpAll_ave_2, 2, 0)
                LpAll_ave_new = np.delete(LpAll_ave, 4, 0)
                if not isPoint1:
                    LpAll_ave_1_new = np.delete(LpAll_ave_1, 0, 0)
                    LpAll_ave_newer = np.delete(LpAll_ave_new, 0, 0)
                    LPweight_new = np.array([1, 0.5, 0.5, 1, 0.5, 0.5])
                    graphUpdate(LpAll_ave_1_new,LpAll_ave_2_new,LpAll_ave_3,LpAll_ave_newer,LPweight_new)
                else:
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 1, 0.5, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2_new,LpAll_ave_3,LpAll_ave_new,LPweight_new)
                isPoint2 = False
            else:
                btn2.config(text="Remove Point?")
                if not isPoint1:
                    LpAll_ave_1_new = np.delete(LpAll_ave_1, 0, 0)
                    LpAll_ave_newish = np.delete(LpAll_ave, 0, 0)
                    LPweight_new = np.array([1, 0.5, 0.5, 0.5, 1, 0.5, 0.5])
                    graphUpdate(LpAll_ave_1_new,LpAll_ave_2,LpAll_ave_3,LpAll_ave_newish,LPweight_new)
                else:
                    graphRestore()
                isPoint2 = True
    def point3(): #command to remove/add point3 and create fitline
        global isPoint3
        if (not isPoint0) or (not isPoint1) or (not isPoint2) or (not isPoint5) or (not isPoint6) or (not isPoint7):
            btn3.configure(text="Remove Point?")
        else:
            if isPoint3:
                btn3.config(text="Add Point?")
                LpAll_ave_2_new = np.delete(LpAll_ave_2, 1, 0)
                LpAll_ave_new = np.delete(LpAll_ave, 3, 0)
                if not isPoint4:
                    LpAll_ave_3_new = np.delete(LpAll_ave_3, 2,0)
                    LpAll_ave_newer = np.delete(LpAll_ave_new, 6, 0)
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 1, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2_new,LpAll_ave_3_new,LpAll_ave_newer,LPweight_new)
                else:
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 1, 0.5, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2_new,LpAll_ave_3,LpAll_ave_new,LPweight_new)
                isPoint3 = False
            else:
                btn3.config(text="Remove Point?")
                if not isPoint4:
                    LpAll_ave_3_new = np.delete(LpAll_ave_3, 2,0)
                    LpAll_ave_newish = np.delete(LpAll_ave, 7, 0)
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 0.5, 1, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2,LpAll_ave_3_new,LpAll_ave_newish,LPweight_new)
                else:
                    graphRestore()
                isPoint3 = True
    def point4(): #command to remove/add point4 and create fitline
        global isPoint4
        if (not isPoint0) or (not isPoint1) or (not isPoint2) or (not isPoint5) or (not isPoint6) or (not isPoint7):
            btn4.config(text="Remove Point?")
        else:
            if isPoint4:
                btn4.config(text="Add Point?")
                LpAll_ave_3_new = np.delete(LpAll_ave_3, 2,0)
                LpAll_ave_new = np.delete(LpAll_ave, 7, 0)
                if not isPoint3:
                    LpAll_ave_2_new = np.delete(LpAll_ave_2, 1,0)
                    LpAll_ave_newer = np.delete(LpAll_ave_new, 3, 0)
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 1, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2_new,LpAll_ave_3_new,LpAll_ave_newer,LPweight_new)
                else:
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 0.5, 1, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2,LpAll_ave_3_new,LpAll_ave_new,LPweight_new)
                isPoint4 = False
            else:
                btn4.config(text="Remove Point?")
                if not isPoint3:
                    LpAll_ave_2_new = np.delete(LpAll_ave_2, 1,0)
                    LpAll_ave_newish = np.delete(LpAll_ave, 3, 0)
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 1, 0.5, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2_new,LpAll_ave_3,LpAll_ave_newish,LPweight_new)
                else:
                    graphRestore()
                isPoint4 = True
    def point5(): #command to remove/add point5 and create fitline
        global isPoint5
        if (not isPoint0) or (not isPoint1) or (not isPoint2) or (not isPoint3) or (not isPoint4) or (not isPoint7):
            btn5.config(text="Remove Point?")
        else:
            if isPoint5:
                btn5.config(text="Add Point?")
                LpAll_ave_2_new = np.delete(LpAll_ave_2, 0,0)
                LpAll_ave_new = np.delete(LpAll_ave, 2,0)
                if not isPoint6:
                    LpAll_ave_3_new = np.delete(LpAll_ave_3, 1, 0)
                    LpAll_ave_newer = np.delete(LpAll_ave_new, 5, 0)
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 1, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2_new,LpAll_ave_3_new,LpAll_ave_newer,LPweight_new)
                else:
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 1, 0.5, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2_new,LpAll_ave_3,LpAll_ave_new,LPweight_new)
                isPoint5 = False
            else:
                btn5.config(text="Remove Point?")
                if not isPoint6:
                    LpAll_ave_3_new = np.delete(LpAll_ave_3, 1, 0)
                    LpAll_ave_newish = np.delete(LpAll_ave, 6, 0)
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 0.5, 1, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2,LpAll_ave_3_new,LpAll_ave_newish,LPweight_new)
                else:
                    graphRestore()
                isPoint5 = True
    def point6(): #command to remove/add point6 and create fitline
        global isPoint6
        if (not isPoint0) or (not isPoint1) or (not isPoint2) or (not isPoint3) or (not isPoint4) or (not isPoint7):
            btn6.config(text="Remove Point?")
        else:
            ax2.clear()
            if isPoint6:
                btn6.config(text="Add Point?")
                LpAll_ave_3_new = np.delete(LpAll_ave_3, 1, 0)
                LpAll_ave_new = np.delete(LpAll_ave, 6, 0)
                if not isPoint5:
                    LpAll_ave_2_new = np.delete(LpAll_ave_2, 0,0)
                    LpAll_ave_newer = np.delete(LpAll_ave_new, 2, 0)
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 1, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2_new,LpAll_ave_3_new,LpAll_ave_newer,LPweight_new)
                else:
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 0.5, 1, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2,LpAll_ave_3_new,LpAll_ave_new,LPweight_new)
                isPoint6 = False
            else:
                btn6.config(text="Remove Point?")
                if not isPoint5:
                    LpAll_ave_2_new = np.delete(LpAll_ave_2, 0,0)
                    LpAll_ave_newish = np.delete(LpAll_ave, 2, 0)
                    LPweight_new = np.array([0.5, 1, 0.5, 0.5, 1, 0.5, 0.5])
                    graphUpdate(LpAll_ave_1,LpAll_ave_2_new,LpAll_ave_3,LpAll_ave_newish,LPweight_new)
                else:
                    graphRestore()
                isPoint6 = True
    def point7(): #command to remove/add point7 and create fitline
        global isPoint7
        if (not isPoint1) or (not isPoint2) or (not isPoint3) or (not isPoint4) or (not isPoint5) or (not isPoint6) or (not isPoint0):
            btn7.config(text="Remove Point?")
        else:
            if isPoint7:
                btn7.config(text="Add Point?")            
                LpAll_ave_3_new = np.delete(LpAll_ave_3, 0, 0)
                LpAll_ave_new= np.delete(LpAll_ave, 5, 0)
                LPweight_new = np.array([0.5, 1, 0.5, 0.5, 0.5, 0.5, 0.5])
                graphUpdate(LpAll_ave_1,LpAll_ave_2,LpAll_ave_3_new,LpAll_ave_new,LPweight_new)
                isPoint7 = False
            else:
                btn7.config(text="Remove Point?")
                graphRestore()
                isPoint7 = True
    
    btn0.config(command=point0)
    btn1.config(command=point1)
    btn2.config(command=point2)
    btn3.config(command=point3)
    btn4.config(command=point4)
    btn5.config(command=point5)
    btn6.config(command=point6)
    btn7.config(command=point7)
    
    #displays max/min residual values
    resMax_lbl.config(text="Max Residual Value (in mm): " + f"{round(np.max(LpAll_ave[:,1] - LpAll_ave[:,0]*e - f),5)}")
    resMin_lbl.config(text="Min Residual Value (in mm): " + f"{round(np.min(LpAll_ave[:,1] - LpAll_ave[:,0]*e - f),5)}")
    lpFidMax_lbl.config(text="Max distance from LP to Fiducial Regression (in mm): " + f"{round(RotatedLpAll[15,0]*e + f - offyD,5)}")
    lpFidMin_lbl.config(text="Min distance from LP to Fiducial Regression (in mm): " + f"{round(RotatedLpAll[0,0]*e + f - offyD,5)}")
    #graphs Residuals of Lockpoints
    ax.clear()
    ax.scatter(LpAll_ave_1[:,0]-offxD, LpAll_ave_1[:,1] - LpAll_ave_1[:,0]*e - f)
    ax.scatter(LpAll_ave_2[:,0]-offxD, LpAll_ave_2[:,1] - LpAll_ave_2[:,0]*e - f)
    ax.scatter(LpAll_ave_3[:,0]-offxD, LpAll_ave_3[:,1] - LpAll_ave_3[:,0]*e - f)
    ax.set_ylabel("Y position in mm")
    ax.set_xlabel("X position in mm")
    ax.set_title("Residuals in Lockpoint positions of Stave " + staveType)
    ax.axhline(y=0, color = "r", linestyle="-" )
    ax.grid()
    canvas.draw()
    fig.savefig(folder_selected + "\\Stave" +staveType +"_Test Results\\"+"Locking Point Residuals_"+ staveType+".png")

    passFail(LpAll_ave[:,1] - LpAll_ave[:,0]*e - f)
    #saves residuals, lockingpoints, and fit curves as a csv
    xColumn = tableLpx-offxD
    yColumn = tableLpy-offyD
    for i in range(8):
        xColumn[i]=round(xColumn[i],5)
        yColumn[i]=round(yColumn[i],5)
    fitList = np.array([(tableLpx[:]*e+f)-offyD]).flatten()
    fitList1 = np.round(fitList, 5)
    resList = np.array([tableLpy[:]-tableLpx[:]*e - f]).flatten()
    resList1 = np.round(resList, 5)
    resList2 = 1000*resList1
    er = round(e,9)
    fr = round(f,5)
    df = pd.DataFrame({"LP X (mm)": xColumn, "LP Y (mm)": yColumn, "Y Fit":fitList1, "Residuals": resList2, "Slope":er, "Intercept":fr})
    df.to_csv(folder_selected + "\\Stave" +staveType +"_Test Results\\"+"LockingPoints_"+ staveType+".csv", index=False)

    slope_lbl.config(text="Slope of Linear Fit of Locking Points: " + f"{er}")
    intercept_lbl.config(text="Intercept of Linear Fit of Locking Points: " + f"{fr}")

#last section is initializing GUI elements
btnFrame = tk.Frame(master=window)
btnFrame.pack()
run_btn = tk.Button(master=btnFrame, text="Run Analysis", command=runScript, font=18)
run_btn.grid(row=0,column=0)
test_lbl = tk.Label(master=btnFrame, text="Result", bg="yellow", font=18)
test_lbl.grid(row=0,column=1, padx=40, sticky="nsew")

graphMaster = tk.Frame(master=window)
graphMaster.pack()

graphFrame = tk.Frame(master=graphMaster)
graphFrame.grid(row=0,column=0,sticky="nw")

graphFrame2 = tk.Frame(master=graphMaster)
graphFrame2.grid(row=0,column=1,sticky="ne")


fig = matplotlib.figure.Figure(figsize=(6.5,4))
ax = fig.add_subplot()

canvas = FigureCanvasTkAgg(fig, master=graphFrame)
canvas.get_tk_widget().grid(row=0, column=0,sticky="w", padx=10)
toolbar = NavigationToolbar2Tk(canvas, graphFrame, pack_toolbar = False)
toolbar.update()
toolbar.grid(row=1,column=0, sticky="w")

fig2 = matplotlib.figure.Figure(figsize=(6.5,4))
ax2 = fig2.add_subplot()

canvas2 = FigureCanvasTkAgg(fig2, master=graphFrame)
canvas2.get_tk_widget().grid(row=2, column=0, padx=10)
toolbar2 = NavigationToolbar2Tk(canvas2, graphFrame, pack_toolbar = False)
toolbar2.update()
toolbar2.grid(row=3, column = 0, sticky="w")

fig3 = matplotlib.figure.Figure(figsize=(10,5))
ax3 = fig3.add_subplot()

canvas3 = FigureCanvasTkAgg(fig3, master=graphFrame2)
canvas3.get_tk_widget().grid(row=0, column=0, padx=10)
toolbar3 = NavigationToolbar2Tk(canvas3, graphFrame2, pack_toolbar = False)
toolbar3.update()
toolbar3.grid(row=1,column=0, sticky="w")

tableFrame = tk.Frame(master=graphFrame2)
tableFrame.grid(row=2, column=0,sticky="w")

x_lbl = tk.Label(master=tableFrame, text="X value in mm", font=18)
x_lbl.grid(row=0,column=0)

y_lbl = tk.Label(master=tableFrame, text="Y value in mm", font=18)
y_lbl.grid(row=0, column=1)

x0 = tk.Label(master=tableFrame, text=f"0", font= 18)  
x1 = tk.Label(master=tableFrame, text=f"0", font= 18)
x2 = tk.Label(master=tableFrame, text=f"0", font= 18)
x3 = tk.Label(master=tableFrame, text=f"0", font= 18)
x4 = tk.Label(master=tableFrame, text=f"0", font= 18)
x5 = tk.Label(master=tableFrame, text=f"0", font= 18)
x6 = tk.Label(master=tableFrame, text=f"0", font= 18)
x7 = tk.Label(master=tableFrame, text=f"0", font= 18)
xLabels = [x0,x1,x2,x3,x4,x5,x6,x7]
for point in xLabels:
    point.grid(row=xLabels.index(point)+1,column=0)

y0 = tk.Label(master=tableFrame, text=f"0", font= 18)
y1 = tk.Label(master=tableFrame, text=f"0", font= 18)
y2 = tk.Label(master=tableFrame, text=f"0", font= 18)
y3 = tk.Label(master=tableFrame, text=f"0", font= 18)
y4 = tk.Label(master=tableFrame, text=f"0", font= 18)
y5 = tk.Label(master=tableFrame, text=f"0", font= 18)
y6 = tk.Label(master=tableFrame, text=f"0", font= 18)
y7 = tk.Label(master=tableFrame, text=f"0", font= 18)
yLabels = [y0,y1,y2,y3,y4,y5,y6,y7]
for point in yLabels:
    point.grid(row=yLabels.index(point)+1,column=1)

btn0 = tk.Button(master=tableFrame, text="Remove Point?", bg="#F55447")
btn1 = tk.Button(master=tableFrame, text="Remove Point?", bg="orange")
btn2 = tk.Button(master=tableFrame, text="Remove Point?", bg="orange")
btn3 = tk.Button(master=tableFrame, text="Remove Point?", bg="yellow")
btn4 = tk.Button(master=tableFrame, text="Remove Point?", bg="yellow")
btn5 = tk.Button(master=tableFrame, text="Remove Point?", bg="#50F97C")
btn6 = tk.Button(master=tableFrame, text="Remove Point?", bg="#50F97C")
btn7 = tk.Button(master=tableFrame, text="Remove Point?", bg="#7ED2F6")
btnArray = [btn0,btn1,btn2,btn3,btn4,btn5,btn6,btn7]
for btn in btnArray:
    btn.grid(row=btnArray.index(btn)+1,column=2)

resMax_lbl = tk.Label(master=tableFrame, text="Max Residual Value (in mm): ",font=18)
resMax_lbl.grid(row=1, column=3, padx=40)
resMin_lbl = tk.Label(master=tableFrame, text="Min Residual Value (in mm): ",font=18)
resMin_lbl.grid(row=2, column=3, padx=40)
lpFidMax_lbl = tk.Label(master=tableFrame, text="Max distance from LP to Fiducial Regression (in mm): ", font=18)
lpFidMax_lbl.grid(row=3, column=3, padx=40)
lpFidMin_lbl = tk.Label(master=tableFrame, text="Min distance from LP to Fiducial Regression (in mm): ", font=18)
lpFidMin_lbl.grid(row=4, column=3, padx=40)
slope_lbl = tk.Label(master=tableFrame, text="Slope of Linear Fit of Locking Points: ", font = 18)
slope_lbl.grid(row=5, column= 3, padx=40)
intercept_lbl = tk.Label(master=tableFrame, text="Intercept of Linear Fit of Locking Points: ", font=18)
intercept_lbl.grid(row=6, column=3, padx=40)

window.mainloop()