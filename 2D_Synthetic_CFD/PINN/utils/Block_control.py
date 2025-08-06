import math 
import numpy as np 
import torch 
from torch import nn 

def weight_calc2(Input,Index,line,dist):
    x = Input[Index,0]
    y = Input[Index,1]
    a,b,c = line[0], line[1] ,line[2]
    r = math.sqrt(a**2 + b**2)
    
    numerator_1 = torch.abs(a * x + b * y + c + r * dist)
    numerator_2 = torch.abs(a * x + b * y + c - r * dist)
    
    distance_1 = numerator_1 / r 
    distance_2 = numerator_2 / r 
        
    Sum = distance_1 + distance_2
    weight1 = 1 - distance_1/ Sum
    weight2 = 1 - distance_2/ Sum
    weight = torch.concat((weight1.reshape(-1,1),weight2.reshape(-1,1)),dim=-1)

    return weight

def Model_Decom_OverlapY(Input_data,line_bank,N_model,dist):
    Model_bank = []
    for i in range(len(Input_data)):
        x = Input_data[i,0]
        y = Input_data[i,1]
        Index = np.zeros(N_model)
        # Decomposition 
        line1 = line_bank[0]
        Dec1 = x * line1[0] + y * line1[1] + line1[2]
        r1 = math.sqrt(line1[0]**2 + line1[1]**2)
        line2 = line_bank[1]
        Dec2 = x * line2[0] + y * line2[1] + line2[2]
        r2 = math.sqrt(line2[0]**2 + line2[1]**2)
        
        if Dec1 -  r1 * dist < 0:
            Index[0] = 1
        if Dec2 -  r2 * dist < 0 and Dec1 + r1 * dist > 0:
            Index[1] = 1
        if Dec2 + r2 * dist > 0:
            Index[2] = 1
        
        Model_bank.append(Index)
    return np.array(Model_bank)

def Model_Decom_Overlap(Input_data,line_bank,N_model,dist):
    Model_bank = []
    for i in range(len(Input_data)):
        x = Input_data[i,0]
        y = Input_data[i,1]
        Index = np.zeros(N_model)
        # Decomposition 
        for i in range(len(line_bank)):
            line = line_bank[i]
            Dec = x * line[0] + y * line[1] + line[2]
            r = math.sqrt(line[0]**2 + line[1]**2)
            
            if Dec -  r * dist < 0:
                Index[0] = 1
            if Dec + r * dist >0:
                Index[1] = 1
        
        Model_bank.append(Index)
    return np.array(Model_bank)

def Model_Decom_Wo_overlap(Input_data,line_bank,N_model,dist):
    Model_bank = []
    for i in range(len(Input_data)):
        x = Input_data[i,0]
        y = Input_data[i,1]
        Index = np.zeros(N_model)
        # Decomposition 
        for i in range(len(line_bank)):
            line = line_bank[i]
            Dec = x * line[0] + y * line[1] + line[2]
            if Dec < 0:
                Index[0] = 1
            if Dec >= 0:
                Index[1] = 1
        
        Model_bank.append(Index)
    return np.array(Model_bank)

def Interface_Area(input,line_bank,distance):
    x = input[:,0]
    y = input[:,1]
    Interface_area = []
    for line in line_bank:
        r = math.sqrt(line[0]**2 + line[1]**2)
        Dec = x * line[0] + y * line[1] + line[2]
        a = Dec - r * distance
        b = Dec + r * distance
        Interface_area.append(input[a*b<0])
    
    return Interface_area

def Decom_Area_Block(input,line_bank,dist):
    x = input[:,0]
    y = input[:,1]
    for line in line_bank:
        Dec = x * line[0] + y * line[1] + line[2]
        r = math.sqrt(line[0]**2 + line[1]**2)
    
    return input[Dec +  r * dist<0],input[Dec -  r * dist >=0]

def Decom_Area_Wall(input,line_bank,dist):
    x = input[:,0]
    y = input[:,1]
    for line in line_bank:
        Dec = x * line[0] + y * line[1] + line[2]
        r = math.sqrt(line[0]**2 + line[1]**2)
    
    return input[Dec -  r * dist<0],input[Dec +  r * dist >=0]

def Decom_Area(input,line_bank):
    x = input[:,0]
    y = input[:,1]
    for line in line_bank:
        Dec = x * line[0] + y * line[1] + line[2]
    
    return input[Dec<0],input[Dec>=0]    

def weight_calc(input,line,dist):
    weight1_total = []
    weight2_total = []
    a,b,c = line[0], line[1] ,line[2]
    
    r = math.sqrt(a**2 + b**2)
    for i in range(len(input)):
            x = input[i,0]
            y = input[i,1]
            numerator_1 = torch.abs(a * x + b * y + c + r * dist)
            numerator_2 = torch.abs(a * x + b * y + c - r * dist)
            
            dominator = np.sqrt(a ** 2.0 + b ** 2.0)
            
            distance_1 = numerator_1 / dominator 
            distance_2 = numerator_2 / dominator 
            
            Sum = distance_1 + distance_2
            weight1 = 1 - distance_1/ Sum
            weight2 = 1 - distance_2/ Sum
            
            weight1_total.append(weight1)
            weight2_total.append(weight2)
    
    total_weight1 = torch.tensor(weight1_total).to('cuda')
    total_weight2 = torch.tensor(weight2_total).to('cuda')
    
    return total_weight1.reshape(-1,1),total_weight2.reshape(-1,1)
     

