import numpy as np
import torch
import random
import math

from torch.autograd import Variable

device = 'cuda'

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

def block_filtering(fluid,fluid_i):
    fluid_indepent = np.array([row for row in fluid if not np.isin(row, fluid_i).all()])
    
    return fluid_indepent

def block_decomposition_fluid(fluid,line):
    def calc_area(point,line):
        out = line[0] * point[0] + line[1] * point[1] + line[2] 
        
        if out * line[3] > 0:
            return True
        else:
            return False
    
    def decompose(point,line):
        for i in range(len(line)):
            if calc_area(point,line[i]):
                continue
            else:
                return False
        
        return True
        
    fluid_filtered = []

    for i in range (len(fluid)):
        if decompose(fluid[i],line) :
            fluid_filtered.append(fluid[i])
    
    fluid_decom = np.array(fluid_filtered)  
    
    return fluid_decom

def block_decomposition_wall(wall,n,line):
    def calc_area(point,line):
        out = line[0] * point[0] + line[1] * point[1] + line[2] 
        
        if out * line[3] > 0:
            return True
        else:
            return False
    
    def decompose(point,line):
        for i in range(len(line)):
            if calc_area(point,line[i]):
                continue
            else:
                return False
        
        return True
        
    wall_filtered = []
    n_filtered = []
    
    for i in range (len(wall)):
        if decompose(wall[i],line) :
            wall_filtered.append(wall[i])
            n_filtered.append(n[i])
            
    wall_decom = np.array(wall_filtered)
    n_decom = np.array(n_filtered)        
    
    return wall_decom, n_decom

def block_interface_wall_Y(wall,n,line1,line2,line3,dist):
    def interface_line(line,dist):
        a, b, c = line[0], line[1] , line[2]
        if a == 0 or b == 0:
            line1_L = np.array([a,b,c-dist,1]).reshape(-1,4)
            line1_R = np.array([a,b,c-dist,-1]).reshape(-1,4)
            line_L = np.array([a,b,c,1]).reshape(-1,4)
            line_R = np.array([a,b,c,-1]).reshape(-1,4)
            return line1_L,line1_R,line_L,line_R
        else:
            r = math.sqrt(a**2 + b**2)
            line1_L = np.array([a,b,c-(dist * r),1]).reshape(-1,4)
            line1_R = np.array([a,b,c-(dist * r),-1]).reshape(-1,4)
            line_L = np.array([a,b,c,1]).reshape(-1,4)
            line_R = np.array([a,b,c,-1]).reshape(-1,4)
            return line1_L,line1_R,line_L,line_R
    
    line1_L,line1_R,line_1_L,line_1_R = interface_line(line1,dist)
    line2_L,line2_R,line_2_L,line_2_R = interface_line(line2,dist)
    line3_L,line3_R,line_3_L,line_3_R = interface_line(line3,dist)
    line_U = np.array([0,1,-0.3300,1]).reshape(-1,4)
    line_D = np.array([0,1,-0.3300,-1]).reshape(-1,4)
    
    line_bank1 = line_1_L
    line_bank2 = np.concatenate((line1_R,line2_R,line3_R),axis=0)
    line_bank3 = np.concatenate((line_2_L,line_U),axis=0)
    line_bank4 = np.concatenate((line_3_L,line_D),axis=0)
    line_bank12 = np.concatenate((line1_R,line_1_L),axis=0)
    line_bank23 = np.concatenate((line2_R,line_2_L,line_U),axis=0)
    line_bank24 = np.concatenate((line3_R,line_3_L,line_D),axis=0)
    
    wall_1,n_1 = block_decomposition_wall(wall,n,line_bank1)
    wall_2,n_2 = block_decomposition_wall(wall,n,line_bank2)
    wall_3,n_3 = block_decomposition_wall(wall,n,line_bank3)
    wall_4,n_4 = block_decomposition_wall(wall,n,line_bank4)
    
    wall_12,n_12 = block_decomposition_wall(wall,n,line_bank12)
    wall_23,n_23 = block_decomposition_wall(wall,n,line_bank23)
    wall_24,n_24 = block_decomposition_wall(wall,n,line_bank24)
    
    w12_1 , w12_2 = weight_pre(wall_12,line1_R.squeeze(0),line_1_L.squeeze(0))
    w23_2 , w23_3 = weight_pre(wall_23,line_2_L.squeeze(0),line2_R.squeeze(0))
    w24_2 , w24_4 = weight_pre(wall_24,line_3_L.squeeze(0),line3_R.squeeze(0))
    
    return wall_1,n_1,wall_2,n_2,wall_3,n_3,wall_4,n_4,\
        wall_12,n_12,w12_1,w12_2,wall_23,n_23,w23_2,w23_3,wall_24,n_24,w24_2,w24_4

def block_decomposition_fluid_Y(fluid,line1,line2,line3,dist):
    def interface_line(line,dist):
        a, b, c = line[0], line[1] , line[2]
        if a == 0 or b == 0:
            line1_L = np.array([a,b,c-dist,1]).reshape(-1,4)
            line1_R = np.array([a,b,c-dist,-1]).reshape(-1,4)
            line_L = np.array([a,b,c,1]).reshape(-1,4)
            line_R = np.array([a,b,c,-1]).reshape(-1,4)
            return line1_L,line1_R,line_L,line_R
        else:
            r = math.sqrt(a**2 + b**2)
            line1_L = np.array([a,b,c-(dist * r),1]).reshape(-1,4)
            line1_R = np.array([a,b,c-(dist * r),-1]).reshape(-1,4)
            line_L = np.array([a,b,c,1]).reshape(-1,4)
            line_R = np.array([a,b,c,-1]).reshape(-1,4)
            return line1_L,line1_R,line_L,line_R
        
    line1_L,line1_R,line_1_L,line_1_R = interface_line(line1,dist)
    line2_L,line2_R,line_2_L,line_2_R = interface_line(line2,dist)
    line3_L,line3_R,line_3_L,line_3_R = interface_line(line3,dist)
    line_U = np.array([0,1,-0.3300,1]).reshape(-1,4)
    line_D = np.array([0,1,-0.3300,-1]).reshape(-1,4)
    
    line_bank1 = line1_L
    line_bank2 = np.concatenate((line_1_R,line_2_R,line_3_R),axis=0)
    line_bank3 = np.concatenate((line2_L,line_U),axis=0)
    line_bank4 = np.concatenate((line3_L,line_D),axis=0)
    line_bank12 = np.concatenate((line1_R,line_1_L),axis=0)
    line_bank23 = np.concatenate((line2_R,line_2_L,line_U),axis=0)
    line_bank24 = np.concatenate((line3_R,line_3_L,line_D),axis=0)
    
    fluid_1 = block_decomposition_fluid(fluid,line_bank1)
    fluid_2 = block_decomposition_fluid(fluid,line_bank2)
    fluid_3 = block_decomposition_fluid(fluid,line_bank3)
    fluid_4 = block_decomposition_fluid(fluid,line_bank4)

    fluid_12 = block_decomposition_fluid(fluid,line_bank12)
    fluid_23 = block_decomposition_fluid(fluid,line_bank23)
    fluid_24 = block_decomposition_fluid(fluid,line_bank24)
    
    w12_1 , w12_2 = weight_pre(fluid_12,line1_R.squeeze(0),line_1_L.squeeze(0))
    w23_2 , w23_3 = weight_pre(fluid_23,line_2_L.squeeze(0),line2_R.squeeze(0))
    w24_2 , w24_4 = weight_pre(fluid_24,line_3_L.squeeze(0),line3_R.squeeze(0))
    
    return fluid_1,fluid_2,fluid_3,fluid_4,\
        fluid_12,w12_1,w12_2,\
        fluid_23,w23_2,w23_3,\
        fluid_24,w24_2,w24_4

def block_decomposition_fluid_L(fluid,line1,line2,dist):
    def interface_line(line,dist):
        a, b, c = line[0], line[1] , line[2]
        if a == 0 or b == 0:
            line1_L = np.array([a,b,c-dist,-1]).reshape(-1,4)
            line1_R = np.array([a,b,c-dist,1]).reshape(-1,4)
            line_L = np.array([a,b,c+dist,-1]).reshape(-1,4)
            line_R = np.array([a,b,c+dist,1]).reshape(-1,4)
            return line1_L,line1_R,line_L,line_R
        else:
            r = math.sqrt(a**2 + b**2)
            line1_L = np.array([a,b,c-(dist * r),-1]).reshape(-1,4)
            line1_R = np.array([a,b,c-(dist * r),1]).reshape(-1,4)
            line_L = np.array([a,b,c+(dist * r),-1]).reshape(-1,4)
            line_R = np.array([a,b,c+(dist * r),1]).reshape(-1,4)
            return line1_L,line1_R,line_L,line_R
        
    line1_L,line1_R,line_1_L,line_1_R = interface_line(line1,dist)
    line2_D,line2_U,line_2_D,line_2_U = interface_line(line2,dist)
    
    line_bank1 = line_1_L
    line_bank2 = np.concatenate((line1_R,line_2_D),axis=0)
    line_bank3 = line2_U
    
    line_bank12 = np.concatenate((line1_L,line_1_R),axis=0)
    line_bank23 = np.concatenate((line2_D,line_2_U),axis=0)
    
    fluid_1 = block_decomposition_fluid(fluid,line_bank1)
    fluid_2 = block_decomposition_fluid(fluid,line_bank2)
    fluid_3 = block_decomposition_fluid(fluid,line_bank3)

    fluid_12 = block_decomposition_fluid(fluid,line_bank12)
    fluid_23 = block_decomposition_fluid(fluid,line_bank23)
    
    w12_1 , w12_2 = weight_pre(fluid_12,line_1_L.squeeze(0),line1_R.squeeze(0))
    w23_2 , w23_3 = weight_pre(fluid_23,line_2_U.squeeze(0),line2_D.squeeze(0))
    
    return fluid_1,fluid_2,fluid_3,\
        fluid_12,w12_1,w12_2,\
        fluid_23,w23_2,w23_3

def block_interface_wall_L(wall,n,line1,line2,dist):
    def interface_line(line,dist):
        a, b, c = line[0], line[1] , line[2]
        if a == 0 or b == 0:
            line1_L = np.array([a,b,c-dist,-1]).reshape(-1,4)
            line1_R = np.array([a,b,c-dist,1]).reshape(-1,4)
            line_L = np.array([a,b,c+dist,-1]).reshape(-1,4)
            line_R = np.array([a,b,c+dist,1]).reshape(-1,4)
            return line1_L,line1_R,line_L,line_R
        else:
            r = math.sqrt(a**2 + b**2)
            line1_L = np.array([a,b,c-(dist * r),-1]).reshape(-1,4)
            line1_R = np.array([a,b,c-(dist * r),1]).reshape(-1,4)
            line_L = np.array([a,b,c+(dist * r),-1]).reshape(-1,4)
            line_R = np.array([a,b,c+(dist * r),1]).reshape(-1,4)
            return line1_L,line1_R,line_L,line_R
    
    line1_L,line1_R,line_1_L,line_1_R = interface_line(line1,dist)
    line2_D,line2_U,line_2_D,line_2_U = interface_line(line2,dist)
    
    line_bank1 = line1_L
    line_bank2 = np.concatenate((line_1_R,line2_D),axis=0)
    line_bank3 = line_2_U
    
    line_bank12 = np.concatenate((line1_L,line_1_R),axis=0)
    line_bank23 = np.concatenate((line2_D,line_2_U),axis=0)
    
    wall_1,n_1 = block_decomposition_wall(wall,n,line_bank1)
    wall_2,n_2 = block_decomposition_wall(wall,n,line_bank2)
    wall_3,n_3 = block_decomposition_wall(wall,n,line_bank3)
    
    wall_12,n_12 = block_decomposition_wall(wall,n,line_bank12)
    wall_23,n_23 = block_decomposition_wall(wall,n,line_bank23)
    
    w12_1 , w12_2 = weight_pre(wall_12,line_1_L.squeeze(0),line1_R.squeeze(0))
    w23_2 , w23_3 = weight_pre(wall_23,line_2_U.squeeze(0),line2_D.squeeze(0))
   
    return wall_1,n_1,wall_2,n_2,wall_3,n_3,\
        wall_12,n_12,w12_1,w12_2,wall_23,n_23,w23_2,w23_3
                  
def block_interface_wall_2(wall,n,line,dist):
    def interface_line(line,dist):
        a, b, c = line[0], line[1] , line[2]
        if a == 0 or b == 0:
            line1 = np.array([a,b,c-dist,-1]).reshape(-1,4)
            line2 = np.array([a,b,c+dist,1]).reshape(-1,4)
            return line1, line2
        else:
            r = math.sqrt(a**2 + b**2)
            line1 = np.array([a,b,c-(dist * r),-1]).reshape(-1,4)
            line2 = np.array([a,b,c+(dist * r),1]).reshape(-1,4)
            
            return line1, line2
          
    def weight_pre(pt,line1,line2):
        weight1_total = []
        weight2_total = []
        
        a1,b1,c1 = line1[0], line1[1] ,line1[2] 
        a2,b2,c2 = line2[0], line2[1] ,line2[2] 

        for i in range(len(pt)):
            x = pt[i,0]
            y = pt[i,1]
            numerator_1 = np.abs(a1 * x + b1 * y + c1)
            numerator_2 = np.abs(a2 * x + b2 * y + c2)
            
            dominator_1 = np.sqrt(a1 ** 2.0 + b1 ** 2.0)
            dominator_2 = np.sqrt(a2 ** 2.0 + b2 ** 2.0)
            
            distance_1 = numerator_1 / dominator_1 
            distance_2 = numerator_2 / dominator_2 
            
            Sum = distance_1 + distance_2
            weight1 = 1 - distance_1/ Sum
            weight2 = 1 - distance_2/ Sum
            
            weight1_total.append(weight1)
            weight2_total.append(weight2)
        
        total_weight1 = torch.from_numpy(np.array(weight1_total).reshape(-1,1)).float().to('cuda')
        total_weight2 = torch.from_numpy(np.array(weight2_total).reshape(-1,1)).float().to('cuda')
        
        return total_weight1,total_weight2
    
    line1, line2 =  interface_line(line,dist)
    line = np.concatenate((line1,line2),axis=0)
    
    wall_inter,n_inter = block_decomposition_wall(wall,n,line) 
    weight1_w , weight2_w = weight_pre(wall_inter,line1.squeeze(0),line2.squeeze(0))
    
    return wall_inter,n_inter , weight1_w , weight2_w

def block_interface_area_2(fluid,wall,n,line,dist):
    def interface_line(line,dist):
        a, b, c = line[0], line[1] , line[2]
        if a == 0 or b == 0:
            line1 = np.array([a,b,c-dist,1]).reshape(-1,4)
            return line1
        else:
            r = math.sqrt(a**2 + b**2)
            line1 = np.array([a,b,c-(dist * r),1]).reshape(-1,4)
            return line1
          
    def weight_pre(pt,line1,line2):
        weight1_total = []
        weight2_total = []
        
        a1,b1,c1 = line1[0], line1[1] ,line1[2] 
        a2,b2,c2 = line2[0], line2[1] ,line2[2] 

        for i in range(len(pt)):
            x = pt[i,0]
            y = pt[i,1]
            numerator_1 = np.abs(a1 * x + b1 * y + c1)
            numerator_2 = np.abs(a2 * x + b2 * y + c2)
            
            dominator_1 = np.sqrt(a1 ** 2.0 + b1 ** 2.0)
            dominator_2 = np.sqrt(a2 ** 2.0 + b2 ** 2.0)
            
            distance_1 = numerator_1 / dominator_1 
            distance_2 = numerator_2 / dominator_2 
            
            Sum = distance_1 + distance_2
            weight1 = 1 - distance_1/ Sum
            weight2 = 1 - distance_2/ Sum
            
            weight1_total.append(weight1)
            weight2_total.append(weight2)
        
        total_weight1 = torch.from_numpy(np.array(weight1_total).reshape(-1,1)).float().to('cuda')
        total_weight2 = torch.from_numpy(np.array(weight2_total).reshape(-1,1)).float().to('cuda')
        
        return total_weight1,total_weight2
    
    line1, line2 =  interface_line(line,dist)
    line = np.concatenate((line1,line2),axis=0)
    
    fluid_inter = block_decomposition_fluid(fluid,line)
    wall_inter,n_inter = block_decomposition_wall(wall,n,line)
    weight1 , weight2 = weight_pre(fluid_inter,line1.squeeze(0),line2.squeeze(0))
    weight1_w , weight2_w = weight_pre(wall_inter,line1.squeeze(0),line2.squeeze(0))
    
    return fluid_inter,wall_inter,n_inter,weight1 , weight2 , weight1_w , weight2_w

def block_interface_wall_4(wall,n,line_ver,line_hor,dist):
    def interface_line(line,dist):
        a, b, c = line[0], line[1] , line[2]
        if a == 0 or b == 0:
            line1_L = np.array([a,b,c+dist,-1]).reshape(-1,4)
            line1_R = np.array([a,b,c+dist,1]).reshape(-1,4)
            line2_L = np.array([a,b,c-dist,-1]).reshape(-1,4)
            line2_R = np.array([a,b,c-dist,1]).reshape(-1,4)
            return line1_L,line1_R,line2_L,line2_R 
        else:
            r = math.sqrt(a**2 + b**2)
            line1_L = np.array([a,b,c+(dist * r),-1]).reshape(-1,4)
            line1_R = np.array([a,b,c+(dist * r),1]).reshape(-1,4)
            line2_L = np.array([a,b,c-(dist * r),-1]).reshape(-1,4)
            line2_R = np.array([a,b,c-(dist * r),1]).reshape(-1,4)
            
            return line1_L,line1_R,line2_L,line2_R 
          
    line1_L,line1_R,line2_L,line2_R  =  interface_line(line_ver,dist)
    line3_D,line3_U,line4_D,line4_U  =  interface_line(line_hor,dist)
    
    line_L = np.array([1,0,-0.5,-1]).reshape(-1,4)
    line_R = np.array([1,0,-0.5,1]).reshape(-1,4)
    line_U = np.array([0,1,-0.5,1]).reshape(-1,4)
    line_D = np.array([0,1,-0.5,-1]).reshape(-1,4)
    
    line_bank1 = np.concatenate((line_L,line_D),axis=0)
    line_bank2 = np.concatenate((line_R,line_D),axis=0)
    line_bank3 = np.concatenate((line_R,line_U),axis=0)
    line_bank4 = np.concatenate((line_L,line_U),axis=0)
    
    line_bank12 = np.concatenate((line1_R,line2_L,line3_D),axis=0)
    line_bank23 = np.concatenate((line4_D,line3_U,line2_R),axis=0)
    line_bank34 = np.concatenate((line1_R,line2_L,line4_U),axis=0)
    line_bank14 = np.concatenate((line4_D,line3_U,line1_L),axis=0)
    
    wall_1,n_1 = block_decomposition_wall(wall,n,line_bank1)
    wall_2,n_2 = block_decomposition_wall(wall,n,line_bank2)
    wall_3,n_3 = block_decomposition_wall(wall,n,line_bank3)
    wall_4,n_4 = block_decomposition_wall(wall,n,line_bank4)
    
    wall_12,n_12 = block_decomposition_wall(wall,n,line_bank12)
    wall_23,n_23 = block_decomposition_wall(wall,n,line_bank23)
    wall_34,n_34 = block_decomposition_wall(wall,n,line_bank34)
    wall_14,n_14 = block_decomposition_wall(wall,n,line_bank14)
    
    w12_1 , w12_2 = weight_pre(wall_12,line1_R.squeeze(0),line2_R.squeeze(0))
    w23_2 , w23_3 = weight_pre(wall_23,line3_U.squeeze(0),line4_D.squeeze(0))
    w34_3 , w34_4 = weight_pre(wall_34,line2_R.squeeze(0),line1_R.squeeze(0))
    w14_1 , w14_4 = weight_pre(wall_14,line3_U.squeeze(0),line4_D.squeeze(0))
    
    return wall_1,n_1,wall_2,n_2,wall_3,n_3,wall_4,n_4,\
        wall_12,n_12,w12_1,w12_2,wall_23,n_23,w23_2,w23_3,wall_34,n_34,w34_3,w34_4,wall_14,n_14,w14_1,w14_4

def block_interface_area_4(fluid,line_ver,line_hor,dist):
    def interface_line(line,dist):
        a, b, c = line[0], line[1] , line[2]
        if a == 0 or b == 0:
            line1_L = np.array([a,b,c+dist,-1]).reshape(-1,4)
            line1_R = np.array([a,b,c+dist,1]).reshape(-1,4)
            line2_L = np.array([a,b,c-dist,-1]).reshape(-1,4)
            line2_R = np.array([a,b,c-dist,1]).reshape(-1,4)
            return line1_L,line1_R,line2_L,line2_R 
        else:
            r = math.sqrt(a**2 + b**2)
            line1_L = np.array([a,b,c+(dist * r),-1]).reshape(-1,4)
            line1_R = np.array([a,b,c+(dist * r),1]).reshape(-1,4)
            line2_L = np.array([a,b,c-(dist * r),-1]).reshape(-1,4)
            line2_R = np.array([a,b,c-(dist * r),1]).reshape(-1,4)
            
            return line1_L,line1_R,line2_L,line2_R 
          
    line1_L,line1_R,line2_L,line2_R  =  interface_line(line_ver,dist)
    line3_D,line3_U,line4_D,line4_U  =  interface_line(line_hor,dist)
    
    
    line_bank1 = np.concatenate((line1_L,line3_D),axis=0)
    line_bank2 = np.concatenate((line2_R,line3_D),axis=0)
    line_bank3 = np.concatenate((line2_R,line4_U),axis=0)
    line_bank4 = np.concatenate((line1_L,line4_U),axis=0)
    
    line_bank12 = np.concatenate((line1_R,line2_L,line3_D),axis=0)
    line_bank23 = np.concatenate((line4_D,line3_U,line2_R),axis=0)
    line_bank34 = np.concatenate((line1_R,line2_L,line4_U),axis=0)
    line_bank14 = np.concatenate((line4_D,line3_U,line1_L),axis=0)
    
    line_bank1234 = np.concatenate((line4_D,line3_U,line1_R,line2_L),axis=0)
    
    fluid_1 = block_decomposition_fluid(fluid,line_bank1)
    fluid_2 = block_decomposition_fluid(fluid,line_bank2)
    fluid_3 = block_decomposition_fluid(fluid,line_bank3)
    fluid_4 = block_decomposition_fluid(fluid,line_bank4)
    
    
    fluid_12 = block_decomposition_fluid(fluid,line_bank12)
    fluid_23 = block_decomposition_fluid(fluid,line_bank23)
    fluid_34 = block_decomposition_fluid(fluid,line_bank34)
    fluid_14 = block_decomposition_fluid(fluid,line_bank14)
    
    fluid_1234 = block_decomposition_fluid(fluid,line_bank1234)
    
    w12_1 , w12_2 = weight_pre(fluid_12,line1_R.squeeze(0),line2_R.squeeze(0))
    w23_2 , w23_3 = weight_pre(fluid_23,line3_U.squeeze(0),line4_D.squeeze(0))
    w34_3 , w34_4 = weight_pre(fluid_34,line2_R.squeeze(0),line1_R.squeeze(0))
    w14_1 , w14_4 = weight_pre(fluid_14,line3_U.squeeze(0),line4_D.squeeze(0))
    
    weighta , weightc = weight_pre(fluid_1234,line1_R.squeeze(0),line2_L.squeeze(0))
    weightb , weightd = weight_pre(fluid_1234,line3_D.squeeze(0),line4_D.squeeze(0))
    
    w1234_1 = weighta * weightb 
    w1234_2 = weightc * weightb
    w1234_3 = weightc * weightd
    w1234_4 = weighta * weightd

    return fluid_1,fluid_2,fluid_3,fluid_4,\
        fluid_12,w12_1,w12_2,\
        fluid_23,w23_2,w23_3,\
        fluid_34,w34_3,w34_4,\
        fluid_14,w14_1,w14_4,\
        fluid_1234,w1234_1,w1234_2,w1234_3,w1234_4

def interpolation_train(x,y,model1,model2,weight1,weight2,Re):
    
    pred1 = model1(x,y)
    pred2 = model2(x,y)
    
    pred_inter=((weight1 * pred1 + weight2 * pred2)/ (weight1 + weight2))
    
    u = pred_inter[:,0].reshape(-1,1)
    v = pred_inter[:,1].reshape(-1,1)
    p = pred_inter[:,2].reshape(-1,1)
    
    u_x = torch.autograd.grad(u.reshape(-1,1).sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.reshape(-1,1).sum(), y, create_graph=True)[0]
    
    v_x = torch.autograd.grad(v.reshape(-1,1).sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.reshape(-1,1).sum(), y, create_graph=True)[0]
    
    p_x = torch.autograd.grad(p.reshape(-1,1).sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.reshape(-1,1).sum(), y, create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones(y.shape).to(device), create_graph=True)[0]
    
    v_xx = torch.autograd.grad(v_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones(y.shape).to(device), create_graph=True)[0]
    
    Eu = u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy)
    Ev = u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy)
    Ec = u_x + v_y
    
    
    return pred_inter, Eu,Ev,Ec, p_x,p_y

def interpolation_infer(x,y,model1,model2,weight1,weight2):
    
    pred1 = model1(x,y)
    pred2 = model2(x,y)
    pred_inter=((weight1 * pred1 + weight2 * pred2)/ (weight1 + weight2))
    
    return pred_inter

def interpolation_train_4point(x,y,model1,model2,model3,model4,weight1,weight2,weight3,weight4,Re):
    
    pred1 = model1(x,y)
    pred2 = model2(x,y)
    pred3 = model3(x,y)
    pred4 = model4(x,y)
    pred_inter = (weight1 * pred1 + weight2 * pred2 + weight3 *  pred3 + weight4 * pred4) / (weight1+weight2+weight3+weight4)
    
    u = pred_inter[:,0].reshape(-1,1)
    v = pred_inter[:,1].reshape(-1,1)
    p = pred_inter[:,2].reshape(-1,1)
    
    u_x = torch.autograd.grad(u.reshape(-1,1).sum(), x, create_graph=True)[0]
    u_y = torch.autograd.grad(u.reshape(-1,1).sum(), y, create_graph=True)[0]
    
    v_x = torch.autograd.grad(v.reshape(-1,1).sum(), x, create_graph=True)[0]
    v_y = torch.autograd.grad(v.reshape(-1,1).sum(), y, create_graph=True)[0]
    
    p_x = torch.autograd.grad(p.reshape(-1,1).sum(), x, create_graph=True)[0]
    p_y = torch.autograd.grad(p.reshape(-1,1).sum(), y, create_graph=True)[0]
    
    u_xx = torch.autograd.grad(u_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    u_yy = torch.autograd.grad(u_y, y, torch.ones(y.shape).to(device), create_graph=True)[0]
    
    v_xx = torch.autograd.grad(v_x, x, torch.ones(x.shape).to(device), create_graph=True)[0]
    v_yy = torch.autograd.grad(v_y, y, torch.ones(y.shape).to(device), create_graph=True)[0]
    
    Eu = u*u_x + v*u_y + p_x - (1/Re)*(u_xx + u_yy)
    Ev = u*v_x + v*v_y + p_y - (1/Re)*(v_xx + v_yy)
    Ec = u_x + v_y
        
    return pred_inter, Eu,Ev,Ec

def interpolation_infer_4points(x,y,model1,model2,model3,model4,weight1,weight2,weight3,weight4):
    pred1 = model1(x,y)
    pred2 = model2(x,y)
    pred3 = model3(x,y)
    pred4 = model4(x,y)
    pred_inter = (weight1 * pred1 + weight2 * pred2 + weight3 *  pred3 + weight4 * pred4) / (weight1+weight2+weight3+weight4)
    
    return pred_inter

def weight_pre(pt,line1,line2):
    weight1_total = []
    weight2_total = []
    
    a1,b1,c1 = line1[0], line1[1] ,line1[2] 
    a2,b2,c2 = line2[0], line2[1] ,line2[2] 

    for i in range(len(pt)):
        x = pt[i,0]
        y = pt[i,1]
        numerator_1 = np.abs(a1 * x + b1 * y + c1)
        numerator_2 = np.abs(a2 * x + b2 * y + c2)
        
        dominator_1 = np.sqrt(a1 ** 2.0 + b1 ** 2.0)
        dominator_2 = np.sqrt(a2 ** 2.0 + b2 ** 2.0)
        
        distance_1 = numerator_1 / dominator_1 
        distance_2 = numerator_2 / dominator_2 
        
        Sum = distance_1 + distance_2
        weight1 = 1 - distance_1/ Sum
        weight2 = 1 - distance_2/ Sum
        
        weight1_total.append(weight1)
        weight2_total.append(weight2)
    
    total_weight1 = torch.from_numpy(np.array(weight1_total).reshape(-1,1)).float().to('cuda')
    total_weight2 = torch.from_numpy(np.array(weight2_total).reshape(-1,1)).float().to('cuda')
    
    return total_weight1,total_weight2
