#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Dec 13 21:22:03 2020

@author: mutian
"""

# Graph that connects T&H Comparator Sequential1 Sequential2 for 14nm SARADC
# Design by Zihao Mai
# Date: 27-09-2020
#==================================================================
#*******************  Initialization  *****************************
#==================================================================

import numpy as np
from ampse_03 import TF_Model#, poly_5
import tensorflow as tf
import reg_database
from netlist_database import Compp_spice3, Seqpart1_spice, Seqpart2_spice, SARDAC_spice


#==================================================================
#*******************  Initialization  *****************************
#==================================================================
N_STAIRS =10
KT=4.14e-21                 # Boltzman Constant * 300


#==================================================================
#****************  Loading the Regressors  ************************
#==================================================================

def modules_instantiate(*args):
    
    # laoding the reg models
    load_address =  "regs/comp"
    comp_model = reg_database.COMP_model()
    comp_gen   = TF_Model(name = 'comp_sigmoid_v1', model = comp_model,load_address = load_address, err_save=True)
    comp_model.trainable = False
    
    load_address =  "regs/sardac"
    sardac_model = reg_database.SARDAC_model()
    sardac_gen   = TF_Model(name = 'sardac_sigmoid_v1', model = sardac_model,load_address = load_address, err_save=True)
    
    load_address =  "regs/seq1"
    seq1_model = reg_database.SEQ1_model()
    seq1_gen   = TF_Model(name = 'seq1_sigmoid_v1', model = seq1_model,load_address = load_address, err_save=True)
    
    load_address =  "regs/seq2"
    seq2_model = reg_database.SEQ2_model()
    seq2_gen   = TF_Model(name = 'seq2_sigmoid_v1', model = seq2_model,load_address = load_address, err_save=True)
    
    
    # loading the SPICE models
    
    comp_spice = Compp_spice3()
    seq1_spice = Seqpart1_spice()
    seq2_spice = Seqpart2_spice()
    sardac_spice = SARDAC_spice()
    
    
    return [seq1_gen,seq2_gen,comp_gen,sardac_gen],[seq1_spice,seq2_spice,comp_spice,sardac_spice]




#==================================================================
#****************  Defining sub-functions ************************
#==================================================================


#------converting 2-dimension result (parameter,merits) to 2 individual arrays for compatibility------
def twodim_maker(inlist):          
    p, y  = inlist[0],inlist[1]
    y = np.array(y)
    y = np.reshape(y,(1,len(y)))
    p = np.reshape(p,(1,len(p)))
    return p, y



#------Scales input from -1 to 1------
def tf_quant_with_sigmoid(sxin,num=10):     
    v = np.linspace(0.5/num,1-0.5/num,num)
    out=[]
    for vv in v:
        out.append(tf.nn.sigmoid(100.0*(sxin-vv)))
    
    return tf.reduce_sum(out,axis=0)+1.0


#==================================================================
#************************  MLG Graphs *****************************
#******   Indicates input parameters for each modules   ***********
#******   Conects inter loading parameters between models    ******
#********   sxin: 26 input parameters for all modules    **********
#**********   u: [number of bits, sampling frequency ]   **********
#****    lst_modules: regression models and spice models     ******
#****     tf_mod = True =>> predict using mode     ****************
#*****             False =>> predict using SPICE simulator   ******
#==================================================================
def mlg(sxin, u,  lst_modules,tf_mod = True):


    nbit = u[0]
    # fs   = u[1]
    
    #----------Loading models---------- 
    seq1_gen,seq2_gen,comp_gen,sardac_gen = lst_modules[0]
    seq1_spice,seq2_spice,comp_spice,sardac_spice = lst_modules[1]
    
    
    #----------SEQ1's graph----------   
    # input variables in SEQ1
#    [ 'drvbuf11','drvbuf21','drvbuf31','drvdffck1','drvinv1','drvnand1','fdffck','finv','fnand','div','mdacbig'] 
#       [1,1,1,1,1,1,1,1,1,1,0]
#     1:variables that is defined in this block     0:variables come from other modules

#    output ['pwrdac','pwrdff','dlydac', 'dlydff']

#   order of varables and output should follow exactly the same order as that in netlist database
   
    l = len(sxin[:,0])
    
#   feed in block defined variables
    vars_seqp1  = sxin[:,0:10]
#   feed in constant
    cnst_seqp1  = seq1_gen.slopeX[-1]*nbit*np.ones((l,1))+seq1_gen.smeanX[-1]
    # connect variables and constants
    sx_seqp1 = tf.concat([vars_seqp1,cnst_seqp1],axis=1)
    # scale inputs
    x_seqp1  = seq1_gen.iscaleX(sx_seqp1)
    
    if tf_mod:                  #tensor flow prediction
        y_seqp1  = seq1_gen.tf_predict(x_seqp1)
    else:                       #SPICE simulation
        lst_seq1  = seq1_spice.wholerun_std(np.array(x_seqp1[0]))
        x_seqp1, y_seqp1 = twodim_maker(lst_seq1)
        # print('x_seqp1=',end="")
        # print(x_seqp1)
        # x_seqp1[:,-1]  = np.log2(x_seqp1[:,-1])  
        # print('x_seqp1=',end="")
        # print(x_seqp1)
    #----------SEQ2's graph----------
#    [ 'nor3','fck1'] 
#    output ['pwrnor','pwrbuf','pwrcmp', 'dlynor','dlybuf','dlycmp']
    vars_seqp2  = sxin[:,10:12]
    sx_seqp2 = vars_seqp2
    x_seqp2  = seq2_gen.iscaleX(sx_seqp2)
    # y_seqp2  = seq2_gen.tf_predict(x_seqp2)
    
    if tf_mod:
        y_seqp2  = seq2_gen.tf_predict(x_seqp2)
    else:
        lst_seq2 = seq2_spice.wholerun_std(np.array(x_seqp2[0]))
        x_seqp2, y_seqp2 = twodim_maker(lst_seq2)
    #----------COMP's graph----------
#    ['fck1','fck2','fcompinv','finnn','fninv','fpinv','fpr1','fpr2','frdynand','fttt','fnor3','frdy','mrdy','fload']
#   [0,1,1,1,1,1,1,1,1,1,0,0,0,0,0]
#    mrdy == macbig, fnor3 == nor3
#    output ['power','dlyrdy','dlyrst','vomin','cin','irn']
    vars_comp  = sxin[:,12:21]
#     
    cnst_comp  = comp_gen.scaleX(tf.concat([x_seqp2[:,1:2],x_seqp2[:,0:1],x_seqp1[:,7:8]*2+x_seqp1[:,6:7],x_seqp1[:,10:11],x_seqp1[:,4:5]],axis=1),np.array([0,10,11,12,13]))
#    'fck1'from seq2, 'fnor3'='nor3' from seq2, 'frdy'= 'finv1'*2 + 'fdffck'  'mrdy' == 'macbig' 'fload'='drvinv1'
    sx_comp  = tf.concat([cnst_comp[:,0:1],vars_comp,cnst_comp[:,1:]],axis=1)
    x_comp  = comp_gen.iscaleX(sx_comp)
    # y_comp  = comp_gen.tf_predict(x_comp)

    if tf_mod:
        y_comp  = comp_gen.tf_predict(x_comp)
    else:
        lst_comp  = comp_spice.wholerun_std(np.array(x_comp[0]))
        x_comp, y_comp = twodim_maker(lst_comp)
        
        
    #----------SARDAC's graph----------
#    [ 'div','mdac',   'cs', 'fthn', 'fthp',   'cp']
    #    'div','mdac',   'cs', 'fthn', 'fthp',   'cp'
    # [0,0,1,1,1,0]    # print('bw1 ='%(bw1))
    # print('bw2 ='%(bw2))
#   output:    ovalue, dlydac, bw1,bw2, 
    vars_sardac=sxin[:,21:24]
    # cp is a variable from comparator output
    cp = y_comp[:,4:5]
    cnst_sardac  = sardac_gen.scaleX(tf.concat([x_seqp1[:,9:10],x_seqp1[:,10:11],tf.cast(cp,tf.float64)], axis=1),np.array([0,1,5]))
    sx_sardac  = tf.concat([cnst_sardac[:,0:2],vars_sardac,cnst_sardac[:,2:]],axis=1)
    x_sardac  = sardac_gen.iscaleX(sx_sardac)
    # y_sardac  = sardac_gen.tf_predict(x_sardac)

    if tf_mod:
        y_sardac  = sardac_gen.tf_predict(x_sardac)
    else:
        lst_sardac  = sardac_spice.wholerun_std(np.array(x_sardac[0]))
        x_sardac, y_sardac = twodim_maker(lst_sardac)
    # print(y_sardac)
#    if tf_mod:
    return [x_seqp1,x_seqp2,x_comp, x_sardac],[y_seqp1, y_seqp2,y_comp,y_sardac]
#    else:
#        return [p_seqp1,p_seqp2,p_comp, p_sardac],[y_seqp1, y_seqp2,y_comp,y_sardac]

#==================================================================
#************************  M2S Graphs *****************************
#******  Using predicted results from MLG to calculate  ***********
#****  require specifications and constrain then return cost   ****
#**************       p: parameters from mlg  *********************
#**************       m: metrics from mlg**************************
#********   sxin: 26 input parameters for all modules    **********
#**********   u: [number of bits, sampling frequency ]   **********
#****    lst_modules: regression models and spice models     ******
#****    weitghts: weights for specifications     *****************
#==================================================================
def m2s(p,m,u,sxin,lst_modules,weights, bottom_constraints, upper_constraints):
    
    nbit = u[0]
    fs   = u[1]
    seq1_gen,seq2_gen,comp_gen,sardac_gen = lst_modules[0]
    seq1_spice,seq2_spice,comp_spice,sardac_spice = lst_modules[1]
    
    # loading inputs
    y_seqp1, y_seqp2,y_comp,y_sardac = tf.cast(m[0],tf.float64),tf.cast(m[1],tf.float64),tf.cast(m[2],tf.float64),tf.cast(m[3],tf.float64)
    x_sardac = tf.cast(p[3],tf.float64)
    
    #--------Other variables--------

    n_dly = tf_quant_with_sigmoid(tf.math.abs(sxin[:,24]),N_STAIRS)
    d_tr  = (sxin[:,25]+1)/2.0
    #==================================================================
    #***************  Define constraints and Cost(P)  *****************
    #==================================================================
    specs = []
    
    dacdelay = y_sardac[:,1]+y_seqp1[:,2]+y_seqp1[:,3]
    digdelay = y_seqp2[:,3]+y_seqp2[:,5]+n_dly*y_seqp2[:,4]
    ctot = x_sardac[:,2]*2*(2**nbit)+y_comp[:,4]
    bw1 = tf.nn.elu(y_sardac[:,2])
    bw2 = tf.nn.elu(y_sardac[:,3])

#    d_ts  = tf.nn.sigmoid( d_tr)*(1.0-fs*100e-12)+fs*100e-12
    d_ts  = d_tr*(1.0-fs*200e-12)+fs*200e-12

    ts = d_ts * 1/fs 
    
    power_seq1 =     nbit*(y_seqp1[:,0]+y_seqp1[:,1])*fs/62.5e6
    power_seq2 =     nbit*(y_seqp2[:,0]+n_dly*y_seqp2[:,1]+y_seqp2[:,2])*fs/200e6
    power_comp =     nbit*y_comp[:,0]
    
    
    
    c10 = tf.cast(tf.math.log(10.0),tf.float64)
    c2  = tf.cast(tf.math.log(2.0),tf.float64)
    #   0       1          2        3       4       5   6       7 
    #'power','readyp','delayr','delayf','kickn','cin','scin','irn'
    
#    specs.append(digdelay-dacdelay-20e-12)                                                        # 0- Delay of DAC vs delay of the loop +
#    specs.append(1/fs - ts - nbit*(y_comp[:,1]+ y_comp[:,2]+ 2*digdelay)-100e-12)                 # 1- loop delay less than fs  +
##    specs.append(10*tf.math.log((4*y_sardac[:,2])**2/(4*KT/ctot+y_comp[:,5]**2))/c10)  
#    specs.append(10*tf.math.log((4*y_sardac[:,2])**2/(4*KT/ctot))/c10)                            # 2- SNR more than 6*nbit + 11.76 
##    specs.append(x_sardac[:,2] - 6*y_comp[:,6])                                                  # 3- Comparator non-linear Caps + 
#    specs.append(ts*bw1*6.28-c2*nbit)                                                             # 4- Track and Hold Law +
#    specs.append(ts*bw2*6.28-c2*nbit)                                                             # 5- Track and Hold Law +
#    specs.append(4-ts*bw1*6.28+c2*nbit)                                                           # 6- Track and Hold Law +
#    specs.append(4-ts*bw2*6.28+c2*nbit)                                                           # 7- Track and Hold Law +
#    specs.append(y_comp[:,3])                                                                     # 8- vomin to be less
#    specs.append(power_seq1+power_seq2+power_comp)                                                # 9- Power consumption - 
    
    
    
    specs.append(digdelay-dacdelay)                                                               # 0- Delay of DAC vs delay of the loop +
    specs.append(y_comp[:,1]+ y_comp[:,2]+ 2*digdelay)                                            # 1- loop delay less than fs  +
#    specs.append(10*tf.math.log((4*y_sardac[:,2])**2/(4*KT/ctot+y_comp[:,5]**2))/c10)  
    specs.append(10*tf.math.log((4*y_sardac[:,2])**2/(4*KT/ctot))/c10)                            # 2- SNR more than 6*nbit + 11.76 



#    specs.append(x_sardac[:,2] - 6*y_comp[:,6])                                                  # 3- Comparator non-linear Caps + 
    specs.append(ts*bw1*6.28-c2*nbit)                                                             # 4- Track and Hold Law +
    specs.append(ts*bw2*6.28-c2*nbit)                                                             # 5- Track and Hold Law +
    specs.append(4-ts*bw1*6.28+c2*nbit)                                                           # 6- Track and Hold Law +
    specs.append(4-ts*bw2*6.28+c2*nbit)                                                           # 7- Track and Hold Law +
    specs.append(y_comp[:,3])                                                                     # 8- vomin to be less
    specs.append(power_seq1+power_seq2+power_comp)                                                # 9- Power consumption - 
    
    
#    non-linear cap removed here specs missed [3]
#    constraints number adjustment?
    
    
    constraints = []    
    
    #======================== loop delay - dac delay====================
    if bottom_constraints[0] != None:
        constraints.append(tf.nn.elu(          -(specs[0] - bottom_constraints[0])/digdelay*weights[0]))
    else:
        constraints.append(tf.nn.elu(          -(specs[0] - 20e-12)/digdelay*weights[0]))
        
    if upper_constraints[0] != None:
       constraints.append(tf.nn.elu((specs[0] - upper_constraints[0])/digdelay*weights[0]))

    #======================= loop delay ================
    if bottom_constraints[1] != None:
        constraints.append(tf.nn.elu(-(specs[1]- bottom_constraints[1])/sardac_gen.slopeY[3]*weights[1]))
        
    if upper_constraints[1] != None:
        constraints.append(tf.nn.elu((specs[1]- upper_constraints[1])/sardac_gen.slopeY[3]*weights[1]))
    else:
        constraints.append(tf.nn.elu((specs[1]- (1/fs - ts - 100e-12)/nbit)/sardac_gen.slopeY[3]*weights[1]))
        
        
        
    # =============== SNR =============================
    if bottom_constraints[2] != None:
        constraints.append(tf.nn.elu(-(specs[2] - bottom_constraints[2])*weights[2]))
    else:
        constraints.append(tf.nn.elu(-(specs[2] - 6*nbit - 10)*weights[2]))
    
    if upper_constraints[2] != None:
        constraints.append(tf.nn.elu((specs[2] - upper_constraints[2])*weights[2]))
        
#    constraints.append(tf.nn.elu(-specs[3]/comp_gen.scYscale[5]*weights[3]))
    constraints.append(tf.nn.elu(                   -specs[3]*weights[3]))
    constraints.append(tf.nn.elu(                   -specs[4]*weights[4]))
    constraints.append(tf.nn.elu(                   -specs[5]*weights[5]))
    constraints.append(tf.nn.elu(                   -specs[6]*weights[6]))
    constraints.append(tf.nn.elu(             -(specs[7]-0.48)*weights[7]))
    
    #================== Power consumption=====================
    if bottom_constraints[-1] != None:
        constraints.append(tf.nn.elu(-(specs[-1] -  bottom_constraints[-1])/comp_gen.slopeY[0]*weights[8]))

    if upper_constraints[-1] != None:
         constraints.append(tf.nn.elu((specs[-1] -  upper_constraints[-1])/comp_gen.slopeY[0]*weights[8]))
    else:
        constraints.append(specs[-1]/comp_gen.slopeY[0]*weights[8])
    
    hardcost_0=tf.reduce_sum(constraints[0:-1],axis=0)
    softcost_0=tf.reduce_sum(constraints,axis=0)
    
    hardcost = tf.reduce_sum(hardcost_0)
    softcost = tf.reduce_sum(softcost_0)
    
    return softcost, hardcost, specs, [n_dly, d_tr], [dacdelay, digdelay, ctot, bw1, bw2, d_ts, power_seq1, power_seq2, power_comp], constraints, softcost_0, hardcost_0




#==================================================================
#************************  P2S Function ***************************
#****** Function for connecting MLG and M2S   *********************
#********   var0: 26 input parameters for all modules    **********
#**********   u: [number of bits, sampling frequency ]   **********
#****    lst_modules: regression models and spice models     ******
#****    weitghts: weights for specifications     *****************
#****     tf_mod = True =>> predict using mode     ****************
#*****             False =>> predict using SPICE simulator   ******
#==================================================================
def p2s(var0, u, weights, lst_modules, bottom_constraints, upper_constraints, tf_mod = True):

    sxin = 2*tf.nn.sigmoid(var0)-1
    # sxin = tf.nn.sigmoid(var0)
    # print(sxin)
#    u  = [6,500e6]
    p, m  = mlg(sxin, u,lst_modules, tf_mod )
#    
    softcost, hardcost, specs, pb, midvalues, const, softcost_0, hardcost_0 = m2s(p,m,u,sxin,lst_modules,weights, bottom_constraints, upper_constraints)

    return softcost, hardcost, specs, pb, midvalues, const, softcost_0, hardcost_0 ,p,m
    

    
    

if __name__ == '__main__':
    
    
    var0 = np.random.rand(1,26)
#    sxin = tf.Variable(var0)
    sxin = var0
    lst_mod = modules_instantiate()
    u  = [6,500e6]
    # p, m  = mlg(sxin, u, lst_mod, True )
    p, m  = mlg(sxin, u, lst_mod, False )
    
    weights = np.array([10.0,10.0,1.0,0.0,1.0,1.0,1.0,1.0,100.0,1.0/5])*1.0
    softcost, hardcost, specs, pb, midvalues, const, softcost_0, hardcost_0 = m2s(p,m,u,sxin, lst_mod,weights)
    
    

    
    
#    minimize(cost)
#    optimize.minimize()
    
