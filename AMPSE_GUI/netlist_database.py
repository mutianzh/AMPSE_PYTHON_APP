# This script shows the netlist database for the VCO-ADC Database Using the new version of spectreIOlib
import sys
#sys.path.insert(0,'/shares/MLLibs/GlobalLibrary')
sys.path.insert(0,'/home/mutianzh/PycharmProjects/GlobalLibrary')
#sys.path.insert(0,'/home/mohsen/PYTHON_PHD/GlobalLibrary')

import numpy as np
# import pandas as pd
import os
import math
from spectreIOlib import TestSpice, Netlists
import time
home_address  = os.getcwd()
#home_address  = '/home/Zihao/PYTHON_PHD/CADtoolForRegression/SAR_ADC'


    
class Compp_spice3(Netlists):
    
    def __init__(self, tech = 14, testfolder =None, paralleling=False,max_parallel_sim=4,memory_capacity=40 ):
        super().__init__(paralleling,max_parallel_sim, memory_capacity)
        if tech ==65:
            self.testbench = home_address + '/Netlists/TSMC65nm/complatch_v1_65_TT.scs'
            self.testfolder = home_address + '/Garbage/Comp65_1_3' if testfolder ==None else home_address + '/Garbage/' + testfolder            
            self.finaldataset = home_address + '/Datasets/PY_COMPPin6501_TT.csv'
        elif tech =='PTM65':
            self.testbench = home_address + '/Netlists/PTM65/complatch_v3_65_PTM.scs'
            self.testfolder = home_address + '/Garbage/CompPTM65_1_1' if testfolder ==None else home_address + '/Garbage/' + testfolder            
            self.finaldataset = home_address + '/Datasets/PY_COMPPinPTM6503.csv' 
        
        elif tech ==14:
            self.testbench = home_address + '/netlist/complatch.scs'
            self.testfolder = home_address + '/temp/Comp14nm_1_1' if testfolder ==None else home_address + '/temp/' + testfolder            
            self.finaldataset = home_address + '/Datasets/PY_COMPPin14nm01_TT.csv' 
            
        # self.minpar  = np.array([1    , 1    , 1        , 1     , 1     , 1     , 1    , 1    , 1        , 2    ,   1   ,   1  ,  4   ,  1    ,  4    ,   1     ,    1      ,100e-6, 0.4 , 1.0 ])
        # self.maxpar  = np.array([12   , 20   , 40       , 20    , 40    , 40    , 8    , 8    , 8        , 80   ,   10  ,   48 ,  12  ,  16   ,  12   ,   1     ,    1      ,100e-6, 0.4 , 1.0 ])
        # self.stppar  = np.array([1    , 1    , 1        , 1     , 1     , 1     , 1    , 1    , 1        , 2    ,   1   ,   1  ,  1   ,  1    ,  1    ,   1     ,    1      , 10e-6, 0.2 , 0.01])
        self.mps     = max_parallel_sim
        self.prl     = paralleling
        self.par_line_number = 8            
        self.minpar  = np.array([1     , 1    , 1        , 1     , 1     , 1     , 1    , 1    , 1        , 2    ,   1   ,   1  ,  4   ,  1    ])
        self.maxpar  = np.array([12    , 20   , 40       , 20    , 40    , 40    , 8    , 8    , 8        , 80   ,   10  ,   48 ,  12  ,  16   ])
        self.stppar  = np.array([1     , 1    , 1        , 1     , 1     , 1     , 1    , 1    , 1        , 2    ,   1   ,   1  ,  1   ,  1    ])
        self.parname =          ['fck1','fck2','fcompinv','finnn','fninv','fpinv','fpr1','fpr2','frdynand','fttt','fnor3','frdy','mrdy','fload']
        self.metricname =       ['power','dlyrdy','dlyrst','vomin','cin','irn']
        self.make_metrics()    


    def change_testfolder(self, testfolder=None):
        self.testfolder = self.testfolder if testfolder ==None else home_address + '/temp/' + testfolder
        self.make_metrics()
        pass
        
        
    def make_metrics(self):
#        afs
#        x=np.array([9,10,11,12,18,22])

#        aps
        x=np.array([10,11,12,13,18,26])
#        z=[]
#        for i in range(7):
#            y = i*13+x;
#            z = z + list(y)
        
        self.metricpositions =x
        self.runspectre1=TestSpice(simulator='aps',dict_folder={'testbench':self.testbench, 'trashfolder':self.testfolder},paralleling=self.prl,maxparallelsim=self.mps,verbose = True)
        pass   
            
    def normal_run(self,param,lst_alter=[],parallelid=None):
        
        self.dict_parameters = {'line_number':self.par_line_number,'name_params':self.parname,'value_params':param}
        x = self.runspectre1.runspectre(dict_parameters=self.dict_parameters,alter_list=lst_alter,parallelid=parallelid)
#        code for afs
#        out_measure = '/test'+str(x)+'.out/test'+str(x)+'.measure'

#        code for aps
        out_measure = '/test'+str(x)+'.measure'
        
        self.lst_metrics=[{'read':'c','filename':self.testfolder + out_measure,'number':2,'measurerange':self.metricpositions}]
        out1 = self.runspectre1.readmetrics(lst_metrics=self.lst_metrics)
        return out1
    
    
    def analysis(self,lst_out,length=0):
  
        
        out1=[]
        x      = np.array(lst_out[0])
        power  =-x[0]
        dlyrdy  = x[1]
        dlyrst = x[2]
        vomin  = x[3]
        cin    = x[4]
        irn    = 2*x[-1]
        # noise data not avaliable
        # irn    = 0
        if length>0:
            out1.append([power,dlyrdy,dlyrst,vomin,cin,irn])
        else:
            out1 = [power,dlyrdy,dlyrst,vomin,cin,irn]
        return out1
    
    
    def wholerun_normal(self,param,lst_alter=[],parallelid=None):
        x = self.normal_run(param,lst_alter,parallelid=parallelid)
        w = self.analysis(x,len(lst_alter))       
        return w
    
    
   



    
class SARDAC_spice(Netlists):
    
    def __init__(self, tech = 14, testfolder =None, paralleling=False,max_parallel_sim=4,memory_capacity=40 ):
        super().__init__(paralleling,max_parallel_sim, memory_capacity)
        if tech ==65:
            self.testbench = home_address + '/Netlists/sardac_v2_65_TT.scs'
            self.testfolder = home_address + '/temp/sardac65_2_1' if testfolder ==None else home_address + '/temp/' + testfolder     
            self.minpar  = np.array([2    ,  4    ,0.5e-15, 2     , 2     , 1     , 1     , 2.0e-15 ])
            self.maxpar  = np.array([16   ,  12   ,5.0e-15, 40    , 60    , 1     , 1     ,  30e-15 ])
            self.stppar  = np.array([2    ,  1    ,0.1e-15, 2     , 2     , 1     , 1     , 0.5e-15 ])
            self.finaldataset = home_address + '/Datasets/PY_sardac6501.csv'

        elif tech =='PTM65':
            self.testbench = home_address + '/Netlists/PTM65/dardac_v2_65_PTM.scs'
            self.testfolder = home_address + '/temp/sardacPTM65_2_1' if testfolder ==None else home_address + '/temp/' + testfolder
            self.minpar  = np.array([2    ,  4    ,0.5e-15, 2     , 2     , 1     , 1     , 2.0e-15 ])
            self.maxpar  = np.array([16   ,  12   ,5.0e-15, 40    , 60    , 1     , 1     ,  30e-15 ])
            self.stppar  = np.array([2    ,  1    ,0.1e-15, 1     , 1     , 1     , 1     , 0.5e-15 ])
            self.finaldataset = home_address + '/Datasets/PY_DACTHPTM6501.csv'  
        
        elif tech ==14:
            self.testbench = home_address + '/netlist/sardac.scs'
            self.testfolder = home_address + '/temp/sardac14nm_1_1' if testfolder ==None else home_address + '/temp/' + testfolder
            self.minpar  = np.array([2    ,  4    ,0.5e-15, 2     , 2     ,  1.6e-15 ])
            self.maxpar  = np.array([16   ,  12   ,5.0e-15, 40    , 60    ,  40.0e-15 ])
            self.stppar  = np.array([2    ,  1    ,0.1e-15, 1     , 1     ,  0.1e-15 ])
            self.finaldataset = home_address + '/Datasets/PY_sardac14nm01.csv'
            
        self.mps     = max_parallel_sim
        self.prl     = paralleling
        self.par_line_number = 8   
        self.parname =          [ 'div','mdac',   'cs', 'fthn', 'fthp',   'cp']            
        self.metricname = ['ovalue','dlydac','bw1','bw2']
        self.make_metrics()
        
#        save_address = "/home/Zihao/python_code/SAR_ADC_v2/regs/sardac"
#    sardac_gen.save_model(save_address,err_save=True)
    def make_metrics(self):
        z = [10,11,16,17]
        self.metricpositions =z
        
        self.runspectre1=TestSpice(simulator='aps',dict_folder={'testbench':self.testbench, 'trashfolder':self.testfolder},paralleling=self.prl,maxparallelsim=self.mps,verbose = True)
        pass
    
    def analysis(self,lst_out):
        z = lst_out[0]
        bw1 = z[2]
        bw2 = z[3]
        ovalue = z[1]
        dlydac= z[0]

#        out1 = [bw1,bw2, ovalue, dlydac]
        out1 = [ovalue, dlydac, bw1,bw2,]
        return out1

    def normal_run(self,param,lst_alter=[],parallelid=None):
        p = np.copy(param)
        p[1] = 2**param[1]
#        why need 2*param
        self.dict_parameters = {'line_number':self.par_line_number,'name_params':self.parname,'value_params':p}
        x = self.runspectre1.runspectre(dict_parameters=self.dict_parameters,alter_list=lst_alter,parallelid=parallelid)
#        code for afs
#        out_measure = '/test'+str(x)+'.out/test'+str(x)+'.measure'
#        code for aps
        out_measure = '/test'+str(x)+'.measure'
        self.lst_metrics=[{'read':'c','filename':self.testfolder + out_measure,'number':2,'measurerange':self.metricpositions}]
        out1 = self.runspectre1.readmetrics(lst_metrics=self.lst_metrics)
        return out1

    def wholerun_normal(self,param,parallelid=None):
        x = self.normal_run(param,parallelid=parallelid)
        w = self.analysis(x)       
        return w



class Seqpart1_spice(Netlists):
    
    def __init__(self, tech = 14, testfolder =None, paralleling=False,max_parallel_sim=4, memory_capacity=40 ):
        super().__init__(paralleling,max_parallel_sim, memory_capacity)

        if tech ==65:
            self.testbench = home_address + '/Netlists/sequential_v2_part1_65_TT.scs'
            self.testfolder = home_address + '/Garbage/sequential65p1_1_2' if testfolder ==None else home_address + '/Garbage/' + testfolder
            self.minpar  = np.array([          1,         2,         2,          1,        2,         2,       1,     1,      1,       1,       1,    2,        3 ])
            self.maxpar  = np.array([         12,        24,        96,         10,       16,        24,      16,    16,     16,       1,       1,   16,        11])
            self.stppar  = np.array([          1,         2,         2,          1,        2,         2,       1,     1,      1,       1,       1,    2,        1 ])
            self.finaldataset = home_address + '/Datasets/PY_Seqp1_6501_TT.csv'
        elif tech =='PTM65':
            self.testbench = home_address + '/Netlists/PTM65/sequential_v2_part1_65_PTM.scs'
            self.testfolder = home_address + '/Garbage/sequentialPTM65p1_1_1' if testfolder ==None else home_address + '/Garbage/' + testfolder
            self.minpar  = np.array([          1,         2,         2,          1,        2,         2,       1,     1,      1,       1,       1,    2,        3 ])
            self.maxpar  = np.array([         12,        24,        96,         10,       16,        32,      16,    16,     16,       1,       1,   16,        11])
            self.stppar  = np.array([          1,         2,         2,          1,        2,         2,       1,     1,      1,       1,       1,    2,        1 ])
            self.finaldataset = home_address + '/Datasets/PY_Seqp1_PTM6501.csv'  
        elif tech ==14:
            self.testbench = home_address + '/netlist/sequential_part1.scs'
            self.testfolder = home_address + '/temp/sequential14nmp1_1_1' if testfolder ==None else home_address + '/temp/' + testfolder   
            self.minpar  = np.array([          1,         2,         2,          1,        2,         2,       1,     1,      1,    2,        3 ])
            self.maxpar  = np.array([         12,        24,        96,         10,       16,        32,      16,    16,     16,   16,        11])
            self.stppar  = np.array([          1,         2,         2,          1,        2,         2,       1,     1,      1,    2,        1 ])
            self.finaldataset = home_address + '/Datasets/PY_Seqp1_14nm01_TT.csv' 
            
        self.mps     = max_parallel_sim
        self.prl     = paralleling
        self.par_line_number = 8 
#        self.parname =          [ 'drvbuf11','drvbuf21','drvbuf31','drvdffck1','drvinv1','drvnand1','fdffck','finv','fnand','frefnn','frefpp','div','mdacbig']  
        self.parname =          [ 'drvbuf11','drvbuf21','drvbuf31','drvdffck1','drvinv1','drvnand1','fdffck','finv','fnand','div','mdacbig']            
#        self.parname =          [ 'drvbuf11','drvbuf21','drvbuf31','drvdffck1','drvinv1','drvnand1','fdffck','finv','fnand','frefnn','frefpp','nor3','mdacbig']
        self.metricname = ['pwrdac','pwrdff','dlydac', 'dlydff']
        self.make_metrics()
        
#     def change_testfolder(self, testfolder=None):
#        self.testfolder = self.testfolder if testfolder ==None else home_address + '/temp/' + testfolder
#        self.make_metrics()
#        pass
    
    def make_metrics(self):
#        afs
#        z = [9,10,11,12]
#        aps
        z = [10,11,12,13]
        self.metricpositions =z
        
        self.runspectre1=TestSpice(simulator='aps',dict_folder={'testbench':self.testbench, 'trashfolder':self.testfolder},paralleling=self.prl,maxparallelsim=self.mps,verbose = True)
        pass
        
        
        
        pass
    
    def analysis(self,lst_out):
        z = lst_out[0]
        
        pwrdac  = -z[2]/2
        pwrdff  = -z[3]/2
        dlydac  = z[0]
        dlydff  = z[1]

        out1 = [pwrdac, pwrdff, dlydac, dlydff]
        return out1

    def normal_run(self,param,lst_alter=[],parallelid=None):
        p = np.copy(param)
        p[-1] = 2**param[-1]
        
        self.dict_parameters = {'line_number':self.par_line_number,'name_params':self.parname,'value_params':p}
        x = self.runspectre1.runspectre(dict_parameters=self.dict_parameters,alter_list=lst_alter,parallelid=parallelid)        
#        code for afs
#        out_measure = '/test'+str(x)+'.out/test'+str(x)+'.measure'
#        code for aps
        out_measure = '/test'+str(x)+'.measure'
        self.lst_metrics=[{'read':'c','filename':self.testfolder + out_measure,'number':2,'measurerange':self.metricpositions}]
        out1 = self.runspectre1.readmetrics(lst_metrics=self.lst_metrics)
        return out1

    def wholerun_normal(self,param,parallelid=None):
        x = self.normal_run(param,parallelid=parallelid)
        w = self.analysis(x)       
        return w  

class Seqpart2_spice(Netlists):
    
    def __init__(self, tech = 14, testfolder =None, paralleling=False,max_parallel_sim=4,memory_capacity=40):
        super().__init__(paralleling,max_parallel_sim, memory_capacity)

        if tech ==65:
            self.testbench = home_address + '/Netlists/sequential_v1_part2_65_TT.scs'
            self.testfolder = home_address + '/Garbage/sequential65p2_1_1' if testfolder ==None else home_address + '/Garbage/' + testfolder
            self.minpar  = np.array([      1.,     1.])
            self.maxpar  = np.array([     10.,    12.])
            self.stppar  = np.array([      1.,     1.])
                       
            self.finaldataset = home_address + '/Datasets/PY_Seqp2_6501_TT.csv'#     p,w = mysardac1.wholerun_random()
        elif tech =='PTM65':
            self.testbench = home_address + '/Netlists/PTM65/sequential_v1_part2_65_PTM.scs'
            self.testfolder = home_address + '/Garbage/sequentialPTM65p2_1_1' if testfolder ==None else home_address + '/Garbage/' + testfolder
            self.minpar  = np.array([      1.,     1.])
            self.maxpar  = np.array([     10.,    12.])
            self.stppar  = np.array([      1.,     1.])                     
            self.finaldataset = home_address + '/Datasets/PY_Seqp2_PTM6501.csv' 
            
        elif tech ==14:
            self.testbench = home_address + '/netlist/sequential_part2.scs'
            self.testfolder = home_address + '/temp/sequential14nmp2_1_1' if testfolder ==None else home_address + '/temp/' + testfolder
            self.minpar  = np.array([      1.,     1.])
            self.maxpar  = np.array([     10.,    12.])
            self.stppar  = np.array([      1.,     1.])                     
            self.finaldataset = home_address + '/Datasets/PY_Seqp2_14nm01_TT.csv' 
            
        self.mps     = max_parallel_sim
        self.prl     = paralleling
        self.par_line_number = 8   
        self.parname =          [ 'nor3','fck1']            
        self.metricname = ['pwrnor','pwrbuf','pwrcmp', 'dlynor','dlybuf','dlycmp']
        self.make_metrics() 
    
    def make_metrics(self):
#        afs
#        z = [9,10,11,12,13,14]
#        aps
        z = [10,11,12,13,14,15]
        self.metricpositions =z
        
        self.runspectre1=TestSpice(simulator='aps',dict_folder={'testbench':self.testbench, 'trashfolder':self.testfolder},paralleling=self.prl,maxparallelsim=self.mps,verbose = True)
        
        pass
    
    def analysis(self,lst_out):
        z = lst_out[0]
        
        pwrnor  = -z[5]/5
        pwrbuf  = -z[3]/5
        pwrcmp  = -z[4]/5
        dlynor  =  z[2]
        dlybuf  =  z[0]
        dlycmp  =  z[1]
        
        out1 = [pwrnor, pwrbuf, pwrcmp, dlynor, dlybuf, dlycmp]
        return out1

    def normal_run(self,param,lst_alter=[],parallelid=None):

        
        self.dict_parameters = {'line_number':self.par_line_number,'name_params':self.parname,'value_params':param}
        x = self.runspectre1.runspectre(dict_parameters=self.dict_parameters,alter_list=lst_alter,parallelid=parallelid)        
#        code for afs
#        out_measure = '/test'+str(x)+'.out/test'+str(x)+'.measure'
#        code for aps
        out_measure = '/test'+str(x)+'.measure'
        self.lst_metrics=[{'read':'c','filename':self.testfolder + out_measure,'number':2,'measurerange':self.metricpositions}]
        out1 = self.runspectre1.readmetrics(lst_metrics=self.lst_metrics)
        return out1
    
    
    def wholerun_normal(self,param,parallelid=None):
        x = self.normal_run(param,parallelid=parallelid)
        w = self.analysis(x)       
        return w  


if __name__ == '__main__':
    
    mysardac1 = SARDAC_spice(tech=14)
#     out = mysardac1.random_run()
    p,w = mysardac1.wholerun_random()
    
    # mycomp3 = Compp_spice3(tech=14)
#     out = mycomp3.random_run()
    # p, w = mycomp3.wholerun_random()
    
    
    # myseq1 = Seqpart1_spice(tech=14)
#     out = myseq1.random_run()
    # p,w = myseq1.wholerun_random()    
#
    # myseq2 = Seqpart2_spice(tech=14)
# #     out = myseq2.random_run()
    # p,w = myseq2.wholerun_random()  

