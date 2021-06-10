#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 24 11:39:13 2020

User interface with

@author: mutian
"""

import sys
#from .utils import nbit, sampling_frequency, nbit_lower_bound, fs_lower_bound, power_upper_bound
import threading
import numpy as np
import tensorflow as tf
from AMPSE_Graphs_custom import p2s, modules_instantiate
from time import time
import json
import wx
import wx.grid as grid
import wx.lib.scrolledpanel as scrolled
from multiprocessing import Process



# path for figures

# logo_path = '.\\logo.JPG'
# KGD_path = '.\\circuit_graph.png'
# circuit_graph_path = '.\\circuit_graph.png'
# suggested_region_path = '.\\suggested_region.png'

#Mutian's path
logo_path = './logo.jpg'
KGD_path = './circuit_graph.png'
circuit_graph_path = './circuit_graph.png'
suggested_region_path = './suggested_region.png'


# results_dict = {'np_specs':[],
#        'np_mids' :[],
#        'np_const':[],
#        'np_cost' :[],
#        'np_var'  :[],
#        'np_u'    :[],
#        'np_p'    :[],
#        'np_m'    :[],
#        'np_weights':[],
#        'np_grads':[]}

def search(fs, b, power_upper_bound):
    log_dir = 'tb'
    lst_mod = modules_instantiate()

    EPS = 1e-3  # minimum discrepancy to stop optimizing
    MAXITER = 30  # maximum iteration
    NMC = 10  # number of monte-carlo simulations
    TFMOD = True

    u = [b, fs]
    bottom_constraints = [0, None, None, None, None, None, None, None, None]  # Enforce larger than
    upper_constraints = [None, None, None, None, None, None, None, None, power_upper_bound]  # Enforce smaller than
    weights = np.array([10.0, 1.0, 1.0, 1.0, 1.0, 0.0, 0.0, 10.0, 1.0 / 50]) * 1.0  # weights for specs

    # ----------generating ramdom input variables----------
    var_Adam = np.random.rand(NMC, 26)
    var_Adam = tf.Variable(var_Adam)

    # ----------generating ramdom input variables----------
    def cost_tf_Adam():
        softcost, _, _, _, _, _, _, _, _, m = p2s(var_Adam, u, weights, lst_mod, bottom_constraints, upper_constraints,
                                                  TFMOD)
        softcost_new = softcost  # +tf.nn.elu(m[0][1]-2m)
        return softcost_new

    def cost_tf_hard_Adam():
        _, hardcost, _, _, _, _, _, _, _, _ = p2s(var_Adam, u, weights, lst_mod, bottom_constraints, upper_constraints,
                                                  TFMOD)
        return hardcost

    # initializing
    current_cost_Adam = 1e12
    t_Adam_Start = time()

    # ----------initializing keras optimizer and parameters----------
    opt_Adam = tf.keras.optimizers.Adam(learning_rate=0.1)

    # ----------optimizing hardcost----------
    for i in range(MAXITER):
        prev_cost_Adam = current_cost_Adam
        step_count_Adam = opt_Adam.minimize(cost_tf_hard_Adam, [var_Adam]).numpy()
        current_cost_Adam = cost_tf_hard_Adam().numpy()
        print('Iteration %1.0f, Cost = %1.3f, time elapsed %1.2f' % (i, current_cost_Adam, time() - t_Adam_Start))
        if abs(current_cost_Adam - prev_cost_Adam) < EPS * 100 * NMC:
            break

    # ----------optimizing softcost----------

    opt_Adam.lr = 0.01
    for i in range(MAXITER):
        prev_cost_Adam = current_cost_Adam
        step_count_Adam = opt_Adam.minimize(cost_tf_Adam, [var_Adam]).numpy()
        current_cost_Adam = cost_tf_Adam().numpy()
        print('Iteration %1.0f, Cost = %1.3f, time elapsed %1.2f' % (i, current_cost_Adam, time() - t_Adam_Start))
        if abs(current_cost_Adam - prev_cost_Adam) < EPS * NMC:
            break

    # ----------save total time used----------
    t_Adam = time() - t_Adam_Start

    _, _, specs_nn, pb, midvalues_nn, const, softcost0, hardcost0, ph, mh = p2s(var_Adam, u, weights, lst_mod,
                                                                                bottom_constraints, upper_constraints,
                                                                                True)
    # _, _, specs, _, midvalues, _, sp_cost0, _,p,m = p2s(var_Adam[2:3,:],u, weights,lst_mod,False )

    # ----------get variable gradients----------
    with tf.GradientTape() as tape:
        loss = cost_tf_Adam()
    grads = tape.gradient(loss, var_Adam)

    # ==================================================================
    # *******************  Saving results  *****************************
    # ==================================================================
    np_pb = np.array(pb)
    np_specs = np.array(specs_nn)
    np_mids = np.array(midvalues_nn)
    np_const = np.array(const)
    np_cost = np.array(softcost0)
    np_var = var_Adam.numpy()
    np_u = np.array(u)
    np_p = [[ph[j][i, :].numpy().tolist() for j in range(4)] + [np_pb[:, i].tolist()] for i in range(NMC)]
    np_m = [[mh[j][i, :].numpy().tolist() for j in range(4)] for i in range(NMC)]
    np_grads = grads.numpy()

    result_dict = dict()
    if power_upper_bound == None:
        const_range = np_const.shape[0] - 1
    else:
        const_range = np_const.shape[0]

    num_found = 0
    candidate_index = 0
    min_power = 99

    # Find the design that meet all constraints with minimum power consumption
    for i in range(np_const.shape[1]):
        violations = 0
        for j in range(const_range):
            if np_const[j][i] > 0:
                violations += 1

        if violations == 0:  # all constraints are meet
            num_found += 1

            current_power = np_specs[-1,i]
            if current_power < min_power:
                candidate_index = i
                min_power = current_power


    if num_found > 0:
        print("Found!")

        mdict = {'power': np_specs[-1,candidate_index],
                 'params': np_p[candidate_index],
                 'specs': list(np_specs[:,candidate_index])
                 }

        result_dict = mdict
        return result_dict
    else:
        return None


def binary_search(f_min, f_max, b):
    print(f'Trying for nbit {b}, sampling frequency {f_min}')
    best_result = None

    result = search(f_min, b, power_upper_bound)
    if result == None:
        return None
    else:
        best_result = result
        best_fs = f_min
        f_mid = np.round((f_min + f_max) / 2)
        while np.abs(f_mid - f_max) > 50e6:
            print(f'Trying for nbit {b}, sampling frequency {f_mid}')
            result = search(f_mid, b, power_upper_bound)
            if result == None:
                f_max = f_mid
            else:
                f_min = f_mid
                best_result = result
                best_fs = f_mid

            f_mid = np.round((f_max + f_min) / 2)

    if best_result == None:
        return None
    else:
        return {'candidates': best_result, 'best_fs': best_fs}


def scale_bitmap(bitmap, width, height):
    image = bitmap.ConvertToImage()
    image = image.Scale(width, height, wx.IMAGE_QUALITY_HIGH)
    result = wx.Bitmap(image)
    return result

class SecondFrame(wx.Frame):
    """"""
    #----------------------------------------------------------------------
    def __init__(self, title, image, size):
        """Constructor"""
        wx.Frame.__init__(self, None, title=title, size=size)
        panel = wx.Panel(self)
        bmp1 = wx.Image(image, wx.BITMAP_TYPE_ANY).ConvertToBitmap()
        self.bitmap1 = wx.StaticBitmap(self, -1, bmp1, (0, 0))

class ThirdFrame(wx.Frame):
    def __init__(self, position, dict, dict_m):
        wx.Frame.__init__(self, None, title=f"Design {position+1}", size=(2000,1000))

        grid = wx.grid.Grid(self, -1)

        grid.CreateGrid(30, 20)
        grid.SetColSize(0, 240)
        grid.EnableEditing(False)

        # Display parameters
        grid.SetCellValue(0, 0, 'Parameters')
        grid.SetCellFont(0,0, wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))

        keys = list(dict.keys())
        row_index = 2
        for key in keys:
            element = dict[key]
            try:
                names = list(element.keys())
                grid.SetCellValue(row_index, 0, key)
                for i in range(len(names)):
                    grid.SetCellValue(row_index - 1, i + 1, names[i])  # Write parameter names
                    grid.SetCellValue(row_index, i + 1, f'{element[names[i]]}')  # Write param values

                row_index += 3
            except:
                grid.SetCellValue(row_index-1, 0, key)
                grid.SetCellValue(row_index-1, 1, f'{element}')
                row_index += 2


        # Display metrics
        grid.SetCellValue(row_index, 0, 'Metrics')
        grid.SetCellFont(row_index, 0, wx.Font(12, wx.SWISS, wx.NORMAL, wx.BOLD))
        row_index += 1
        keys = list(dict_m.keys())
        for key in keys:
            grid.SetCellValue(row_index, 0, key)
            grid.SetCellValue(row_index, 1, f'{dict_m[key]}')
            row_index += 1

        self.Show()



class MyFrame(wx.Frame):
    def __init__(self):
        super().__init__(parent=None, title='AMPSE')

        # panel = wx.Panel(self, size=(1000,400))
        panel = scrolled.ScrolledPanel(self, -1, size=(1000, 400), pos=(0, 28),style=wx.SIMPLE_BORDER)
        panel.SetupScrolling()


        sizer = wx.GridBagSizer(5, 5)

        '''Section 1 '''
        section1 = wx.BoxSizer(wx.HORIZONTAL)

        logo = wx.StaticBitmap(panel, bitmap=wx.Bitmap(logo_path))
        posh_grp_title = wx.StaticText(panel, pos=(0, 1), label="USC POSH GROUP")
        posh_grp_title.SetFont(wx.Font(10, wx.DEFAULT, wx.NORMAL, wx.BOLD))

        posh_tool_title = wx.StaticText(panel, pos=(0, 1), label="Analog Mixed Signal Parameter Search Engine")
        posh_tool_title.SetFont(wx.Font(14, wx.DEFAULT, wx.NORMAL, wx.BOLD))


        introduction = wx.BoxSizer(wx.VERTICAL)
        introduction.Add(posh_tool_title, flag=wx.TOP | wx.RIGHT, border=10)
        introduction.Add(posh_grp_title, flag=wx.TOP|wx.RIGHT, border=10)
        introduction.Add(wx.StaticText(panel, pos=(0, 1), label="Tony Levi"), flag=wx.TOP, border=5)
        introduction.Add(wx.StaticText(panel, pos=(0, 1), label="Mike Chen"), flag=wx.TOP, border=3)
        introduction.Add(wx.StaticText(panel, pos=(0, 1), label="Sandeep Gupta"), flag=wx.TOP, border=3)

        section1.Add(logo, flag=wx.TOP|wx.LEFT)
        section1.Add(introduction, flag=wx.LEFT, border=20)

        sizer.Add(section1, pos=(0, 0), flag=wx.EXPAND|wx.LEFT|wx.TOP, border=10)


        '''Section 2'''

        #line
        sizer.Add(wx.StaticLine(panel), pos=(1, 0), span=(1, 5),
            flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=10)

        label_sar_adc = wx.StaticText(panel, label="SAR ADC")
        font = wx.Font(14, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        label_sar_adc.SetFont(font)

        section2_description = wx.BoxSizer(wx.VERTICAL)
        section2_description.Add(label_sar_adc)
        section2_description.Add(wx.StaticText(panel, label="A SAR-ADC is a type of analog-to-digital converter using a binary search through all possible \n "
                                                            "quantization levels before finally converging upon a digital output for each conversion."))

        sizer.Add(section2_description, pos=(2, 0), flag=wx.LEFT | wx.BOTTOM, border=10)

        section2 = wx.BoxSizer(wx.HORIZONTAL)
        kgd_section = wx.BoxSizer(wx.VERTICAL)
        circuit_section = wx.BoxSizer(wx.VERTICAL)
        suggested_section = wx.BoxSizer(wx.VERTICAL)

        # kgd_image = wx.StaticBitmap(panel, bitmap=wx.Bitmap('./kgd_small.png'))
        # circuit_image = wx.StaticBitmap(panel, bitmap=wx.Bitmap('./kgd_small.png'))

        # suggested_image = wx.StaticBitmap(panel, bitmap=wx.Bitmap('./suggested_region_small.png'))
        suggested_image = wx.Bitmap('./suggested_region.png')
        suggested_image = scale_bitmap(suggested_image, 500, 300)
        suggested_image = wx.StaticBitmap(panel, bitmap=suggested_image)

        # #===Display KGD===
        # kgd_btn = wx.Button(panel, label='KGD')
        # kgd_section.Add(kgd_btn, 1, wx.CENTER)
        # kgd_section.Add(kgd_image, flag=wx.TOP, border=10)
        # kgd_btn.Bind(wx.EVT_BUTTON, self.OnShowKGD)
        # kgd_image.Bind(wx.EVT_LEFT_DOWN, self.OnShowKGD)

        # #===Display viable region===
        # suggested_btn = wx.Button(panel, label='Suggested Region')
        # suggested_section.Add(suggested_btn, 1, wx.CENTER)
        # suggested_btn.Bind(wx.EVT_BUTTON, self.OnShowRegion)

        suggested_section.Add(suggested_image, flag=wx.TOP, border=10)
        # suggested_image.Bind(wx.EVT_LEFT_DOWN, self.OnShowRegion)

        # #===Display MLG===
        circuit_btn = wx.Button(panel, label='Click to see circuit graph')
        circuit_section.Add(circuit_btn, 1, wx.RIGHT, border=10)
        circuit_btn.Bind(wx.EVT_BUTTON, self.OnShowCircuit)
        # circuit_section.Add(wx.StaticText(panel, label = 'Circuit Graph'))
        # circuit_section.Add(circuit_image, flag=wx.TOP, border=10)
        # circuit_image.Bind(wx.EVT_LEFT_DOWN, self.OnShowCircuit)

        section2.Add(suggested_section, flag=wx.RIGHT, border=10)
        # section2.Add(kgd_section, flag=wx.RIGHT, border=10)
        section2.Add(circuit_section, flag=wx.RIGHT, border=10)


        sizer.Add(section2, pos=(3, 0),
            flag=wx.EXPAND|wx.LEFT, border=10)

        #line
        sizer.Add(wx.StaticLine(panel), pos=(4, 0), span=(1, 5),
            flag=wx.EXPAND|wx.BOTTOM|wx.TOP, border=10)

        '''Section 3 '''

        label_sar_adc = wx.StaticText(panel, label="User inputs")
        font = wx.Font(14, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        label_sar_adc.SetFont(font)
        sizer.Add(label_sar_adc, pos=(5, 0), flag=wx.LEFT | wx.BOTTOM, border=10)


        #n bits
        nbits_section = wx.BoxSizer(wx.HORIZONTAL)
        label_nbits = wx.StaticText(panel, label="Number of bits:")
        self.input_nbits = wx.TextCtrl(panel)
        nbits_section.Add(label_nbits, flag=wx.LEFT, border=10)
        nbits_section.Add(self.input_nbits, flag=wx.LEFT, border=110)
        sizer.Add(nbits_section, pos=(6, 0), flag=wx.TOP|wx.EXPAND)

        #sampling frequency
        fs_section = wx.BoxSizer(wx.HORIZONTAL)
        label_fs = wx.StaticText(panel, label="Sampling frequency:")
        self.input_fs = wx.TextCtrl(panel)
        fs_section.Add(label_fs, flag=wx.LEFT, border=10)
        fs_section.Add(self.input_fs, flag=wx.LEFT, border=84)
        sizer.Add(fs_section, pos=(7, 0), flag=wx.TOP|wx.EXPAND)

        #nbits_lb
        nbits_lb_section = wx.BoxSizer(wx.HORIZONTAL)
        label_nbits_lb = wx.StaticText(panel, label="Bits lower bound:")
        self.input_nbits_lb = wx.TextCtrl(panel)
        nbits_lb_section.Add(label_nbits_lb, flag=wx.LEFT, border=10)
        nbits_lb_section.Add(self.input_nbits_lb, flag=wx.LEFT, border=101)
        sizer.Add(nbits_lb_section, pos=(8, 0), flag=wx.TOP|wx.EXPAND)

        #fs_lb
        fs_lb_section = wx.BoxSizer(wx.HORIZONTAL)
        label_fs_lb = wx.StaticText(panel, label="Sampling frequency lower bound:")
        self.input_fs_lb = wx.TextCtrl(panel)
        fs_lb_section.Add(label_fs_lb, flag=wx.LEFT, border=10)
        fs_lb_section.Add(self.input_fs_lb, flag=wx.LEFT, border=14)
        sizer.Add(fs_lb_section, pos=(9, 0), flag=wx.TOP|wx.EXPAND)

        #power_ub
        power_ub_section = wx.BoxSizer(wx.HORIZONTAL)
        label_power_ub = wx.StaticText(panel, label="Power upper bound:")
        self.input_power_ub = wx.TextCtrl(panel)
        power_ub_section.Add(label_power_ub, flag=wx.LEFT, border=10)
        power_ub_section.Add(self.input_power_ub, flag=wx.LEFT, border=85)
        sizer.Add(power_ub_section, pos=(10, 0), flag=wx.TOP|wx.EXPAND)

        search_btn = wx.Button(panel, label='Search')
        sizer.Add(search_btn, pos=(11, 0), flag=wx.LEFT|wx.TOP, border=10)
        # search_btn.Bind(wx.EVT_BUTTON, self.on_press_start_routine)
        search_btn.Bind(wx.EVT_BUTTON, self.on_press)



        '''Section 4 '''
        # sizer.Add(wx.StaticText(panel, label="Search Result:"), pos=(12, 0), flag=wx.TOP|wx.LEFT, border=10)
        # sizer.Add(wx.StaticText(panel, label="No result"), pos=(13, 0), flag=wx.TOP|wx.LEFT|wx.BOTTOM, border=10)
        # sizer.Add(wx.StaticLine(panel), pos=(14, 0), span=(1, 5),
        #     flag=wx.TOP, border=10)

        sizer.Add(wx.StaticLine(panel), pos=(12, 0), span=(1, 5),
                  flag=wx.EXPAND | wx.BOTTOM | wx.TOP, border=10)

        label_sar_adc = wx.StaticText(panel, label="AMPSE Created Designs")
        font = wx.Font(14, wx.DEFAULT, wx.NORMAL, wx.BOLD)
        label_sar_adc.SetFont(font)
        # sizer.Add(label_sar_adc, pos=(13, 0), flag=wx.LEFT | wx.BOTTOM, border=10)

        section4_description = wx.BoxSizer(wx.VERTICAL)
        section4_description.Add(label_sar_adc)
        section4_description.Add(wx.StaticText(panel, label="Click on row labels to see details"))
        section4_description.Add(wx.StaticText(panel, label="Please find netlists in ./netlist/"))
        sizer.Add(section4_description, pos=(13, 0), flag=wx.LEFT | wx.BOTTOM, border=10)


        self.text_results = wx.TextCtrl(panel, style=wx.TE_MULTILINE | wx.TE_READONLY, size=(600,50))
        self.text_results.SetValue("Results will show here\n\n")
        sizer.Add(self.text_results, pos=(15,0), span=(1,5))
        sizer.AddGrowableCol(2)


        self.results_grid = grid.Grid(panel)
        self.results_grid.CreateGrid(0,4)
        self.results_grid.EnableEditing(False)
        self.results_grid.SetColLabelValue(0, "Frequency")
        self.results_grid.SetColLabelValue(1, "NBits")
        self.results_grid.SetColLabelValue(2, "Power")
        self.results_grid.SetColLabelValue(3, "Tags")

        self.results_grid.Bind(wx.grid.EVT_GRID_LABEL_LEFT_CLICK, self.OnShowDetails)

        sizer.Add(self.results_grid, pos=(16,0), span=(1,5))

        panel.SetSizer(sizer)
        sizer.Fit(self)
        self.Show()
        self.sizer = sizer

    def on_press_start_routine(self, event):
        thread = threading.Thread(target=self.on_press)
        thread.start()

        # p = Process(target=self.on_press)
        # p.start()



    def OnShowKGD(self, event):
        frame = SecondFrame("KGD", KGD_path, (1040,450))
        frame.Show()

    def OnShowCircuit(self, event):
        frame = SecondFrame("Circuit Graph", circuit_graph_path, (1040,450))
        frame.Show()

    def OnShowRegion(self, event):
        frame = SecondFrame("Suggested Region", suggested_region_path, (1040,670))
        frame.Show()

    def OnShowDetails(self, event):
        if (event.GetRow() < len(self.design_params_dicts)):
            frame = ThirdFrame(event.GetRow(), self.design_params_dicts[event.GetRow()], self.design_metrics_dicts[event.GetRow()])
            frame.Show()



    # def on_press(self):
    def on_press(self, event):

        # Clean up table rows
        numRows = self.results_grid.GetNumberRows()
        if numRows > 0:
            self.results_grid.DeleteRows(pos=0, numRows = numRows)

        self.text_results.SetValue("Beginning Analysis\n\n")

        nbit = self.input_nbits.GetValue()
        nbit_lower_bound = self.input_nbits_lb.GetValue()
        sampling_frequency = self.input_fs.GetValue()
        fs_lower_bound = self.input_fs_lb.GetValue()
        power_upper_bound = self.input_power_ub.GetValue()

        if nbit == "":
            nbit = None
        else:
            try:
                nbit = int(nbit)
            except:
                self.text_results.AppendText("Please enter valid nbit value\n\n")
                return

        if nbit_lower_bound == "":
            nbit_lower_bound = None
        else:
            try:
                nbit_lower_bound = int(nbit_lower_bound)
            except:
                self.text_results.AppendText("Please enter valid nbit lower bound value\n\n")
                return

        if sampling_frequency == "":
            sampling_frequency = None
        else:
            try:
                sampling_frequency = float(sampling_frequency)
            except:
                self.text_results.AppendText("Please enter valid nbit value\n\n")
                return

        if fs_lower_bound == "":
            fs_lower_bound = None
        else:
            try:
                fs_lower_bound = float(fs_lower_bound)
            except:
                self.text_results.AppendText("Please enter valid nbit lower bound value\n\n")
                return

        if power_upper_bound == "":
            power_upper_bound = None
        else:
            try:
                power_upper_bound = float(power_upper_bound)
            except:
                self.text_results.AppendText("Please enter valid nbit lower bound value\n\n")
                return

        if nbit_lower_bound == None:
            if nbit == None:
                self.text_results.AppendText("No valid nbit inputs\n\n")
                return
            else:
                b1 = nbit
                b2 = nbit
        else:
            b1 = nbit_lower_bound
            b2 = 12

        if fs_lower_bound == None:
            if sampling_frequency == None:
                self.text_results.AppendText("No valid sampling frequency\n\n")
                return
            else:
                f1 = sampling_frequency
                f2 = sampling_frequency
        else:
            f1 = fs_lower_bound
            f2 = 1000e6


        found = False  # Indicate whether there are possible candidates found
        results_dict = dict()
        fs_found = []
        bit_found = []

        designs = []
        design_table_dicts = []
        self.design_params_dicts = []
        self.design_metrics_dicts = []

        for b in range(b1, b2 + 1):
            for f in range(int(f1), int(f2) + 1, int(50e6)):
                result = search(f, b, power_upper_bound)

                if result == None:
                    break
                else:
                    found = True
                    tag = f'{b}bit{f}Hz'
                    results_dict[tag] = result
                    fs_found.append(f)
                    bit_found.append(b)



                    nbits = b
                    fs = f
                    power = result['power']
                    designs.append([fs, nbits, power, ''])



                    # Prepare table of parameters for each design
                    # ----------SEQ1's graph----------
                    #    [ 'drvbuf11','drvbuf21','drvbuf31','drvdffck1','drvinv1','drvnand1','fdffck','finv','fnand','div','mdacbig']
                    # ----------SEQ2's graph----------
                    #    [ 'nor3','fck1']
                    # ----------COMP's graph----------
                    #    ['fck1','fck2','fcompinv','finnn','fninv','fpinv','fpr1','fpr2','frdynand','fttt','fnor3','frdy','mrdy','fload']
                    # ----------SARDAC's graph----------
                    #    [ 'div','mdac',   'cs', 'fthn', 'fthp',   'cp']
                    # [n_dly, d_tr]
                    params_found = result['params']
                    params = dict()
                    seq1_names = [ 'drvbuf11','drvbuf21','drvbuf31','drvdffck1','drvinv1','drvnand1','fdffck','finv','fnand','div','mdacbig']
                    seq2_names = [ 'nor3','fck1']
                    comp_names = ['fck1','fck2','fcompinv','finnn','fninv','fpinv','fpr1','fpr2','frdynand','fttt','fnor3','frdy','mrdy','fload']
                    sar_dac_names = [ 'div','mdac',   'cs', 'fthn', 'fthp',   'cp']

                    params['Sequential_1'] = dict()
                    params['Sequential_2'] = dict()
                    params['Comparator'] = dict()
                    params['SAR_DAC'] = dict()

                    for i in range(len(seq1_names)):
                        params['Sequential_1'][seq1_names[i]] = params_found[0][i]

                    for i in range(len(seq2_names)):
                        params['Sequential_2'][seq1_names[i]] = params_found[1][i]

                    for i in range(len(comp_names)):
                        params['Comparator'][comp_names[i]] = params_found[2][i]

                    for i in range(len(sar_dac_names)):
                        params['SAR_DAC'][sar_dac_names[i]] = params_found[3][i]

                    params['n_dly'] = params_found[4][0]
                    params['d_tr'] = params_found[4][1]

                    self.design_params_dicts.append(params)

                    # Prepare table of metrics for each design
                    # 0- Delay of DAC vs delay of the loop +
                    # 1- loop delay less than fs  +

                    # 2- SNR more than 6*nbit + 11.76
                    # 4- Track and Hold Law +
                    # 5- Track and Hold Law +
                    # 6- Track and Hold Law +
                    # 7- Track and Hold Law +
                    # 8- vomin to be less
                    # 9- Power consumption -
                    metrics_found = result['specs']
                    metrics = dict()
                    metrics['Delay of DAC - delay of the loop (s)'] = metrics_found[0]
                    metrics['loop delay(s)'] = metrics_found[1]
                    metrics['SNR(dB)'] = metrics_found[2]
                    metrics['Track and Hold Law 1'] = metrics_found[3]
                    metrics['Track and Hold Law 2'] = metrics_found[4]
                    metrics['Track and Hold Law 3'] = metrics_found[5]
                    metrics['Track and Hold Law 4'] = metrics_found[6]
                    metrics['Vomin(V)'] = metrics_found[7]
                    metrics['Power(W)'] = metrics_found[8]
                    self.design_metrics_dicts.append(metrics)






        if found:

            # Prepare table for all designs found
            designs = np.array(designs)
            max_fs_index = np.argmax(designs[:,0])
            max_bit_index = np.argmax(designs[:,1])
            min_power_index = np.argmin(designs[:,2])

            for i in range(designs.shape[0]):
                tag = ''
                if i == max_fs_index:
                    tag += 'Maximum sampling frequency design, '

                if i == max_bit_index:
                    tag += 'Maximum ENOB design, '

                if i == min_power_index:
                    tag += 'Minimum power design'

                design_dict = dict()
                design_dict['fs'] = designs[i, 0]
                design_dict['nbits'] = designs[i,1]
                design_dict['power'] = designs[i,2]
                design_dict['tag'] = tag
                design_table_dicts.append(design_dict)

            self.results_grid.AppendRows(len(design_table_dicts))
            row_index: int = 0
            for design_ops in design_table_dicts:
                self.results_grid.SetRowLabelValue(row_index, "Design %d" % (row_index + 1))
                self.results_grid.SetCellValue(row_index, 0, str(design_ops["fs"]))
                self.results_grid.SetCellValue(row_index, 1, str(design_ops["nbits"]))
                self.results_grid.SetCellValue(row_index, 2, str(design_ops["power"]))
                self.results_grid.SetCellValue(row_index, 3, str(design_ops["tag"]))
                row_index += 1
            self.results_grid.AutoSize()
            self.sizer.Layout()
            self.sizer.Fit(self)

            savename = "%s_%s_%s_%s" % (nbit, nbit_lower_bound, sampling_frequency, fs_lower_bound)
            with open('results/global_opt/%s.json' % (savename), 'w') as f:
                json.dump(results_dict, f)


            for i in range(len(fs_found)):
                self.text_results.AppendText(f'Possible designs at {bit_found[i]} bits and {fs_found[i]}Hz\n')
            self.text_results.AppendText("Results are saved in ./results/global_opt/%s.json\n" % (savename))
        else:
            self.text_results.AppendText("No possible designs found\n\n")


if __name__ == '__main__':

    app = wx.App()
    frame = MyFrame()
    app.MainLoop()

    # search(500e6, 3, None)
