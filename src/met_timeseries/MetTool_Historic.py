import wx
import wx.grid
import wx.lib.intctrl
from wx.adv import DatePickerCtrl, DP_DROPDOWN
from wx.lib.scrolledpanel import ScrolledPanel
from datetime import datetime, date, timedelta
from dateutil.relativedelta import relativedelta
from matplotlib.figure import Figure
from matplotlib.backends.backend_wxagg import FigureCanvasWxAgg as FigureCanvas
from matplotlib.backends.backend_wx import NavigationToolbar2Wx as NavigationToolbar
import matplotlib.dates as mdates
from osgeo import gdal, ogr
from http.cookiejar import CookieJar
from netCDF4 import Dataset
import geopandas as gpd
import pandas as pd
import numpy as np
import os
import math
import zipfile
import ftplib
import urllib.request
import urllib.error
from wdmtoolbox import wdmtoolbox
from ambhas.rain_disagg import RainDisagg
import random
from scipy.stats import poisson
from scipy.stats import bernoulli
import warnings
from pyproj import Proj, transform, datadir
import cftime
import sys
import traceback

warnings.filterwarnings("ignore")

#try:
#    ref_location = sys._MEIPASS
#except Exception:
#    ref_location = os.path.abspath(".")

ref_location = os.getcwd()
BBOX_REF = os.path.join(ref_location,"GIS","EXTENT.shp")
NLDAS_REF = os.path.join(ref_location,"GIS","NLDAS_reference_grid.shp")
PRISM_REF = os.path.join(ref_location,"GIS","PRISM_reference_grid.shp")
NARR_REF = os.path.join(ref_location,"GIS","NARR_reference_grid.shp")

class GUI(wx.Frame):
    def __init__(self,*args,**kwargs):
        super(GUI,self).__init__(*args,**kwargs)
        self.initGUI()
        self.Bind(wx.EVT_CLOSE, self.OnCloseWindow)

    def OnCloseWindow(self,event):
        self.Destroy()

    def initGUI(self):
        super(GUI,self).__init__(parent=None,title='Gridded Meteorological Time-Series Development Tool',style=wx.DEFAULT_FRAME_STYLE)
        self.Maximize(True)
        self.SetIcon(wx.Icon(os.path.join(ref_location,"mpca-logo.ico")))
        self.SetBackgroundColour(wx.Colour(255,255,255))
        self.panel_ = ScrolledPanel(self,style=wx.SIMPLE_BORDER)
        self.panel_.SetBackgroundColour(wx.Colour(255,255,255))
        self.panel_.SetupScrolling(scroll_x=True,scroll_y=True)
        self.status_bar = self.CreateStatusBar()        
        self.status_bar.SetFieldsCount(2)
        self.status_bar.SetStatusWidths([-1,100])
        self.status_bar.SetStatusText('Version: 1.1.1.1',1)
        self.status_bar.SetStatusText('Ready')

        self.hbox = wx.BoxSizer(wx.HORIZONTAL)
        self.fgs_parent = wx.FlexGridSizer(0,1,25,25)
        self.fgs_child_1 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_2 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_3 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_4 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_5 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_6 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_7 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_8 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_9 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_10 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_11 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_12 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_13 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_14 = wx.FlexGridSizer(1,10,20,20)
        self.fgs_child_15 = wx.FlexGridSizer(1,1,20,20)
        self.fgs_child_16 = wx.FlexGridSizer(1,1,20,20)

        # font used throughout GUI
        self.font_ = wx.Font(10,wx.MODERN,wx.NORMAL,wx.NORMAL)
        self.font_.SetFaceName("ARIAL")

        # table for DSN entry by weather region and tstype
        self.data_table = wx.grid.Grid(self.panel_)
        self.data_table.CreateGrid(1,7)
        self.var_list_long = ['Precipitation','Air Temperature','Solar Radiation','Wind Speed','Dewpoint Temperature','Cloud Cover','Potential Evaporation']
        for i,val in enumerate(self.var_list_long):
            self.data_table.SetColLabelValue(i,val)
        self.data_table.SetRowLabelValue(0,'TSTYPE')
        self.var_list_short = ['PREC','ATEM','SOLR','WIND','DEWP','CLOU','PEVT']        
        for i,val in enumerate(self.var_list_short):
            self.data_table.SetCellValue(0,i,val)            
        self.data_table.SetRowLabelAlignment(wx.ALIGN_LEFT,wx.ALIGN_CENTRE)
        self.data_table.SetLabelFont(self.font_.Bold())
        self.data_table.SetDefaultCellFont(self.font_)
        self.data_table.SetDefaultCellAlignment(wx.ALIGN_CENTRE, wx.ALIGN_CENTRE)
        self.data_table.AutoSize()

        # browse for working directory
        self.working_dir_caption = wx.StaticText(self.panel_,wx.ID_ANY,'Working Directory')
        self.working_dir_caption.SetFont(self.font_)
        self.working_dir_path = wx.TextCtrl(self.panel_,style=wx.TE_READONLY)
        self.working_dir_path.SetBackgroundColour((240,240,240))
        self.working_dir_path.SetFont(self.font_)
        #self.working_dir_path_default = os.path.join(os.environ['USERPROFILE'],'Documents','MetToolWorkingDirectory')
        #self.working_dir_path.SetValue(self.working_dir_path_default)
        self.working_dir_bttn = wx.Button(self.panel_,label='Browse')
        # bind event to browse for working directory
        args = ["WORKING_DIR"]
        self.working_dir_bttn.Bind(wx.EVT_BUTTON,lambda event,arg=args:self.dir_bttn_click(event,arg))
        self.working_dir_bttn.SetFont(self.font_)
        # add to panel
        self.fgs_child_1.AddMany([(self.working_dir_caption,0,wx.ALIGN_CENTER_VERTICAL),(self.working_dir_path,0,wx.EXPAND),(self.working_dir_bttn,0,wx.EXPAND)])
        self.fgs_child_1.AddGrowableCol(1, 0)
        self.fgs_parent.Add(self.fgs_child_1,1,wx.EXPAND)

        # browse for HSPF uci file
        self.hspf_model_caption = wx.StaticText(self.panel_,wx.ID_ANY,'HSPF Model (*.uci)')
        self.hspf_model_caption.SetFont(self.font_)
        self.hspf_model_path = wx.TextCtrl(self.panel_,style=wx.TE_READONLY)
        self.hspf_model_path.SetBackgroundColour((240,240,240))
        self.hspf_model_path.SetFont(self.font_)
        self.hspf_model_bttn = wx.Button(self.panel_,label='Browse')
        self.hspf_model_bttn.SetFont(self.font_)
        # bind event to browse for HSPF uci file
        args = ["UCI"]
        self.hspf_model_bttn.Bind(wx.EVT_BUTTON, lambda event,arg=args:self.file_bttn_click(event,arg))
        # add to panel
        self.fgs_child_2.AddMany([(self.hspf_model_caption,0,wx.ALIGN_CENTER_VERTICAL),(self.hspf_model_path,0,wx.EXPAND),(self.hspf_model_bttn,0,wx.EXPAND)])
        self.fgs_child_2.AddGrowableCol(1, 0)
        self.fgs_parent.Add(self.fgs_child_2,1,wx.EXPAND)

        # browse for weather region shapefile
        self.wrg_caption = wx.StaticText(self.panel_, wx.ID_ANY, 'Weather Region Shapefile (*.shp)')
        self.wrg_caption.SetFont(self.font_)
        self.wrg_path = wx.TextCtrl(self.panel_,style=wx.TE_READONLY)
        self.wrg_path.SetBackgroundColour((240,240,240))
        self.wrg_path.SetFont(self.font_)
        self.wrg_bttn = wx.Button(self.panel_,label='Browse')
        self.wrg_bttn.SetFont(self.font_)
        # Bind event to browse for weather region shapefile
        args = []
        self.wrg_bttn.Bind(wx.EVT_BUTTON,lambda event,arg=args:self.wrg_bttn_click(event,arg))
        #self.map_bttn = wx.Button(self.panel_,label='Open Map')
        #self.map_bttn.SetFont(self.font_)
        #self.map_bttn.Disable()
        #self.fgs_child_3.AddMany([(self.wrg_caption),(self.wrg_path,1,wx.EXPAND),(self.wrg_bttn),(self.map_bttn)])
        #self.fgs_child_3.AddGrowableCol(1, 0)
        #self.fgs_parent.Add(self.fgs_child_3,1,wx.EXPAND)

        # select weather region field
        self.wrg_field_caption = wx.StaticText(self.panel_, wx.ID_ANY, 'Weather Region ID Field')
        self.wrg_field_caption.SetFont(self.font_)
        self.wrg_field_list = []
        self.wrg_field = wx.ComboBox(self.panel_,wx.ID_ANY,choices=self.wrg_field_list,style=wx.CB_READONLY)
        self.wrg_field.SetFont(self.font_)
        # bind event to wrg_field to add rows to existing table
        args = []
        self.wrg_field.Bind(wx.EVT_COMBOBOX,lambda event,arg=args:self.field_select(event,arg))
        self.wrg_field.SetFont(self.font_)
        # add to panel
        self.fgs_child_4.AddMany([(self.wrg_caption,0,wx.ALIGN_CENTER_VERTICAL),(self.wrg_path,0,wx.EXPAND),(self.wrg_bttn,0,wx.EXPAND),(self.wrg_field_caption,0,wx.ALIGN_CENTER_VERTICAL),(self.wrg_field,0,wx.EXPAND)])
        self.fgs_child_4.AddGrowableCol(1, 4)
        self.fgs_child_4.AddGrowableCol(4, 1)
        self.fgs_parent.Add(self.fgs_child_4,1,wx.EXPAND)

        # date range set to 1/1/1980 to last date of month seven months from today
        last_date = date.today().replace(day=1) - timedelta(days=1) - relativedelta(months=6)
        last_date = wx.DateTime.FromDMY(last_date.day,last_date.month-1,last_date.year)
        # start date (initially 1/1/1995)
        self.start_date = DatePickerCtrl(self.panel_,wx.ID_ANY,style=DP_DROPDOWN)
        self.start_date_caption = wx.StaticText(self.panel_, wx.ID_ANY, 'Start Date')
        self.start_date_caption.SetFont(self.font_)
        self.start_date.SetRange(wx.DateTime.FromDMY(1,0,1980),last_date)
        self.start_date.SetValue(wx.DateTime.FromDMY(1,0,1995))
        self.start_date.SetFont(self.font_)
        #self.fgs_child_5.AddMany([(self.start_date_caption),(self.start_date)])
        #self.fgs_parent.Add(self.fgs_child_5,1,wx.EXPAND)

        # end date (initially last date of date range)
        self.end_date = DatePickerCtrl(self.panel_,wx.ID_ANY,style=DP_DROPDOWN)
        self.end_date_caption = wx.StaticText(self.panel_, wx.ID_ANY, 'End Date')
        self.end_date_caption.SetFont(self.font_)
        self.end_date.SetRange(wx.DateTime.FromDMY(1,0,1980),last_date)
        #self.end_date.SetValue(wx.DateTime.FromDMY(1,1,1995))
        self.end_date.SetValue(last_date)
        self.end_date.SetFont(self.font_)

        # select timezone
        self.tz_text = wx.StaticText(self.panel_, wx.ID_ANY, 'Time Zone')
        self.tz_text.SetFont(self.font_)
        self.tz_select = wx.ComboBox(self.panel_,wx.ID_ANY,choices=["EASTERN","CENTRAL","MOUNTAIN","PACIFIC"],style=wx.CB_READONLY)
        self.tz_select.SetValue("CENTRAL")
        self.tz_select.SetFont(self.font_)
        self.tz_select.Disable()
        tz_dict = {"EASTERN":5,"CENTRAL":6,"MOUNTAIN":7,"PACIFIC":8}
        self.tz_offset = tz_dict[self.tz_select.GetValue()]
        # add to panel
        self.fgs_child_6.AddMany([(self.start_date_caption,0,wx.ALIGN_CENTER_VERTICAL),(self.start_date,0,wx.EXPAND),(self.end_date_caption,0,wx.ALIGN_CENTER_VERTICAL),(self.end_date,0,wx.EXPAND),(self.tz_text,0,wx.ALIGN_CENTER_VERTICAL),(self.tz_select,0,wx.EXPAND)])
        self.fgs_parent.Add(self.fgs_child_6,1,wx.EXPAND)

        # local PRISM repository
        self.prism_caption = wx.StaticText(self.panel_, wx.ID_ANY, 'Local PRISM Repository')
        self.prism_caption.SetFont(self.font_)
        self.prism_cb = wx.CheckBox(self.panel_,wx.ID_ANY)
        self.prism_cb.SetValue(wx.CHK_UNCHECKED)
        # bind event to PRISM local repository checkbox
        args = ["PRISM"]
        self.prism_cb.Bind(wx.EVT_CHECKBOX,lambda event,arg=args:self.cb_click(event,arg))
        self.prism_rep_dir = wx.TextCtrl(self.panel_,style=wx.TE_READONLY)
        self.prism_rep_dir.SetBackgroundColour((240,240,240))
        self.prism_rep_dir.SetFont(self.font_)
        self.prism_rep_dir.Disable()
        self.prism_bttn = wx.Button(self.panel_,label='Browse')
        self.prism_bttn.SetFont(self.font_)
        self.prism_bttn.Disable()
        # bind event to browse for PRISM local repocitory
        args = ["PRISM"]
        self.prism_bttn.Bind(wx.EVT_BUTTON, lambda event,arg=args:self.dir_bttn_click(event,arg))
        # add to panel
        self.fgs_child_7.AddMany([(self.prism_caption,0,wx.ALIGN_CENTER_VERTICAL),(self.prism_cb,0,wx.ALIGN_CENTER_VERTICAL),(self.prism_rep_dir,0,wx.EXPAND),(self.prism_bttn,0,wx.EXPAND)])
        self.fgs_child_7.AddGrowableCol(2, 0)
        self.fgs_parent.Add(self.fgs_child_7,1,wx.EXPAND)

        # local NLDAS repository
        self.nldas_caption = wx.StaticText(self.panel_, wx.ID_ANY, 'Local NLDAS Repository')
        self.nldas_caption.SetFont(self.font_)
        self.nldas_cb = wx.CheckBox(self.panel_,wx.ID_ANY)
        self.nldas_cb.SetValue(wx.CHK_UNCHECKED)
       # bind event to NLDAS local repocitory checkbox
        args = ["NLDAS"]
        self.nldas_cb.Bind(wx.EVT_CHECKBOX,lambda event,arg=args:self.cb_click(event,arg))
        self.nldas_rep_dir = wx.TextCtrl(self.panel_,style=wx.TE_READONLY)
        self.nldas_rep_dir.SetBackgroundColour((240,240,240))
        self.nldas_rep_dir.SetFont(self.font_)
        self.nldas_rep_dir.Disable()
        self.nldas_bttn = wx.Button(self.panel_,label='Browse')
        self.nldas_bttn.SetFont(self.font_)
        self.nldas_bttn.Disable()
        # bind event to browse for NLDAS local repocitory
        args = ["NLDAS"]
        self.nldas_bttn.Bind(wx.EVT_BUTTON, lambda event,arg=args:self.dir_bttn_click(event,arg))
        # add to panel
        self.fgs_child_8.AddMany([(self.nldas_caption,0,wx.ALIGN_CENTER_VERTICAL),(self.nldas_cb,0,wx.ALIGN_CENTER_VERTICAL),(self.nldas_rep_dir,0,wx.EXPAND),(self.nldas_bttn,0,wx.EXPAND)])
        self.fgs_child_8.AddGrowableCol(2, 0)
        self.fgs_parent.Add(self.fgs_child_8,1,wx.EXPAND)

        # local NARR repository
        self.narr_caption = wx.StaticText(self.panel_, wx.ID_ANY, 'Local NARR Repository')
        self.narr_caption.SetFont(self.font_)
        self.narr_cb = wx.CheckBox(self.panel_,wx.ID_ANY)
        self.narr_cb.SetValue(wx.CHK_UNCHECKED)
        # bind event to NARR local repocitory checkbox
        args = ["NARR"]
        self.narr_cb.Bind(wx.EVT_CHECKBOX,lambda event,arg=args:self.cb_click(event,arg))
        self.narr_rep_dir = wx.TextCtrl(self.panel_,style=wx.TE_READONLY)
        self.narr_rep_dir.SetBackgroundColour((240,240,240))
        self.narr_rep_dir.SetFont(self.font_)
        self.narr_rep_dir.Disable()
        self.narr_bttn = wx.Button(self.panel_,label='Browse')
        self.narr_bttn.SetFont(self.font_)
        self.narr_bttn.Disable()
        # bind event to browse for NARR local repocitory
        args = ["NARR"]
        self.narr_bttn.Bind(wx.EVT_BUTTON, lambda event,arg=args:self.dir_bttn_click(event,arg))
        # add to panel
        self.fgs_child_9.AddMany([(self.narr_caption,0,wx.ALIGN_CENTER_VERTICAL),(self.narr_cb,0,wx.ALIGN_CENTER_VERTICAL),(self.narr_rep_dir,0,wx.EXPAND),(self.narr_bttn,0,wx.EXPAND)])
        self.fgs_child_9.AddGrowableCol(2, 0)
        self.fgs_parent.Add(self.fgs_child_9,1,wx.EXPAND)

        # add TSTYPE and DSN table to panel
        self.fgs_child_10.AddMany([(self.data_table)])
        self.fgs_parent.Add(self.fgs_child_10,1,wx.EXPAND)

        # download and process data
        self.dwnld_prcs_bttn = wx.Button(self.panel_,label='Download and Process Data')
        self.dwnld_prcs_bttn.SetFont(self.font_.Bold())
        args = []
        # bind event to download and process button
        self.dwnld_prcs_bttn.Bind(wx.EVT_BUTTON, lambda event,arg=args:self.dwnld_prcs_bttn_click(event,arg))
        # add to panel
        self.fgs_child_11.Add(self.dwnld_prcs_bttn)
        self.fgs_parent.Add(self.fgs_child_11,1,wx.EXPAND)

        #hline = wx.StaticLine(self.panel_,wx.ID_ANY,size=wx.DefaultSize,style=wx.LI_HORIZONTAL)
        #self.fgs_child_100 = wx.FlexGridSizer(1,1,20,20)
        #self.fgs_child_100.Add(hline)
        #self.fgs_parent.Add(self.fgs_child_100,1,wx.EXPAND|wx.ALL)

        # select WDM file
        self.wdm_caption = wx.StaticText(self.panel_,wx.ID_ANY,'WDM Filename (*.wdm)')
        self.wdm_caption.SetFont(self.font_)
        self.wdm_path = wx.TextCtrl(self.panel_,style=wx.TE_READONLY)
        self.wdm_path.SetBackgroundColour((240,240,240))
        self.wdm_path.SetFont(self.font_)
        self.wdm_bttn = wx.Button(self.panel_,label='Browse')
        self.wdm_bttn.SetFont(self.font_)
        # bind event to browse for WDM button
        args = ["WDM"]
        self.wdm_bttn.Bind(wx.EVT_BUTTON, lambda event,arg=args:self.file_bttn_click(event,arg))
        # add to panel
        self.fgs_child_12.AddMany([(self.wdm_caption,0,wx.ALIGN_CENTER_VERTICAL),(self.wdm_path,0,wx.EXPAND),(self.wdm_bttn,0,wx.EXPAND)])
        self.fgs_child_12.AddGrowableCol(1,0)
        self.fgs_parent.Add(self.fgs_child_12,1,wx.EXPAND)

        # write to WDM file
        self.wdm_write_bttn = wx.Button(self.panel_,label='Write to WDM File')
        self.wdm_write_bttn.SetFont(self.font_.Bold())
        # bind event to write to WDM button
        args = []
        self.wdm_write_bttn.Bind(wx.EVT_BUTTON, lambda event,arg=args:self.wdm_write_bttn_click(event,arg))
        # overwrite or append if DSN exists
        self.dsn_action_text = wx.StaticText(self.panel_,wx.ID_ANY,"If DSN exists in WDM file,")
        #self.dsn_action_text = wx.StaticText(self.panel_,wx.ID_ANY,"")
        self.dsn_action_text.SetFont(self.font_)
        self.dsn_action = wx.ComboBox(self.panel_,wx.ID_ANY,choices=["overwrite existing DSN","append to existing DSN"],style=wx.CB_READONLY)
        self.dsn_action.SetFont(self.font_)
        self.dsn_action.SetValue("overwrite existing DSN")
        self.dummy_text = wx.StaticText(self.panel_,wx.ID_ANY,"")
        self.dummy_text.SetFont(self.font_)
        # add to panel
        #self.fgs_child_13.AddMany([(self.wdm_write_bttn,0,wx.EXPAND),(self.dsn_action_text,0,wx.ALIGN_CENTER_VERTICAL),(self.dsn_action,0,wx.EXPAND)])
        self.fgs_child_13.AddMany([(self.wdm_write_bttn,0,wx.EXPAND),(self.dsn_action_text,0,wx.ALIGN_CENTER_VERTICAL),(self.dsn_action,0,wx.EXPAND),(self.dummy_text,0,wx.EXPAND|wx.ALIGN_CENTER_VERTICAL)])        
        self.fgs_child_13.AddGrowableCol(2,1)
        self.fgs_child_13.AddGrowableCol(3,8)
        self.fgs_parent.Add(self.fgs_child_13,1,wx.EXPAND)

        # list dsns
        self.dsn_select_caption = wx.StaticText(self.panel_, wx.ID_ANY, 'Time-Series to Plot')
        self.dsn_select_caption.SetFont(self.font_)
        self.dsn_list = []
        self.dsn_select = wx.ComboBox(self.panel_,wx.ID_ANY,choices=self.dsn_list,style=wx.CB_READONLY)
        self.dsn_select.SetFont(self.font_)
        # dsn time-step
        self.dsn_ts_caption = wx.StaticText(self.panel_, wx.ID_ANY, 'Time-Step for Plot')
        self.dsn_ts_caption.SetFont(self.font_)
        self.dsn_ts = wx.ComboBox(self.panel_,wx.ID_ANY,choices=["HOURLY","DAILY","MONTHLY","YEARLY"],style=wx.CB_READONLY)
        self.dsn_ts.SetValue("HOURLY")
        # dsn statistic
        self.dsn_stat_caption = wx.StaticText(self.panel_, wx.ID_ANY, 'Statistic')
        self.dsn_stat_caption.SetFont(self.font_)
        self.dsn_stat = wx.ComboBox(self.panel_,wx.ID_ANY,choices=["SUM","AVG","MAX","MIN"],style=wx.CB_READONLY)
        self.dsn_stat.SetValue("Sum")
        # plot dsn button
        self.plot_dsn_bttn = wx.Button(self.panel_,label='Plot Selected Time-Series')
        self.plot_dsn_bttn.SetFont(self.font_.Bold())
        # bind event to plot dsn button
        args = ["PLOT"]
        self.plot_dsn_bttn.Bind(wx.EVT_BUTTON,lambda event,arg=args:self.plot_or_export_dsn_bttn_click(event,arg))
        # export to text file
        self.to_txt_bttn = wx.Button(self.panel_,label='Export Selected Time-Series to TXT File')
        # bind event to plot dsn button
        args = ["EXPORT"]
        self.to_txt_bttn.Bind(wx.EVT_BUTTON,lambda event,arg=args:self.plot_or_export_dsn_bttn_click(event,arg))
        #self.to_txt_bttn = wx.StaticText(self.panel_, wx.ID_ANY, "")
        self.to_txt_bttn.SetFont(self.font_.Bold())
        # add to panel
        self.fgs_child_14.AddMany([(self.dsn_select_caption,0,wx.ALIGN_CENTER_VERTICAL),(self.dsn_select,0,wx.EXPAND),(self.dsn_ts_caption,0,wx.ALIGN_CENTER_VERTICAL),(self.dsn_ts,0,wx.EXPAND),(self.dsn_stat_caption,0,wx.ALIGN_CENTER_VERTICAL),(self.dsn_stat,0,wx.EXPAND),(self.plot_dsn_bttn,0,wx.EXPAND),(self.to_txt_bttn,0,wx.EXPAND)])
        self.fgs_child_14.AddGrowableCol(1, 5)
        self.fgs_child_14.AddGrowableCol(3, 1)
        self.fgs_child_14.AddGrowableCol(5, 1)
        self.fgs_parent.Add(self.fgs_child_14,1,wx.EXPAND)

        # plot dsn
        self.figr = Figure(facecolor="#f0f0f0",dpi=150)
        self.cnvs = FigureCanvas(self.panel_,wx.ID_ANY,self.figr)
        self.toolbar = NavigationToolbar(self.cnvs)
        self.fgs_child_15.Add(self.cnvs,1,wx.EXPAND|wx.ALL)
        self.fgs_child_15.AddGrowableCol(0, 0)
        self.fgs_child_15.AddGrowableRow(0, 0)
        self.fgs_parent.Add(self.fgs_child_15,1,wx.EXPAND|wx.ALL)
        self.fgs_child_16.Add(self.toolbar)
        self.fgs_parent.Add(self.fgs_child_16,1,wx.EXPAND)

        self.fgs_parent.AddGrowableCol(0,1)

        self.hbox.Add(self.fgs_parent, proportion=1, flag=wx.ALL|wx.EXPAND, border=10)
        self.panel_.SetSizer(self.hbox)

    def dir_bttn_click(self,event,arg):
        try:
            txt = arg[0]
            if txt == "WORKING_DIR":
                dlg = wx.DirDialog(self.panel_,"Choose working directory","",wx.DD_DEFAULT_STYLE)
            if txt == "PRISM" or txt == "NLDAS" or txt == "NARR":
                dlg = wx.DirDialog(self.panel_,"Choose working directory", "",wx.DD_DEFAULT_STYLE|wx.DD_DIR_MUST_EXIST)
            if dlg.ShowModal() == wx.ID_OK:
                if txt == "WORKING_DIR":
                    self.working_dir_path.SetValue(dlg.GetPath())
                    # create tool run log file
                    self.log_file_path = os.path.join(self.working_dir_path.GetValue(),"tool_log.txt")
                    log_file = open(self.log_file_path,"w")
                    log_file.close()
                    self.prism_dwnld_dir = os.path.join(self.working_dir_path.GetValue(),"prism_data")
                    if not os.path.exists(self.prism_dwnld_dir): os.mkdir(self.prism_dwnld_dir)
                    self.nldas_dwnld_dir = os.path.join(self.working_dir_path.GetValue(),"nldas_data")
                    if not os.path.exists(self.nldas_dwnld_dir): os.mkdir(self.nldas_dwnld_dir)
                    self.narr_dwnld_dir = os.path.join(self.working_dir_path.GetValue(),"narr_data")
                    if not os.path.exists(self.narr_dwnld_dir): os.mkdir(self.narr_dwnld_dir)
                    self.gis_dir = os.path.join(self.working_dir_path.GetValue(),"gis")
                    if not os.path.exists(self.gis_dir): os.mkdir(self.gis_dir)
                    self.out_dir = os.path.join(self.working_dir_path.GetValue(),"output")
                    if not os.path.exists(self.out_dir): os.mkdir(self.out_dir)
                if txt == "PRISM": self.prism_rep_dir.SetValue(dlg.GetPath())
                if txt == "NLDAS": self.nldas_rep_dir.SetValue(dlg.GetPath())
                if txt == "NARR": self.narr_rep_dir.SetValue(dlg.GetPath())
                dlg.Destroy()
        except Exception as e:
            print_text = "{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return

    def file_bttn_click(self,event,arg):
        try:
            txt = arg[0]
            if txt == "UCI":
                dlg = wx.FileDialog(self.panel_,"Open HSPF model",wildcard="HSPF Model User Control Input File (*.uci)|*.uci",style=wx.FD_OPEN|wx.FD_FILE_MUST_EXIST)
            if txt == "WDM":
                dlg = wx.FileDialog(self.panel_,"Open WDM file",wildcard="HSPF Model Watershed Data Management File (*.wdm)|*.wdm",style=wx.FD_OPEN)
            if dlg.ShowModal() == wx.ID_OK:
                if txt == "UCI": self.hspf_model_path.SetValue(dlg.GetPath())
                if txt == "WDM":
                    self.wdm_path.SetValue(dlg.GetPath())
                    self.get_wdm_dsns(self.wdm_path.GetValue())
                dlg.Destroy()
        except Exception as e:
            print_text = "{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return

    def wrg_bttn_click(self,event,arg):
        try:
            dlg = wx.FileDialog(self.panel_,"Open weather region shapefile",wildcard="Polygon Shapefile (*.shp)|*.shp",style=wx.FD_OPEN | wx.FD_FILE_MUST_EXIST)
            if dlg.ShowModal() == wx.ID_OK:                
                self.wrg_path.SetValue(dlg.GetPath())
                dlg.Destroy()
                weather_regions = gpd.read_file(self.wrg_path.GetValue())                
                self.wrg_field_list = []
                for col in weather_regions.columns:
                    if col == "GRIDCODE" or col == "ROW" or col == "COL":
                        print_text = "Weather region shapefile cannot have GRIDCODE, ROW or COL as field names"
                        self.update_log_file(print_text)
                        wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
                        return
                    if weather_regions[col].dtype in ['uint8','uint16','uint32','uint64','int8','int16','int32','int64']:
                        # don't append field name if negative values present or values are not unique
                        minval = 0
                        if weather_regions[col].dtype in ['int8','int16','int32','int64']: minval = weather_regions[col].min()
                        if minval >= 0 and weather_regions[col].is_unique == True: self.wrg_field_list.append(col)
                self.wrg_field.Clear()
                self.wrg_field.AppendItems(self.wrg_field_list)
        except Exception as e:
            print_text = "{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return

    def field_select(self,event,arg):
        try:
            if self.wrg_path.GetValue() != "" and self.wrg_field.GetValue() != "":
                nrows = self.data_table.GetNumberRows()
                if nrows > 1: self.data_table.DeleteRows(pos=1,numRows=nrows-1)
                df = gpd.read_file(self.wrg_path.GetValue())
                nrows = len(df[self.wrg_field.GetValue()])
                self.data_table.AppendRows(numRows=nrows)
                for i,row in df.iterrows():
                    self.data_table.SetRowLabelValue(i+1,"WRG_"+str(row[self.wrg_field.GetValue()]))
                    for j in range(0,7):
                        self.data_table.SetCellValue(i+1,j,str((j+1)*100+(i+1)))
                self.data_table.AutoSize()
                self.Maximize(False)
                self.Maximize(True)
        except Exception as e:
            print_text = "{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return

    def cb_click(self,event,arg):
        try:
            txt = arg[0]
            if txt == "PRISM":
                cb = self.prism_cb
                dir_ = self.prism_rep_dir
                bttn = self.prism_bttn
            if txt == "NLDAS":
                cb = self.nldas_cb
                dir_ = self.nldas_rep_dir
                bttn = self.nldas_bttn
            if txt == "NARR":
                cb = self.narr_cb
                dir_ = self.narr_rep_dir
                bttn = self.narr_bttn
            if cb.GetValue() == True:
                dir_.Enable()
                bttn.Enable()
            if cb.GetValue() == False:
                dir_.Disable()
                bttn.Disable()
            self.dwnld_prcs_bttn.Enable()
        except Exception as e:
            print_text = "{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)            
            return

    def calculate_grid_wrg_fractions(self,reference_grid,weather_regions,flag):
        try:
            grid_row_col = gpd.read_file(reference_grid)
            #weather_regions = gpd.read_file(self.wrg_path.GetValue())
            weather_regions = gpd.read_file(weather_regions)
    
            # project boundary to reference grid projection
            weather_regions = weather_regions.to_crs(grid_row_col.crs)
            weather_regions['Area_by_WS'] = weather_regions['geometry'].area/ 10**6
            grid_row_col['Area'] = grid_row_col['geometry'].area/ 10**6
    
            intersect_ = gpd.overlay(grid_row_col,weather_regions,how='intersection')
            intersect_['Area'] = intersect_['geometry'].area/ 10**6
            intersect_['Fr'] = intersect_['Area']/intersect_['Area_by_WS']
            if flag == "NLDAS_BBOX":
                 df = intersect_[['GRIDCODE','Fr']]
            else:
                intersect_.sort_values([self.wrg_field.GetValue()],ascending=True,inplace=True)
                df = intersect_[[self.wrg_field.GetValue(),'GRIDCODE','Fr']]
            return df
        except Exception as e:
            print_text = "Error intersecting {} and {}\n{}\n{}".format(reference_grid,weather_regions,str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return

    def daterange(self,start_date,end_date):
        try:
            for n in range(int((end_date - start_date).days)):
                yield start_date + timedelta(n)
        except Exception as e:
            print_text = "Error generating a date range from start date and end date\n{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return

    def update_log_file(self,print_text):
        try:
            with open(self.log_file_path,"a",newline='') as log_file:
                log_file.write(print_text + "\n")
            log_file.close()
        except:
            pass

    def PanEvaporationValueComputedByPenman(self,aMinTmp,aMaxTmp,aDewTmp,aWindSp,aSolRad):
        try:
            lAirTmp = (aMinTmp + aMaxTmp) / 2.0
            if aSolRad <= 0.0: aSolRad = 0.00001
            lQNDelt = math.exp((lAirTmp - 212.0) * (0.1024 - 0.01066 * math.log(aSolRad))) - 0.0001
            lEsMiEa = (6413252.0 * math.exp(-7482.6 / (lAirTmp + 398.36))) - (6413252.0 * math.exp(-7482.6 / (aDewTmp + 398.36)))
            if lEsMiEa < 0.0: lEsMiEa = 0.0
            lEaGama = 0.0105 * (lEsMiEa ** 0.88) * (0.37 + 0.0041 * aWindSp)
            lDelta = 47987800000.0 * math.exp(-7482.6 / (lAirTmp + 398.36)) / ((lAirTmp + 398.36) ** 2)
    
            lPanEvap = (lQNDelt + lEaGama) / (lDelta + 0.0105)
            if lPanEvap < 0.0: lPanEvap = 0.0
    
            return lPanEvap
        except Exception as e:
            print_text = "Error generating daily Penman pan evaporation timeseries\n{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return

    def PETDST(self,aLatDeg,aMonth,aDay,aDayPet):
        try:
            JulDay = 30.5 * (aMonth - 1) + aDay
            LatRdn = math.radians(aLatDeg)
            Phi = LatRdn
            AD = 0.40928 * math.cos(0.0172141 * (172.0 - JulDay))
            SS = math.sin(Phi) * math.sin(AD)
            CS = math.cos(Phi) * math.cos(AD)
            X2 = -SS / CS
            Delt = 7.6394 * (1.5708 - math.atan(X2 / math.sqrt(1.0 - X2 ** 2)))
            SunR = 12.0 - Delt / 2.0
    
            DTR2 = Delt / 2.0
            DTR4 = Delt / 4.0
            CRAD = 0.66666667 / DTR2
            SL = CRAD / DTR4
            TRise = SunR
            TR2 = TRise + DTR4
            TR3 = TR2 + DTR2
            TR4 = TR3 + DTR4
    
            aHrPet = []
            for IK in range(24):
                RK = IK
                if RK > TRise:
                    if RK > TR2:
                        if RK > TR3:
                            if RK > TR4: aHrPet.append(0.0)
                            else: aHrPet.append((CRAD - (RK - TR3) * SL) * aDayPet)
                        else: aHrPet.append(CRAD * aDayPet)
                    else: aHrPet.append((RK - TRise) * SL * aDayPet)
                else: aHrPet.append(0.0)
    
            return aHrPet
        except Exception as e:
            print_text = "Error disaggregating daily Penman pan evaporation timeseries to hourly\n{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return

    def download_and_parse_nldas_data(self):
        try:
            # check if local repository has been provided
            if self.nldas_cb.GetValue() == False:
                flag = 1
            else:
                flag = 0
                
            if flag == 0:
                if self.nldas_rep_dir.GetValue() == "":
                    wx.MessageBox("NLDAS respository not provided","Error",style=wx.OK|wx.ICON_ERROR)
                    return 1
    
            # convert wx date to datetime date
            start_date = self.start_date.GetValue().Format('%d%m%y')
            start_date = datetime.strptime(start_date,'%d%m%y') - timedelta(hours=48)
            end_date = self.end_date.GetValue().Format('%d%m%y')
            end_date = datetime.strptime(end_date,'%d%m%y') + timedelta(hours=48)
    
            # project weather region boundary to reference grid projection and get min and max co-ordinates
            reference_grid = NLDAS_REF
            grid_row_col = gpd.read_file(reference_grid)
            weather_regions = gpd.read_file(self.wrg_path.GetValue())
            weather_regions_prj = weather_regions.copy()
            weather_regions_prj['geometry'] = weather_regions_prj['geometry'].to_crs(grid_row_col.crs)
            weather_regions_prj.crs = grid_row_col.crs
            # get latitudes of centroids of weather regions; required for PET disaggregation later
            lats = dict(zip('WRG_' + weather_regions_prj[self.wrg_field.GetValue()].astype(str),weather_regions_prj['geometry'].centroid.y))

            #min_lat = np.floor(weather_regions_prj.bounds['miny'].min())
            #min_long = np.floor(weather_regions_prj.bounds['minx'].min())
            #max_lat = np.ceil(weather_regions_prj.bounds['maxy'].max())
            #max_long = np.ceil(weather_regions_prj.bounds['maxx'].max())
            min_lat = 41.5
            min_long = -100.125
            max_lat = 50.0
            max_long = -87.375
            
            # generate bounding box polygon from min and max co-ordinates
            bbox = ogr.Geometry(ogr.wkbLinearRing)
            bbox.AddPoint(min_long,min_lat)
            bbox.AddPoint(max_long,min_lat)
            bbox.AddPoint(max_long,max_lat)
            bbox.AddPoint(min_long,max_lat)
            bbox.AddPoint(min_long,min_lat)
            bbox_polygon = ogr.Geometry(ogr.wkbPolygon)
            bbox_polygon.AddGeometry(bbox)
            
            shapefile = ogr.Open(NLDAS_REF)
            layer = shapefile.GetLayer(0)    
            bbox_shp = os.path.join(self.gis_dir,"nldas_bbox_prj.shp")
            bbox_driver = ogr.GetDriverByName("ESRI Shapefile")
            bbox_ds = bbox_driver.CreateDataSource(bbox_shp)
            bbox_lyr = bbox_ds.CreateLayer("nldas_bbox_prj",layer.GetSpatialRef(),ogr.wkbPolygon)
            bbox_defn = bbox_lyr.GetLayerDefn()
            feature = ogr.Feature(bbox_defn)
            feature.SetGeometry(bbox_polygon)
            bbox_lyr.CreateFeature(feature)
            feature = None
            bbox_ds = None
    
            # intersect bounding box polygon with reference grid to determine uique gridcodes
            df_intersect = self.calculate_grid_wrg_fractions(reference_grid,bbox_shp,"NLDAS_BBOX")
            df_intersect['GRIDCODE'] = df_intersect['GRIDCODE'].astype('str').str.zfill(6)
            df_intersect['Row'] = df_intersect['GRIDCODE'].str[:3]
            df_intersect['Col'] = df_intersect['GRIDCODE'].str[-3:]
            df_intersect['Row'] = df_intersect['Row'].astype(int)
            df_intersect['Col'] = df_intersect['Col'].astype(int)
            min_row = np.min(df_intersect['Row'])
            max_row = np.max(df_intersect['Row'])
            min_col = np.min(df_intersect['Col'])
            max_col = np.max(df_intersect['Col'])
    
            # create header row for dataframe
            cols = ['VAR']
            for r in range(min_row,max_row+1):
                for c in range(min_col,max_col+1):
                            cols.append(str(r*1000+c).zfill(6))
            # update tool_log.txt
            print_text = "Successfully generated bounding box for NLDAS data download and processing"
            self.update_log_file(print_text)
    
            #gdal.GetDriverByName('EHdr').Register()
            var = {1:'TMP',2:'SPFH',3:'PRES',4:'UGRD',5:'VGRD',6:'DLWRF',7:'CONVFrac',8:'CAPE',9:'PEVAP',10:'APCP',11:'DSWRF'}
            var_ = {'TMP':'air_temperature','SPFH':'SPFH','PRES':'PRES','UGRD':'UGRD','VGRD':'VGRD','CONVFrac':'CONVFrac','CAPE':'CAPE','PEVAP':'PEVAP','APCP':'APCP','DSWRF':'solar_radiation'}
            # unit conversion multipliers and constants
            m = {'TMP':1.8,'SPFH':1.0,'PRES':1.0,'UGRD':2.23694,'VGRD':2.23694,'CONVFrac':1.0,'CAPE':1.0,'PEVAP':1.0,'APCP':0.0393701,'DSWRF':0.0860437102}
            c = {'TMP':32.0,'SPFH':0.0,'PRES':0.0,'UGRD':0.0,'VGRD':0.0,'CONVFrac':0.0,'CAPE':0.0,'PEVAP':0.0,'APCP':0.0,'DSWRF':0.0}
    
            # download NLDAS data
            url = 'https://hydro1.gesdisc.eosdis.nasa.gov/daac-bin/OTF/HTTP_services.cgi?'
            username, password = 'TetraTech', 'Tetra_Tech1'
            password_manager = urllib.request.HTTPPasswordMgrWithDefaultRealm()
            password_manager.add_password(None, "https://urs.earthdata.nasa.gov", username, password)
            cookie_jar = CookieJar()
            opener = urllib.request.build_opener(
                urllib.request.HTTPBasicAuthHandler(password_manager),
                urllib.request.HTTPCookieProcessor(cookie_jar))
            urllib.request.install_opener(opener)
            
            for single_date in self.daterange(start_date,end_date):
                stamp = single_date.strftime("%j")
                year_ = single_date.strftime("%Y")
                for x in range(0, 24):
                    fin_file = "FILENAME=%2Fdata%2FNLDAS%2FNLDAS_FORA0125_H.002%2F" + year_ + "%2F" + str(stamp) + '%2F'
                    fin_file = fin_file + "NLDAS_FORA0125_H.A" + single_date.strftime("%Y%m%d") + "." + str(x).zfill(2) + "00.002.grb&FORMAT=Z3JiLw&BBOX="
                    fin_file = fin_file + str(min_lat) + '%2C' + str(min_long) + '%2C' + str(max_lat)+ '%2C' + str(max_long) + '&LABEL='
                    fin_file = fin_file + "NLDAS_FORA0125_H.A" + single_date.strftime("%Y%m%d") + "." + str(x).zfill(2) + "00.002.grb.SUB.grb"
                    fin_file = fin_file + "&SHORTNAME=NLDAS_FORA0125_H&SERVICE=L34RS_LDAS&VERSION=1.02&DATASET_VERSION=002"
                    fin_url = url + fin_file
                    file_name = "NLDAS_FORA0125_H.A" + single_date.strftime("%Y%m%d") + "." + str(x).zfill(2) + "00.002.grb"                
                    if flag == 1:
                        out_file = os.path.join(self.nldas_dwnld_dir,file_name)
                        if os.path.exists(out_file) == False:
                            request = urllib.request.Request(fin_url)
                            try:                            
                                response = urllib.request.urlopen(request)
                                body = response.read()
                                with open(out_file, 'wb') as f:
                                    f.write(body)
                                    f.close()
                                    response.close()
                                print_text = "Successfully downloaded NLDAS file {}".format(file_name)
                                self.update_log_file(print_text)
                            except Exception as e:
                                print_text = "Error downloading NLDAS file {}\n{}\n{}".format(file_name,str(e),traceback.format_exc())
                                self.update_log_file(print_text)
                                wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
                                return 1
                    if flag == 0:
                        out_file = os.path.join(self.nldas_rep_dir.GetValue(),file_name)
                    
                    try:
                        date_time = single_date.strftime("%Y-%m-%d") + " " + str(x).zfill(2) + ":00:00"
                        img = gdal.Open(out_file)
                        for i in [1,2,3,4,5,10,11]:
                            band = img.GetRasterBand(i)
                            data = band.ReadAsArray().astype(float)
                            # assumes extent of repository NLDAS files is the state of MN
                            data = data[min_row:max_row+1,min_col:max_col+1]
                            data = data.flatten()
                            temp_df = pd.DataFrame(data)
                            temp_df = temp_df.T                            
                            temp_df.insert(0,'Date_Time',date_time)
                            if single_date == start_date and x == 0:
                                with open(os.path.join(self.out_dir,var[i] + "_gridcode.txt"),"w") as data_dump:
                                    data_dump.write("Date_Time," + ",".join(x for x in cols) + "\n")
                                data_dump.close()
                            with open(os.path.join(self.out_dir,var[i] + "_gridcode.txt"),"a",newline='') as data_dump:
                                temp_df.to_csv(data_dump,index=False,header=False,float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
                            data_dump.close()
                        print_text = "Successfully processed NLDAS file {}".format(out_file)
                        self.update_log_file(print_text)
                    except Exception as e:
                        print_text = "Error occurred while processing NLDAS file {}\n{}\n{}".format(out_file,str(e),traceback.format_exc())
                        self.update_log_file(print_text)
                        wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
                        return 1
    
            # intersect wrg polygon with reference grid to determine wrg-grid fractions
            df_intersect = self.calculate_grid_wrg_fractions(reference_grid,self.wrg_path.GetValue(),"NLDAS")
            df_intersect['GRIDCODE'] = df_intersect['GRIDCODE'].astype('str').str.zfill(6)    
            # update tool_log.txt
            print_text = "Successfully intersected WRG and NLDAS reference grid"
            self.update_log_file(print_text)
            print_text = "Result table"
            self.update_log_file(("-") * len(print_text))
            self.update_log_file(print_text)
            self.update_log_file(("-") * len(print_text))
            with open(self.log_file_path,"a",newline='') as log_file:
                df_intersect.to_csv(log_file,index=True,header=True,float_format='%.6f')
            log_file.close()
            
            for j in [1,2,3,4,5,10,11]:
                df_var = pd.read_csv(os.path.join(self.out_dir,var[j] + "_gridcode.txt"),header=0,index_col="Date_Time",parse_dates=True)                
                df_var.columns = cols
                df_var.index.name = 'Date_Time'
    
                new_cols = df_intersect[self.wrg_field.GetValue()].unique()
                new_cols = ['WRG_' + str(x) for x in new_cols]
                for col in new_cols:
                    df_var[col] = 0
                for i,row in df_intersect.iterrows():
                    df_var['WRG_'+str(row[self.wrg_field.GetValue()])] += df_var[row['GRIDCODE']] * row['Fr']
    
                # convert to appropriate units
                df_var = np.multiply(df_var[new_cols],m[var[j]])+c[var[j]]
                # timezone correction
                df_var.index = df_var.index + pd.DateOffset(hours=-self.tz_offset)
    
                # update tool_log.txt
                print_text = "Successfully generated timeseries for {} aggregated spatially and corrected for timezone".format(var[j])
                self.update_log_file(print_text)
    
                # truncate to user-specified start_date (at 0:00 hours) and end_date (at 23:00 hours)
                # maintain consistency between NLDAS and PRISM precipitation time-series start and end dates
                if var[j] == "APCP":
                    df_var = df_var.truncate(before=start_date+timedelta(hours=24),after=end_date-timedelta(hours=1),axis=0)
                else:
                    df_var = df_var.truncate(before=start_date+timedelta(hours=48),after=end_date-timedelta(hours=25),axis=0)
                df_var.to_csv(os.path.join(self.out_dir,var_[var[j]] + '.txt'),index=True,header=True,float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
            
            # aggregate UGRD and VGRD
            df_ugrd = pd.read_csv(os.path.join(self.out_dir,'UGRD.txt'),index_col='Date_Time',header=0,parse_dates=True)
            df_vgrd = pd.read_csv(os.path.join(self.out_dir,'VGRD.txt'),index_col='Date_Time',header=0,parse_dates=True)
            df_wnd = np.sqrt(np.square(df_ugrd)+np.square(df_vgrd))
            # correct for measurement height at 10m to 2m
            df_wnd = np.multiply(df_wnd,np.power(0.2,0.143))
            df_wnd.to_csv(os.path.join(self.out_dir,'wind_speed.txt'),index=True,header=True,float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
            # update tool_log.txt
            print_text = "Successfully generated wind travel timeseries"
            self.update_log_file(print_text)
    
            # calculate dew point temperature
            df_spfh = pd.read_csv(os.path.join(self.out_dir,'SPFH.txt'),index_col='Date_Time',header=0,parse_dates=True)
            df_pres = pd.read_csv(os.path.join(self.out_dir,'PRES.txt'),index_col='Date_Time',header=0,parse_dates=True)
            # Pa to kPa
            df_pres = np.divide(df_pres,1000.0)
            df_tmp = pd.read_csv(os.path.join(self.out_dir,'air_temperature.txt'),index_col='Date_Time',header=0,parse_dates=True)
            df_mr = np.divide(df_spfh,1.0-df_spfh)
            df_acvp = np.multiply(np.divide(df_mr,0.622+df_mr),df_pres)
            # reference vapor pressure
            REF_VP = 0.6113
            df_dewpK = np.divide(1,1/273.15-np.multiply(0.0001844,np.log(np.divide(df_acvp,REF_VP))))
            df_dewpF = np.multiply(df_dewpK-273.15,9.0/5.0)+32.0
            #df_dewpF = np.min(df_dewpF,df_tmp)
            df_dewpF.to_csv(os.path.join(self.out_dir,'dewpoint_temperature.txt'),index=True,header=True,float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
            # update tool_log.txt
            print_text = "Successfully generated dew point temperature timeseries"
            self.update_log_file(print_text)
    
            # calculate penman-pan evaporation
            df_tmin = df_tmp.resample('D').min()
            df_tmax = df_tmp.resample('D').max()
            df_ddew = df_dewpF.resample('D').mean()
            df_dwin = pd.read_csv(os.path.join(self.out_dir,'wind_speed.txt'),index_col='Date_Time',header=0,parse_dates=True).resample('D').sum()
            df_dsol = pd.read_csv(os.path.join(self.out_dir,'solar_radiation.txt'),index_col='Date_Time',header=0,parse_dates=True).resample('D').sum()
            cols = df_tmin.columns
            df_devp = pd.DataFrame()
            for col in cols:
                df_devp['TMIN'] = df_tmin[col]
                df_devp['TMAX'] = df_tmax[col]
                df_devp['DDEW'] = df_ddew[col]
                df_devp['DWIN'] = df_dwin[col]
                df_devp['DSOL'] = df_dsol[col]
                df_devp.index.name = 'Date'
                df_devp[col] = df_devp.apply(lambda x: self.PanEvaporationValueComputedByPenman(x['TMIN'],x['TMAX'],x['DDEW'],x['DWIN'],x['DSOL']),axis=1)
            #df_devp[cols].to_csv(os.path.join(idir1,'DEVP_v2.txt'),float_format='%.6f',date_format='%Y/%m/%d',header=True)
            df_devp['month'] = df_devp.index.month
            df_devp['day'] = df_devp.index.day
            index = pd.date_range(df_devp.index[0],df_devp.index[-1] + timedelta(days=1) - timedelta(hours=1),freq='1H')
            df_pevt = pd.DataFrame(index=index,columns=cols)
            df_pevt.index.name = 'Date_Time'
            for col in cols:
                arr = df_devp.apply(lambda x: self.PETDST(lats[col],int(x['month']),int(x['day']),x[col]),axis=1).values
                arr = np.hstack(arr)
                df_pevt[col] = arr
            df_pevt.to_csv(os.path.join(self.out_dir,'potential_evaporation.txt'),index=True,header=True,float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
            # update tool_log.txt
            print_text = "Successfully generated Penman pan evaporation timeseries"
            self.update_log_file(print_text)
    
            df_prec = pd.read_csv(os.path.join(self.out_dir,'APCP.txt'),index_col='Date_Time',header=0,parse_dates=True)
            df_prec['Date_PRISM'] = np.where(df_prec.index.hour < self.tz_offset + 1,df_prec.index.date + pd.DateOffset(hours=self.tz_offset),df_prec.index.date + pd.DateOffset(days=1) + pd.DateOffset(hours=self.tz_offset))
            df_dly = df_prec.groupby(['Date_PRISM']).sum()
            df_prec.drop('Date_PRISM',axis=1,inplace=True)
            df_hly = df_dly.resample('H').bfill()
            fr = df_prec/df_hly
            fr.index.name = "Date_Time"
            df_prec = df_dly = df_hly = None
            for col in fr.columns:
                srs = fr[col]
                fr[col] = np.where(np.isnan(srs)==True,-9999,srs)
            fr.to_csv(os.path.join(self.out_dir,'APCP_FR.txt'),float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
            df_pevt.to_csv(os.path.join(self.out_dir,'potential_evaporation.txt'),index=True,header=True,float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
            # update tool_log.txt
            print_text = "Successfully generated NLDAS hourly fractions timeseries for PRISM disaggregation"
            self.update_log_file(print_text)
            return 0
        except Exception as e:
            print_text = "Error downloading and processing NLDAS data\n{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return 1

    def RainDisagg_jbb(self,params,rf):
        try:
            len_rf = len(rf)
            # generating rainfall from t h to t/2 h
            rf_pre = np.zeros((1,len_rf*2))
            # dry intermittency probability following Molnar and Burlando
            prob = 1.0 - 2**(-1.0 + params.tau_obs[0])
            # adjust intermittency probability to account for for split with both dry that will be reassigned to wet, restoring original prob
            prob = prob * (1 + prob)
            for j in range(1):
                for i in range(0,len_rf*2,2):
                    #imt = 0 means dry period, so use 1 - dry probability
                    imt = bernoulli.rvs(1 - prob, size = 2)
                    W = params.A*(params.lp[1])**poisson.rvs(1, size=2)
                    #changed logic here; setting W<0 to 1e-6 introduces small error.  If both <=0, divide evenly.
                    # I don't think this condition is ever met
                    #W[W<0] = 1e-6
                    W[W<0] = 0
                    # account for intermittency
                    if imt[0] == 0 and imt[1] ==0:
                        imt[0] = random.randint(0,1)
                        imt[1] = 1 - imt[0]
    
                    W[0] = W[0] * imt[0]
                    W[1] = W[1] * imt[1]
    
                    if (W[0]+W[1]) <= 0:
                        W[0] = 0.5
                        W[1] = 0.5
    
                    wt0 = W[0]/(W[0]+W[1])
                    wt1 = W[1]/(W[0]+W[1])
    
                    if rf[int(i/2)] == 0:
                        rf_pre[j,i] = 0
                        rf_pre[j,i+1] = 0
                    else:
                        rf_pre[j,i] = wt0 * rf[int(i/2)]
                        rf_pre[j,i+1] = wt1 * rf[int(i/2)]
    
            rf_pre = np.mean(rf_pre, axis=0)
    
            return rf_pre
        except Exception as e:
            print_text = "Error implementing RMC method \n{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return

    def rmc(self,col,df_rmc):
        try:
            r_hist = df_rmc[col].values
            params = RainDisagg(r_hist)
            data = self.RainDisagg_jbb(params,r_hist)
            for i in range(4):
                data = self.RainDisagg_jbb(params,data)
            # convert 45-min to 60-min using methodology in Guntner et al. (2001), Hydrology and Earth System Sciences, 5(2), 145-164
            # 45-min / 3 = 15-min, 15min * 4 = 60-min
            data = data / 3.0
            data = np.repeat(data,3)
            data = np.reshape(data,(int(data.size/4),4)).sum(axis=1)
    
            df_rmc = df_rmc.resample('1H').ffill()
            # cascade produces N*24 values but the hourly index has (N-1)*24+1;
            # trim surplus from the *end* to keep alignment with the first timestep
            data = data[:df_rmc.index.size]
            df_rmc['RMC'] = data
            df_rmc.drop([col],inplace=True,axis=1)
    
            return df_rmc
        except Exception as e:
            print_text = "Error implementing RMC method \n{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return

    def download_and_parse_prism_data(self):
        try:
            # check if local repository has been provided
            if self.prism_cb.GetValue() == False:
                flag = 1
            else:
                flag = 0
    
            if flag == 0:
                if self.prism_rep_dir.GetValue() == "":
                    wx.MessageBox("PRISM respository not provided","Error",style=wx.OK|wx.ICON_ERROR)
                    return 1
    
            # convert wx date to datetime date
            start_date = self.start_date.GetValue().Format('%d%m%y')
            start_date = datetime.strptime(start_date,'%d%m%y') - timedelta(hours=48)
            end_date = self.end_date.GetValue().Format('%d%m%y')
            end_date = datetime.strptime(end_date,'%d%m%y') + timedelta(hours=48)
    
            # intersect wrg shapefile with reference grid to determine uique gridcodes and wrg-gridcode fractions
            reference_grid = PRISM_REF
            df_intersect = self.calculate_grid_wrg_fractions(reference_grid,self.wrg_path.GetValue(),"PRISM")
            df_intersect['GRIDCODE'] = df_intersect['GRIDCODE'].astype('str').str.zfill(8)
            df_intersect['Row'] = df_intersect['GRIDCODE'].str[:4]
            df_intersect['Col'] = df_intersect['GRIDCODE'].str[-4:]
            df_intersect['Row'] = df_intersect['Row'].astype(int)
            df_intersect['Col'] = df_intersect['Col'].astype(int)
            min_row = np.min(df_intersect['Row'])
            max_row = np.max(df_intersect['Row'])
            min_col = np.min(df_intersect['Col'])
            max_col = np.max(df_intersect['Col'])
            
            cols = []
            for r in range(min_row,max_row+1):
                for c in range(min_col,max_col+1):
                            cols.append(str(r*10000+c).zfill(8))
    
            # update tool_log.txt
            print_text = "Successfully intersected WRG and PRISM reference grid"
            self.update_log_file(print_text)
            print_text = "Result table"
            self.update_log_file(("-") * len(print_text))
            self.update_log_file(print_text)
            self.update_log_file(("-") * len(print_text))
            with open(self.log_file_path,"a",newline='') as log_file:
                df_intersect.to_csv(log_file,index=True,header=True,float_format='%.6f')
            log_file.close()

            # make ftp connection
            if flag == 1:
                try:
                    ftp_ = ftplib.FTP("prism.nacse.org")
                    ftp_.login("anonymous", "email@email.com")
                    ftp_.cwd("daily/ppt")
                except Exception as e:
                    print_text = "Error making FTP connection to download PRISM data\n{}\n{}".format(str(e),traceback.format_exc())
                    self.update_log_file(print_text)
                    wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
                    return 1
    
            # download PRISM data
            gdal.GetDriverByName('EHdr').Register()
   
            for single_date in self.daterange(start_date,end_date):
                stamp = single_date.strftime("%Y%m%d")
                year_ = single_date.strftime("%Y")
                file_name = "PRISM_ppt_stable_4kmD2_" + stamp + "_bil.zip"
                if flag == 1:
                    out_path = os.path.join(self.prism_dwnld_dir,file_name)
                    if os.path.exists(out_path) == False:
                        try:
                            ftp_.cwd(year_)
                            out_file = open(out_path,"wb")
                            ftp_.retrbinary("RETR {}".format(file_name), out_file.write)
                            out_file.close()
                            ftp_.cwd("../")
                            print_text = "Successfully donwloaded PRISM file {}".format(file_name)
                        except Exception as e:
                            print_text = "Error downloading PRISM file {}\n{}\n{}".format(file_name,str(e),traceback.format_exc())
                            self.update_log_file(print_text)
                            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
                            return 1
    
                # download file if it does not exist in the repository
                if flag == 0:
                    out_path = os.path.join(self.prism_rep_dir.GetValue(),file_name)
    #                if os.path.exists(out_path) == False:
    #                    try:
    #                        ftp_.cwd(year_)
    #                        out_file = open(out_path,"wb")
    #                        ftp_.retrbinary("RETR {}".format(file_name), out_file.write)
    #                        out_file.close()
    #                        ftp_.cwd("../")
    #                    except Exception as e:
    #                        print_text = "PRISM file " + file_name + " was not found in the user-provided repository, download attempted and failed - " + str(e)
    #                        self.update_log_file(print_text)
    #                        wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
    #                        return 1
    
                # process PRISM data and write to dataframe
                try:
                    date_time = single_date.strftime('%Y/%m/%d') + " 00:00:00"
                    zfile = zipfile.ZipFile(out_path)
                    zfile.extractall(path=self.prism_dwnld_dir)
                    zfile.close()
                    # Extracted file always goes to the prism_dwnld_dir
                    out_path = os.path.join(self.prism_dwnld_dir,file_name)
                    bil = out_path[:-3] + "bil"
                    img = gdal.Open(bil)
                    band = img.GetRasterBand(1)
                    data = band.ReadAsArray().astype(float)
                    data = data[min_row:max_row+1,min_col:max_col+1]
                    data = data.flatten()
                    temp_df = pd.DataFrame(data)
                    temp_df = temp_df.T
                    temp_df.insert(0,'Date_Time',date_time)
                    if single_date == start_date:
                        with open(os.path.join(self.out_dir,"PPT_gridcode.txt"),"w") as data_dump:
                            data_dump.write("Date_Time," + ",".join(x for x in cols) + "\n")
                        data_dump.close()
                    with open(os.path.join(self.out_dir,"PPT_gridcode.txt"),"a",newline='') as data_dump:
                        temp_df.to_csv(data_dump,index=False,header=False,float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
                    data_dump.close()
                    print_text = "Successfully processed PRISM file {}".format(out_path)
                    self.update_log_file(print_text)
                except Exception as e:
                    print_text = "Error processing PRISM file {}\n{}\n{}".format(out_path,str(e),traceback.format_exc())
                    self.update_log_file(print_text)
                    wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
                    return 1
    
            if flag == 1:
                try:
                    ftp_.close()
                except:
                    pass

            # aggregate by weather regions
            df = pd.read_csv(os.path.join(self.out_dir,"PPT_gridcode.txt"),header=0,index_col="Date_Time",parse_dates=True)
            new_cols = df_intersect[self.wrg_field.GetValue()].unique()
            new_cols = ['WRG_' + str(x) for x in new_cols]
            for col in new_cols:
                df[col] = 0
            for i,row in df_intersect.iterrows():
                df['WRG_'+str(row[self.wrg_field.GetValue()])] += df[row['GRIDCODE']] * row['Fr']
    
            df = df[new_cols]
            # PRISM data is in mm while HSPF requires data in inches
            df = np.divide(df,25.4)
            # timezone correction - PRISM data in UTC mid-day to mid-day
            df.index = df.index + pd.DateOffset(hours=12-self.tz_offset)
    
            # update tool_log.txt
            print_text = "Successfully generated PRISM timeseries aggregated spatially and corrected for timezone"
            self.update_log_file(print_text)
            # truncate to user-specified start_date (at 0:00 hours) and end_date (at 23:00 hours)
            # keep an extra 24 hours at the start and end for PRISM disaggregation
            df = df.truncate(before=start_date + timedelta(hours=24),after=end_date - timedelta(hours=1),axis=0)
            df.to_csv(os.path.join(self.out_dir,'PPT.txt'),index=True,header=True,float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
    
            # disaggregate PRISM data
            df_nldas_fr = pd.read_csv(os.path.join(self.out_dir,'APCP_FR.txt'),index_col='Date_Time',header=0,parse_dates=True)
            df_nldas_fr = df_nldas_fr.truncate(before=str(df.index[0]),after=str(df.index[-1]),axis=0)
            df_nldas_fr.columns = [i + '_FR' for i in df_nldas_fr.columns]
    
            df_prec = pd.DataFrame()
            for col in new_cols:
                df_ = None
                df_ = df[col].to_frame()
                # generate cascade based estimates
                df_rmc = self.rmc(col,df_)
                # same daily mid-day to mid-day daily totals
                df_ = df_.resample('1H').bfill()
                df_ = df_.join(df_nldas_fr[col + '_FR'])
                df_ = df_.join(df_rmc)
                df_.columns = ['Value','Fraction','RMC']
                # multiply PRISM daily amounts by NLDAS fraction or RMC if NLDAS reports 0 rainfall for a day
                df_['Result'] = np.where(df_['Fraction']==-9999,df_['RMC'],df_['Value']*df_['Fraction'])
                df_prec[col] = df_['Result']
            df_prec.set_index(df_.index,inplace=True)
            df_prec = df_prec.truncate(before=start_date + timedelta(hours=48),after=end_date - timedelta(hours=25),axis=0)
            df_prec.to_csv(os.path.join(self.out_dir,'precipitation.txt'),index=True,header=True,float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
            # update tool_log.txt
            print_text = "Successfully diaggregated daily PRISM precipitation timeseries to hourly using NLDAS and RMC"
            self.update_log_file(print_text)
            return 0
        except Exception as e:
            print_text = "Error downloading and processing PRISM data\n{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return 1

    def download_and_parse_narr_data(self):
        try:
            # check if local repository option has been checked
            if self.narr_cb.GetValue() == False:
                flag = 1
            else:
                flag = 0
    
            if flag == 0:
                if self.narr_rep_dir.GetValue() == "":
                    wx.MessageBox("NARR respository not provided","Error",style=wx.OK|wx.ICON_ERROR)
                    return 1
    
            # make ftp connection
            if flag == 1:
                try:
                    ftp_ = ftplib.FTP("ftp.cdc.noaa.gov")
                    ftp_.login("anonymous", "email@email.com")
                    ftp_.cwd("NARR/monolevel")
                except Exception as e:
                    print_text = "Error making ftp connection to download NARR data\n{}\n{}".format(str(e),traceback.format_exc())
                    self.update_log_file(print_text)
                    wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
                    return 1
    
            # convert wx date to datetime date
            start_date = self.start_date.GetValue().Format('%d%m%y')
            start_date = datetime.strptime(start_date,'%d%m%y') - timedelta(hours=24)
            end_date = self.end_date.GetValue().Format('%d%m%y')
            end_date = datetime.strptime(end_date,'%d%m%y') + timedelta(hours=24)
            start_year = start_date.year
            end_year = end_date.year
            
            # dtermine min and max co-ordinates to extract NARR data
            reference_grid = NARR_REF
            grid_row_col = gpd.read_file(reference_grid)
            weather_regions = gpd.read_file(self.wrg_path.GetValue())
            weather_regions_prj = weather_regions.copy()
            # project boundary to reference grid projection
            weather_regions_prj['geometry'] = weather_regions_prj['geometry'].to_crs(grid_row_col.crs)
            weather_regions_prj.crs = grid_row_col.crs
            
            min_lat = np.floor(weather_regions_prj.bounds['miny'].min())
            min_long = np.floor(weather_regions_prj.bounds['minx'].min())
            max_lat = np.ceil(weather_regions_prj.bounds['maxy'].max())
            max_long = np.ceil(weather_regions_prj.bounds['maxx'].max())
            
            # update tool_log.txt
            print_text = "Successfully generated bounding box for NARR data download and processing"
            self.update_log_file(print_text)
            
            # process NARR data and write to dataframe
            df = pd.DataFrame()
            index = pd.date_range('1/1/'+str(start_year),'1/1/'+str(end_year+1),freq='3H')
            for single_year in range(start_year,end_year+1):
                file_name = "tcdc." + str(single_year) + ".nc"
                if flag == 1:
                    out_path = os.path.join(self.narr_dwnld_dir,file_name)
                    if os.path.exists(out_path) == False:
                        try:
                            out_file = open(out_path,"wb")
                            ftp_.retrbinary("RETR {}".format(file_name), out_file.write)
                            out_file.close()
                            print_text = "Successfully downloaded NARR file {}".format(file_name)
                            self.update_log_file(print_text)
                        except Exception as e:
                            print_text = "Error downloading NARR file {}\n{}\n{}".format(file_name,str(e),traceback.format_exc())
                            self.update_log_file(print_text)
                            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
                            return 1
                
                if flag == 0:
                    out_path = os.path.join(self.narr_rep_dir.GetValue(),file_name)
                    # download file if it does not exist in repository
    #                if os.path.exists(out_path) == False:
    #                    try:
    #                        out_file = open(out_path,"wb")
    #                        ftp_.retrbinary("RETR {}".format(file_name), out_file.write)
    #                        out_file.close()
    #                    except Exception as e:
    #                        print_text = "NARR file " + file_name + " was not found in the user-provided repository, download attempted and failed - " + str(e)
    #                        self.update_log_file(print_text)
    #                        wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
    #                        return 1
    
                # process NARR data
                ncf = Dataset(out_path)
                # need to get the indices only once
                if single_year == start_year:
                    lats = ncf.variables['y'][:]
                    lons = ncf.variables['x'][:]
                    # latitude lower and upper index
                    latli = np.argmin(np.abs(lats-min_lat))
                    latui = np.argmin(np.abs(lats-max_lat))
                    # longitude lower and upper index
                    lonli = np.argmin(np.abs(lons-min_long))
                    lonui = np.argmin(np.abs(lons-max_long))
                    cols = []
                    for r in range(latli,latui+1):
                        for c in range(lonli,lonui+1):
                            cols.append(str(r*1000+c).zfill(6))
    
                # subset netcdf file
                ncf_sbst = ncf.variables['tcdc'][:,latli:latui+1,lonli:lonui+1]
                shp = ncf_sbst.shape
                ncf_sbst = np.reshape(ncf_sbst,(shp[0],-1))
                temp_df = pd.DataFrame(ncf_sbst)
                df = df.append(temp_df)
                temp_df = ncf_sbst = None
                ncf.close()
            
            if flag == 1:
                try:
                    ftp_.close()
                except:
                    pass
    
            df.set_index(index[:len(df.index)],inplace=True)
            df.columns = cols
            df.index.name = 'Date_Time'
    
            # interect wrg shapefile and NARR reference grid to determine wrg-gridcode fractions
            df_intersect = self.calculate_grid_wrg_fractions(reference_grid,self.wrg_path.GetValue(),"NARR")
            df_intersect['GRIDCODE'] = df_intersect['GRIDCODE'].astype('str')
            # update tool_log.txt
            print_text = "Successfully intersected WRG and NARR reference grid"
            self.update_log_file(print_text)
            print_text = "Result table"
            self.update_log_file(("-") * len(print_text))
            self.update_log_file(print_text)
            self.update_log_file(("-") * len(print_text))
            with open(self.log_file_path,"a",newline='') as log_file:
                df_intersect.to_csv(log_file,index=True,header=True,float_format='%.6f')
            log_file.close()
            # write NARR timeseries by gridcode
            with open(os.path.join(self.out_dir,"CLOU_gridcode.txt"),"a",newline='') as data_dump:
                df.to_csv(data_dump,index=True,header=True,float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
            data_dump.close()
            new_cols = df_intersect[self.wrg_field.GetValue()].unique()
            new_cols = ['WRG_' + str(x) for x in new_cols]
            for col in new_cols:
                df[col] = 0
            for i,row in df_intersect.iterrows():
                df['WRG_'+str(row[self.wrg_field.GetValue()])] += df[row['GRIDCODE']] * row['Fr']
    
            # update tool_log.txt
            print_text = "Successfully generated timeseries for NARR cloud cover by gridcode"
            self.update_log_file(print_text)
    
            df = df[new_cols]
            # NARR data is in percentage while HSPF requires data in tenths
            df = np.divide(df,10.0)
            # timezone correction
            df.index = df.index + pd.DateOffset(hours=-self.tz_offset)
    
            # update tool_log.txt
            print_text = "Successfully generated timeseries for NARR cloud cover aggregated spatially and corrected for timezone"
            self.update_log_file(print_text)
            #print_text = "3-hour timeseries"
            #self.update_log_file(("-") * len(print_text))
            #self.update_log_file(print_text)
            #self.update_log_file(("-") * len(print_text))
            #with open (self.log_file_path,"a",newline='') as log_file:
            #    df.to_csv(log_file,index=True,header=True,float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
            #log_file.close()
    
            # resample 3-hourly to hourly
            df = df.resample('1H').pad()
            # truncate to user-specified start_date (at 0:00 hours) and end_date (at 23:00 hours)
            df = df.truncate(before=start_date + timedelta(hours=24),after=end_date - timedelta(hours=1),axis=0)
            df.to_csv(os.path.join(self.out_dir,'cloud_cover.txt'),index=True,header=True,float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
            # udate tool log
            print_text = "Successfully generated cloud cover timeseries"
            self.update_log_file(print_text)
            return 0
        except Exception as e:
            print_text = "Error downloading and processing NARR data \n{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return 1

    def dwnld_prcs_bttn_click(self,event,arg):
        try:
            # update tool_log.txt
            print_text = "'Download and Process Data' started by " + os.environ['USERNAME'] + " on " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.update_log_file(("=") * len(print_text))
            self.update_log_file(print_text)
            self.update_log_file(("=") * len(print_text))

            if os.path.exists(self.working_dir_path.GetValue()) == False or self.working_dir_path.GetValue() == "":
                wx.MessageBox("Working directory not selected","Error",style=wx.OK|wx.ICON_ERROR)
                return
            if os.path.exists(self.wrg_path.GetValue()) == False or self.wrg_path.GetValue() == "":
                wx.MessageBox("Weather region shapefile not selected","Error",style=wx.OK|wx.ICON_ERROR)
                return
            # check if wrg within MN HUC8 watersheds
            weather_regions = gpd.read_file(self.wrg_path.GetValue())
            valid_extent = gpd.read_file(BBOX_REF)
            weather_regions_prj = weather_regions.copy()
            weather_regions_prj['geometry'] = weather_regions_prj['geometry'].to_crs(valid_extent.crs)
            weather_regions_prj['_id_'] = 0
            weather_regions_prj = weather_regions_prj.dissolve(by='_id_')
            if weather_regions_prj.within(valid_extent).values[0] == False:
                print_text = "Selected weather region shapefile is not completely within Minnesota HUC8 watersheds"
                self.update_log_file(print_text)
                wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
                return
            if self.wrg_field.GetValue() == "":
                wx.MessageBox("Weather region ID field not selected","Error",style=wx.OK|wx.ICON_ERROR)
                return
            if self.data_table.GetNumberRows() <= 1:
                wx.MessageBox("TSTYPE and DSN information table is incomplete","Error",style=wx.OK|wx.ICON_ERROR)
                return
            if self.end_date.GetValue() < self.start_date.GetValue():
                wx.MessageBox("End date must be after start date","Error",style=wx.OK|wx.ICON_ERROR)
                return
            # ensure that start_date - end_date > 31 days; this is required for the RMC methds to work
            sd = self.start_date.GetValue().Format('%d%m%y')
            sd = datetime.strptime(sd,'%d%m%y')
            ed = self.end_date.GetValue().Format('%d%m%y')
            ed = datetime.strptime(ed,'%d%m%y')
            if (ed-sd).days < 32:
                wx.MessageBox("End date must be at least 31 days after start date","Error",style=wx.OK|wx.ICON_ERROR)
                return
            # # write status messages to status bar
            self.status_bar.SetStatusText('Downloading and processing data ... this might take a while')
            # download NLDAS data
            status = self.download_and_parse_nldas_data()
            if status == 1:
                self.status_bar.SetStatusText('Ready')
                return
            # download PRISM data
            status = self.download_and_parse_prism_data()
            if status == 1:
                self.status_bar.SetStatusText('Ready')
                return
            # download NARR data
            status = self.download_and_parse_narr_data()
            if status == 1:
                self.status_bar.SetStatusText('Ready')
                return
    
            self.status_bar.SetStatusText('Ready')
        except Exception as e:
            print_text = "{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return

    def wdm_write_one_dsn(self,wdm_path,dsn,scenario,location,constituent,tstype,statid,description,srs):
        try:
            if os.path.isfile(wdm_path):
                dsns = wdmtoolbox.listdsns(wdm_path)
                if len(dsns) > 0:
                    if int(dsn) in list(dsns.keys()):
                        srs_ = wdmtoolbox.extract(wdm_path,dsn)
                        srs = srs.to_frame()                    
                        if self.dsn_action.GetValue() == "append to existing DSN":
                            srs_.columns = srs.columns
                            res = srs.combine_first(srs_)
                        if self.dsn_action.GetValue() == "overwrite existing DSN":
                            wdmtoolbox.deletedsn(wdm_path,dsn)
                            wdmtoolbox.createnewdsn(wdm_path,dsn,scenario=scenario,location=location,constituent=constituent,tstype=tstype,base_year=1900,tcode=3,tsstep=1,statid=statid,description=description,tsfill=-999.0)
                            res = srs
                        res.sort_index()
                        wdmtoolbox.csvtowdm(wdm_path,dsn,input_ts=res)
                    else:
                        wdmtoolbox.createnewdsn(wdm_path,dsn,scenario=scenario,location=location,constituent=constituent,tstype=tstype,base_year=1900,tcode=3,tsstep=1,statid=statid,description=description,tsfill=-999.0)
                        wdmtoolbox.csvtowdm(wdm_path,dsn,input_ts=srs)
                else:
                    wdmtoolbox.createnewdsn(wdm_path,dsn,scenario=scenario,location=location,constituent=constituent,tstype=tstype,base_year=1900,tcode=3,tsstep=1,statid=statid,description=description,tsfill=-999.0)
                    wdmtoolbox.csvtowdm(wdm_path,dsn,input_ts=srs)
            else:
                wdmtoolbox.createnewwdm(wdm_path)
                wdmtoolbox.createnewdsn(wdm_path,dsn,scenario=scenario,location=location,constituent=constituent,tstype=tstype,base_year=1900,tcode=3,tsstep=1,statid=statid,description=description,tsfill=-999.0)
                wdmtoolbox.csvtowdm(wdm_path,dsn,input_ts=srs)
            print_text = "Successful writing DSN {} to {}".format(str(dsn),wdm_path)
            self.update_log_file(print_text)
        except Exception as e:
            print_text = "Error writing DSN {} to {}\n{}\n{}".format(str(dsn),wdm_path,str(e),traceback.format_exc())
            self.update_log_file(print_text)            
            pass

    def wdm_write_bttn_click(self,event,arg):
        try:
            # check to ensure that TSTYPE is of length 4 characters and DSN is numeric less than 9999
            nrows = self.data_table.GetNumberRows()
            ncols = self.data_table.GetNumberCols()
            for r in range(nrows):
                for c in range(ncols):
                    val = self.data_table.GetCellValue(r,c)
                    if r == 0 and len(val) > 4:
                        wx.MessageBox("TSTPYE length cannot exceed 4 characters","Error",style=wx.OK|wx.ICON_ERROR)
                        return
                    if r > 0:
                        if val.isdigit() == True:
                            if int(val) > 9999:
                                wx.MessageBox("DSN must be a positive integer less than 9999","Error",style=wx.OK|wx.ICON_ERROR)
                                return    
                        else:
                            wx.MessageBox("DSN must be a positive integer less than 9999","Error",style=wx.OK|wx.ICON_ERROR)
                            return
                            
            # update tool_log.txt
            print_text = "'Write to WDM File' started by " + os.environ['USERNAME'] + " on " + datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            self.update_log_file(("=") * len(print_text))
            self.update_log_file(print_text)
            self.update_log_file(("=") * len(print_text))
            wdm_path = os.path.normpath(self.wdm_path.GetValue())
            if self.wdm_path.GetValue() == "":
                wx.MessageBox("WDM filename not provided","Error",style=wx.OK|wx.ICON_ERROR)
                return
            if os.path.exists(self.working_dir_path.GetValue()) == False or self.working_dir_path.GetValue() == "":
                wx.MessageBox("Working directory not selected","Error",style=wx.OK|wx.ICON_ERROR)
                return
            if os.path.exists(self.wrg_path.GetValue()) == False or self.wrg_path.GetValue() == "":
                wx.MessageBox("Weather region shapefile not selected","Error",style=wx.OK|wx.ICON_ERROR)
                return
            if self.wrg_field.GetValue() == "":
                wx.MessageBox("Weather region ID field not selected","Error",style=wx.OK|wx.ICON_ERROR)
                return
            if self.data_table.GetNumberRows() <= 1:
                wx.MessageBox("TSTYPE and DSN information table is incomplete","Error",style=wx.OK|wx.ICON_ERROR)
                return
            # write status messages to status bar 
            self.status_bar.SetStatusText("Writing time-series data to WDM file ...")
            nrows = self.data_table.GetNumberRows()
            ncols = self.data_table.GetNumberCols()
            for c in range(ncols):
                file_name = self.data_table.GetColLabelValue(c)
                scenario = "NLDAS"
                if file_name == "Precipitation": scenario = "PRISM"
                if file_name == "Cloud Cover": scenario = "NARR"            
                file_name = file_name.lower().replace(" ","_") + ".txt"
                file_path = os.path.join(self.out_dir,file_name)
                if os.path.exists(file_path):
                    df = pd.read_csv(file_path,index_col='Date_Time',header=0,parse_dates=True)
                    for r in range(nrows):
                        if r == 0:
                            #constituent = self.data_table.GetColLabelValue(c)
                            constituent = self.data_table.GetCellValue(r,c)
                            tstype = self.data_table.GetCellValue(r,c)
                            description = self.data_table.GetColLabelValue(c)
                        else:
                            dsn = self.data_table.GetCellValue(r,c)
                            location = self.data_table.GetRowLabelValue(r)
                            statid = self.data_table.GetRowLabelValue(r)
                            srs = df[statid]
                            if tstype != "" and dsn != "":
                                self.wdm_write_one_dsn(wdm_path,dsn,scenario,location,constituent,tstype,statid,description,srs)
            self.get_wdm_dsns(wdm_path)
            # write status messages to status bar
            print_text = "Successful writing timeseries data to WDM file"
            self.update_log_file(print_text)
            self.status_bar.SetStatusText("Ready")
        except Exception as e:
            print_text = "Error writing timeseries data to WDM file\n{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            self.status_bar.SetStatusText("Ready")
            return

    def get_wdm_dsns(self,wdm_path):
        try:
            if os.path.isfile(wdm_path):
                dsns = wdmtoolbox.listdsns(wdm_path)
                dict_ = list(dsns.values())
                self.dsn_list = []
                for i,res in enumerate(dict_):
                    dsn = res['dsn']
                    location = res['location']
                    tstype = res['tstype']
                    self.dsn_list.append("DSN:" + str(dsn) + ",LOCATION:" + str(location.decode('utf-8')) + ",TSTYPE:" + str(tstype.decode('utf-8')))
                self.dsn_select.Clear()
                self.dsn_select.AppendItems(self.dsn_list)
        except:
            pass

    def plot_or_export_dsn_bttn_click(self,event,arg):
        flag = arg[0]
        try:
            wdm_path = os.path.normpath(self.wdm_path.GetValue())
            if self.wdm_path.GetValue() == "":
                wx.MessageBox("WDM filename not provided","Error",style=wx.OK|wx.ICON_ERROR)
                return
            if self.dsn_select.GetValue() == "":
                wx.MessageBox("Timerseries not selected","Error",style=wx.OK|wx.ICON_ERROR)
                return
            dsn_ts_dict = {"HOURLY":"H","DAILY":"D","MONTHLY":"MS","YEARLY":"A"}
            dsn_stat_dict = {"SUM":"sum","AVG":"mean","MAX":"max","MIN":"min"}
            str_ = self.dsn_select.GetValue()
            dsn = str_.split(",")[0].split(":")[1]
            tstype = str_.split(",")[2].split(":")[1]
            if wdm_path != "" and dsn != "":
                dsns = wdmtoolbox.listdsns(wdm_path)
                if len(dsns) > 0:
                    if int(dsn) in list(dsns.keys()):
                        srs_ = wdmtoolbox.extract(wdm_path,dsn)
                        srs_= srs_.resample(dsn_ts_dict[self.dsn_ts.GetValue()]).apply(dsn_stat_dict[self.dsn_stat.GetValue()])
                        # write dsn time-series to text file in working directory
                        if flag == "EXPORT":                            
                            dlg = wx.FileDialog(self.panel_,"Open TXT file",wildcard="Text File (*.txt)|*.txt",style=wx.FD_OPEN)
                            if dlg.ShowModal() == wx.ID_OK:
                                txt_file = dlg.GetPath()
                                dlg.Destroy()                            
                                srs_.to_csv(txt_file,index=True,header=True,float_format='%.6f',date_format='%Y/%m/%d %H:%M:%S')
                        if flag == "PLOT":
                            dates = srs_.index
                            values = srs_.values
                            self.figr.clf()
                            ax = self.figr.add_subplot(111)
                            ax.plot_date(x=dates,y=values,fmt='C1o--',label=tstype,linewidth=0.5,markersize=0)
                            #ax.set_ylabel(tstype)
                            ax.grid(True)
                            ax.xaxis.set_major_formatter(mdates.DateFormatter('%m-%d-%Y %H:%M'))
                            ax.xaxis.set_major_locator(mdates.AutoDateLocator())
                            #ax.xaxis.set_tick_params(rotation=30)
                            for label in ax.xaxis.get_majorticklabels():
                                label.set_rotation(15)
                                label.set_horizontalalignment('right')
                                label.set_fontsize(8)
                            for label in ax.yaxis.get_majorticklabels():
                                label.set_fontsize(8)
                            ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.2), ncol=1, fontsize=12)
                            self.figr.subplots_adjust(top=0.85,bottom=0.15,right=0.95,left=0.05,wspace=0.35,hspace=0.35)
                            self.cnvs.draw()
        except Exception as e:
            print_text = "Error plotting or exporting DSN time-series\n{}\n{}".format(str(e),traceback.format_exc())
            self.update_log_file(print_text)
            wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
            return
def main():
    try:
        app = wx.App()
        frame_ = GUI()
        frame_.Show()
        app.MainLoop()
    except Exception as e:
        print_text = "{}\n{}".format(str(e),traceback.format_exc())
        wx.MessageBox(print_text,"Error",style=wx.OK|wx.ICON_ERROR)
        return

if __name__ == '__main__': main()
