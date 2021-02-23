import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import AutoMinorLocator


def phase_diagram(indexname,PC1,PC2,dates,months,days,plotname='./MJO_phase_diagram',plottype='png'):
    """
    Plot phase diagram for OMI or RMM. Do not flip the sign and PCs for OMI before
    passing to the routine.
    :param indexname: name of index, should be either OMI or RMM
    :type indexname: string
    :param PC1: first principal component, either from observations or computed
    :param PC2: second principal component, either from observations or computed
    :param dates: numpy array of dates
    :param months: numpy array of month integers for each time in PC1 and PC2
    :param days: numpy array of day of month intergers for each time in PC1 and PC2
    :param plotname: name of figure file
    :param plottype: type of figure file to save
    :return: none
    """

    # flip the sign for OMI to match RMM phases
    if indexname=='OMI':
        tmp = PC1.copy()
        PC1 = PC2
        PC2 = -tmp

    #####################################################
    # set parameters and settings for plot
    monthnames = ['Jan','Feb','Mar','Apr','May','Jun','Jul','Aug','Sep','Oct','Nov','Dec']
    colors = ['black','gold','darkgreen','tab:red','tab:purple','tab:orange',
    'tab:blue','tab:grey','tab:green','tab:pink','tab:olive','tab:cyan']

    nMon            = 0
    monName         = [monthnames[months[0]-1]]
    monCol          = [colors[months[0]-1]]

    # plot phase diagram
    fig = plt.figure()
    ax = fig.add_subplot(111)

    # lines to separate the phases  
    plt.plot([-4,4],[-4,4],linewidth=0.2,linestyle='--',color='k')
    plt.plot([-4,4],[4,-4],linewidth=0.2,linestyle='--',color='k')
    plt.plot([-4,4],[0,0],linewidth=0.2,linestyle='--',color='k')
    plt.plot([0,0],[-4,4],linewidth=0.2,linestyle='--',color='k')
    plt.xlim([-4.0,4.0])
    plt.ylim([-4.0,4.0])

    # circle in the center of the plot to denote weak index  
    circle = plt.Circle((0, 0), radius=1.0, fc='k', ec='k', alpha=0.2)
    plt.gca().add_patch(circle)

    # cycle over dates and plot the two princial components against each other  
    alph=1.0
    time = dates
    labelday = 5  # label every fifth day

    # start marker
    plt.plot(PC1[0],PC2[0],color='k',marker='o',markersize=5)
    # loop through dates and plot line from current to next time 
    for im in np.arange(0,len(months)-1):
        lcolor = colors[months[im]-1]
        plt.plot(PC1[im:im+2],PC2[im:im+2],'-',color=lcolor,alpha=alph)
        plt.plot(PC1[im],PC2[im],color='k',alpha=alph,marker='o',markersize=2)
        # text labels for the days
        if days[im]%labelday==0:
            plt.text(PC1[im],PC2[im],str(days[im]),color='k')
        # get current month color for month labels
        if im>0 and days[im]==1:
            nMon = nMon+1
            monName.append(monthnames[months[im]-1])
            monCol.append(colors[months[im]-1])   
    # last date marker        
    plt.plot(PC1[len(months)-1],PC2[len(months)-1],color='k',alpha=alph,marker='o',markersize=2)    

    # axis labels and title 
    #plt.xlabel(indexname+'1')
    #plt.ylabel(indexname+'2')
    plt.title(indexname+' '+str(dates.min())+' to '+str(dates.max()))

    # axes
    ax.set_aspect('equal')
    ax.tick_params(bottom=True, top=True, left=True, right=True,which='both')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())

    # text for phases
    plt.text(0.5, 0.99, "Phase 7 (Western Pacific) Phase 6", horizontalalignment='center', verticalalignment='top',
    transform=ax.transAxes)
    plt.text(0.5, 0.01, "Phase 2 (Indian) Phase 3", horizontalalignment='center', verticalalignment='bottom',
    transform=ax.transAxes)
    plt.text(0.99, 0.5, "Phase 5 (Maritime) Phase 4", horizontalalignment='right', verticalalignment='center',
    transform=ax.transAxes,rotation=-90)
    plt.text(0.01, 0.5, "Phase 1 (Western Hem, Africa) Phase 8", horizontalalignment='left', verticalalignment='center',
    transform=ax.transAxes,rotation=90)

    # text for month names
    xstrt=0.01
    for im in np.arange(0,nMon+1,1):
        plt.text(xstrt, 0.01, monName[im], color=monCol[im], horizontalalignment='left', verticalalignment='bottom',
        transform=ax.transAxes)
        xstrt=xstrt+0.08

    # save figure to file
    plt.savefig(plotname+'.'+plottype,format=plottype)



def pc_time_series(indexname,PC1,PC2,dates,months,days,plotname='./MJO_time_series',plottype='png'):
    """
    Plot OMI or RMM PC1 and PC2 time series. Do not flip the sign and PCs for OMI before
    passing to the routine.
    :param indexname: name of index, should be either OMI or RMM
    :type indexname: string
    :param PC1: first principal component, either from observations or computed
    :param PC2: second principal component, either from observations or computed
    :param dates: numpy array of dates
    :param months: numpy array of month integers for each time in PC1 and PC2
    :param days: numpy array of day of month intergers for each time in PC1 and PC2
    :param plotname: name of figure file
    :param plottype: type of plot to save
    :return: none
    """

    # flip the sign for OMI to match RMM phases
    if indexname=='OMI':
        tmp = PC1.copy()
        PC1 = PC2
        PC2 = -tmp

    #####################################################
    # set parameters and settings for plot
    colors = [ 'tab:blue', 'tab:orange']

    ymax = 3.
    ymin = -3.

    # plot phase diagram
    fig = plt.figure(figsize=(10,3))
    ax = fig.add_subplot(111)

    # zero line  
    plt.plot([dates[0], dates[-1]],[0,0],linewidth=0.2,linestyle='-',color='k')

    plt.plot(dates,PC1,linewidth=2,linestyle='-',color=colors[0],label=indexname+'1')  
    plt.plot(dates,PC2,linewidth=2,linestyle='--',color=colors[1],label=indexname+'2')    

    # axis labels and title 
    plt.title(indexname)

    # axes
    #ax.set_aspect('equal')
    ax.tick_params(bottom=True, top=True, left=True, right=True,which='both')
    ax.xaxis.set_minor_locator(AutoMinorLocator())
    ax.yaxis.set_minor_locator(AutoMinorLocator())
    ax.set_xlim(dates[0], dates[-1])
    ax.set_ylim(ymin, ymax)    

    # save figure to file
    plt.savefig(plotname+'.'+plottype,format=plottype)


def plot_rmm_eofs(EOF1,EOF2,plotname='RMM_EOFs',plottype='png'):
    """
    Plot RMM EOFs for OLR, U850 and U200 at all longitudes.
    :param EOF1: DataArray containing the EOF1 components for OLR, U850 and U200 (3, lon)
    :param EOF2: DataArray containing the EOF2 components for OLR, U850 and U200 (3, lon)
    :param plotname: name for the figure file
    :param plottype: type of figure file to output
    """

    lon = EOF1['lon']

    ymax = 1.
    ymin = -1.
    xmax = 360
    xmin = 0

    fig, (ax1, ax2) = plt.subplots(2,figsize=(10,5), sharex = True, sharey = True)
    fig.suptitle('RMM')

    # EOF1 
    ax1.plot(lon,EOF1[0,:],linewidth=2, linestyle='-', color='k',label='OLR')
    ax1.plot(lon,EOF1[1,:],linewidth=2, linestyle='--', color='tab:blue',label='U850')  
    ax1.plot(lon,EOF1[2,:],linewidth=2, linestyle=':', color='tab:orange',label='U200') 
    ax1.legend()
    ax1.set_xlim(xmin,xmax)

    # EOF2 
    ax2.plot(lon,EOF2[0,:],linewidth=2, linestyle='-', color='k',label='OLR')
    ax2.plot(lon,EOF2[1,:],linewidth=2, linestyle='--', color='tab:blue',label='U850')  
    ax2.plot(lon,EOF2[2,:],linewidth=2, linestyle=':', color='tab:orange',label='U200')  
    ax2.legend() 
    ax2.set_xlim(xmin,xmax) 

    # axes
    ax1.tick_params(bottom=True, top=True, left=True, right=True,which='both')
    ax1.xaxis.set_minor_locator(AutoMinorLocator())
    ax1.yaxis.set_minor_locator(AutoMinorLocator())

    # save figure to file
    plt.savefig(plotname+'.'+plottype,format=plottype)

