# import standard plotting and animation
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib import gridspec
from IPython.display import clear_output
from mpl_toolkits.mplot3d import proj3d
from matplotlib.text import Annotation
from mpl_toolkits.mplot3d.proj3d import proj_transform

# import autograd functionality
import numpy as np
import math
import time
import copy

class Visualizer:
    '''
    Various plotting functions for hoemwork 3 of deep learning from scratch course
    '''             

    # plot data and predict function
    def plot_data_fit(self,x,y,**kwargs):
        # create figure and plot data
        fig, ax = plt.subplots(1, 1, figsize=(6,3))
        ax.scatter(x,y,color = 'k',edgecolor = 'w'); 

        # cleanup panel
        xmin = copy.deepcopy(min(x))
        xmax = copy.deepcopy(max(x))
        xgap = (xmax - xmin)*0.1
        xmin -= xgap
        xmax += xgap 

        ymin = copy.deepcopy(min(y))
        ymax = copy.deepcopy(max(y))
        ygap = (ymax - ymin)*0.25
        ymin -= ygap
        ymax += ygap

        # set viewing limits
        ax.set_xlim(xmin,xmax)
        ax.set_ylim(ymin,ymax)

        # check if we have a model to fit
        if 'predict' in kwargs:
            predict = kwargs['predict']
            weights = kwargs['weights']

            s = np.linspace(xmin,xmax,200)
            t = [predict(v,weights) for v in s]
            ax.plot(s,t,linewidth = 2.25, color = 'r',zorder = 3)
        plt.show()

    # a small Python function for plotting the distributions of input features
    def feature_distributions(self,x,y,title,**kwargs):
        # create figure 
        fig, ax = plt.subplots(1, 1, figsize=(9,3))

        # loop over input and plot each individual input dimension value
        N = np.shape(x)[1]    # dimension of input
        for n in range(N):
            ax.scatter((n+1)*np.ones((len(y),1)),x[:,n],color = 'k',edgecolor = 'w')

        # set xtick labels 
        ticks = np.arange(1,N+1)
        labels = [r'$x_' + str(n+1) + '$' for n in range(N)]
        ax.set_xticks(ticks)
        if 'labels' in kwargs:
            labels = kwargs['labels']
        ax.set_xticklabels(labels, minor=False)

        # label axes and title of plot, then show
        ax.set_xlabel('input dimension / feature')
        ax.set_ylabel(r'$\mathrm{value}$',rotation = 0,labelpad = 20)
        ax.set_title(title)
        plt.show()
    

    # compare cost to counting
    def compare_regression_histories(self,histories):
        ##### setup figure to plot #####
        # initialize figure
        fig = plt.figure(figsize = (7,3))

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 1) 
        ax1 = plt.subplot(gs[0]); 

        # run through weights, evaluate classification and counting costs, record
        c = 1
        for history in histories:
            # plot both classification and counting cost histories
            ax1.plot(history,label = 'run ' + str(c),linewidth = 4*(0.8)**(c))
            c += 1

        ax1.set_xlabel('iteration',fontsize = 10)
        ax1.set_ylabel('cost function value',fontsize = 10)
        plt.legend(loc='upper right')
        plt.show()
    
    
    # show contour plot of input function
    def draw_setup(self,g,**kwargs):
        self.g = g                         # input function        
        xmin = -3.1
        xmax = 3.1
        ymin = -3.1
        ymax = 3.1
        num_contours = 20
        if 'xmin' in kwargs:            
            xmin = kwargs['xmin']
        if 'xmax' in kwargs:
            xmax = kwargs['xmax']
        if 'ymin' in kwargs:            
            ymin = kwargs['ymin']
        if 'ymax' in kwargs:
            ymax = kwargs['ymax']            
        if 'num_contours' in kwargs:
            num_contours = kwargs['num_contours']   
            
        # choose viewing range using weight history?
        if 'view_by_weights' in kwargs:
            view_by_weights = True
            weight_history = kwargs['weight_history']
            if view_by_weights == True:
                xmin = min([v[0] for v in weight_history])[0]
                xmax = max([v[0] for v in weight_history])[0]
                xgap = (xmax - xmin)*0.25
                xmin -= xgap
                xmax += xgap

                ymin = min([v[1] for v in weight_history])[0]
                ymax = max([v[1] for v in weight_history])[0]
                ygap = (ymax - ymin)*0.25
                ymin -= ygap
                ymax += ygap
        
        ##### construct figure with panels #####
        # construct figure
        fig = plt.figure(figsize = (10,4))

        # remove whitespace from figure
        fig.subplots_adjust(left=0, right=1, bottom=0, top=1) # remove whitespace

        # create subplot with 3 panels, plot input function in center plot
        gs = gridspec.GridSpec(1, 3, width_ratios=[1,4,1]) 
        ax = plt.subplot(gs[0]);ax.axis('off');
        ax1 = plt.subplot(gs[1],aspect='equal');
        ax2 = plt.subplot(gs[2]);ax2.axis('off');

        ### plot function as contours ###
        #self.draw_surface(ax,wmin,wmax,wmin,wmax)
        self.draw_contour_plot(ax1,num_contours,xmin,xmax,ymin,ymax)
        
        ### cleanup panels ###
        ax1.set_xlabel('$w_0$',fontsize = 12)
        ax1.set_ylabel('$w_1$',fontsize = 12,labelpad = 15,rotation = 0)
        ax1.axhline(y=0, color='k',zorder = 0,linewidth = 0.5)
        ax1.axvline(x=0, color='k',zorder = 0,linewidth = 0.5)
        
        ax1.set_xticks(np.arange(round(xmin),round(xmax)+1))
        ax1.set_yticks(np.arange(round(ymin),round(ymax)+1))
        
        # set viewing limits
        ax1.set_xlim(xmin,xmax)
        ax1.set_ylim(ymin,ymax)
        
        # if weight history are included, plot on the contour
        if 'weight_history' in kwargs:
            self.w_hist = kwargs['weight_history']
            self.draw_weight_path(ax1)
        
        # plot
        plt.show()
        
    ### function for drawing weight history path
    def draw_weight_path(self,ax):
        # make color range for path
        s = np.linspace(0,1,len(self.w_hist[:round(len(self.w_hist)/2)]))
        s.shape = (len(s),1)
        t = np.ones(len(self.w_hist[round(len(self.w_hist)/2):]))
        t.shape = (len(t),1)
        s = np.vstack((s,t))
        colorspec = []
        colorspec = np.concatenate((s,np.flipud(s)),1)
        colorspec = np.concatenate((colorspec,np.zeros((len(s),1))),1)

        ### plot function decrease plot in right panel
        for j in range(len(self.w_hist)):  
            w_val = self.w_hist[j]
            g_val = self.g(w_val)

            # plot each weight set as a point
            ax.scatter(w_val[0],w_val[1],s = 30,c = colorspec[j],edgecolor = 'k',linewidth = 2*math.sqrt((1/(float(j) + 1))),zorder = 3)

            # plot connector between points for visualization purposes
            if j > 0:
                w_old = self.w_hist[j-1]
                w_new = self.w_hist[j]
                g_old = self.g(w_old)
                g_new = self.g(w_new)
         
                ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = colorspec[j],linewidth = 2,alpha = 1,zorder = 2)      # plot approx
                ax.plot([w_old[0],w_new[0]],[w_old[1],w_new[1]],color = 'k',linewidth = 2 + 0.4,alpha = 1,zorder = 1)      # plot approx
             
    ### function for creating contour plot
    def draw_contour_plot(self,ax,num_contours,xmin,xmax,ymin,ymax):
            
        #### define input space for function and evaluate ####
        w1 = np.linspace(xmin,xmax,400)
        w2 = np.linspace(ymin,ymax,400)
        w1_vals, w2_vals = np.meshgrid(w1,w2)
        w1_vals.shape = (len(w1)**2,1)
        w2_vals.shape = (len(w2)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = np.asarray([ self.g(np.reshape(s,(2,1))) for s in h])

        w1_vals.shape = (len(w1),len(w1))
        w2_vals.shape = (len(w2),len(w2))
        func_vals.shape = (len(w1),len(w2)) 

        ### make contour right plot - as well as horizontal and vertical axes ###
        # set level ridges
        levelmin = min(func_vals.flatten())
        levelmax = max(func_vals.flatten())
        cut = 0.4
        cutoff = (levelmax - levelmin)
        levels = [levelmin + cutoff*cut**(num_contours - i) for i in range(0,num_contours+1)]
        levels = [levelmin] + levels
        levels = np.asarray(levels)
   
        a = ax.contour(w1_vals, w2_vals, func_vals,levels = levels,colors = 'k')
        b = ax.contourf(w1_vals, w2_vals, func_vals,levels = levels,cmap = 'Blues')
        
    ### draw surface plot
    def draw_surface(self,ax,xmin,xmax,ymin,ymax):
        #### define input space for function and evaluate ####
        w1 = np.linspace(xmin,xmax,200)
        w2 = np.linspace(ymin,ymax,200)
        w1_vals, w2_vals = np.meshgrid(w1,w2)
        w1_vals.shape = (len(w1)**2,1)
        w2_vals.shape = (len(w2)**2,1)
        h = np.concatenate((w1_vals,w2_vals),axis=1)
        func_vals = np.asarray([ self.g(np.reshape(s,(2,1))) for s in h])

        ### plot function as surface ### 
        w1_vals.shape = (len(w1),len(w2))
        w2_vals.shape = (len(w1),len(w2))
        func_vals.shape = (len(w1),len(w2))

        ax.plot_surface(w1_vals, w2_vals, func_vals, alpha = 0.1,color = 'w',rstride=25, cstride=25,linewidth=1,edgecolor = 'k',zorder = 2)

        # plot z=0 plane 
        ax.plot_surface(w1_vals, w2_vals, func_vals*0, alpha = 0.1,color = 'w',zorder = 1,rstride=25, cstride=25,linewidth=0.3,edgecolor = 'k') 
        
        # clean up axis
        ax.xaxis.pane.fill = False
        ax.yaxis.pane.fill = False
        ax.zaxis.pane.fill = False

        ax.xaxis.pane.set_edgecolor('white')
        ax.yaxis.pane.set_edgecolor('white')
        ax.zaxis.pane.set_edgecolor('white')

        ax.xaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.yaxis._axinfo["grid"]['color'] =  (1,1,1,0)
        ax.zaxis._axinfo["grid"]['color'] =  (1,1,1,0)