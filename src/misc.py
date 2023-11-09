import matplotlib.pyplot as plt
import matplotlib
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np
import tensorflow as tf
from tqdm import tqdm as tqdm

class Plotter():
    def __init__(self) -> None:
        import os
        os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"   
        pass
    
    def _get_mgridDict(self,x_min=-0.15,x_max=0.15,nx=25,y_min=-0.15,y_max=0.15,ny=25,z_min=0.3,z_max=0.65,nz=49):
        X,Y,Z= np.mgrid[x_min:x_max:nx*1j, y_min:y_max:ny*1j, z_min:z_max:nz*1j]
        return {'X':X,'Y':Y,'Z':Z}

    def _maxProj(self,x,collapse,cmap='jet',colorbar=False, transpose=False, cmin=0,cmax=1,origin='lower',fontsize=15, scale='lin'):
        def dB(x):
            return 20*np.log10(np.abs(x))

        matplotlib.rcParams.update({'font.size': fontsize})

        if scale=='lin':
            x=x
        elif scale=='log':
            x=dB(x)

        x2D=np.max(x,axis=collapse)

        if transpose:
            plt.imshow(x2D.T,cmap=cmap,aspect=(x2D.shape[0]/x2D.shape[1]),origin=origin)
            
            if collapse%3 == 2:
                plt.ylabel('y (m)')
                plt.xlabel('x (m)')
                plt.yticks([0,12,24],['-0.15','0','0.15'])
                plt.xticks([0,12,24],['-0.15','0','0.15'])
            elif collapse%3 == 1:        
                plt.ylabel('z (m)')
                plt.xlabel('x (m)')
                plt.yticks([0,24,48],['-0.45','0','0.65'])
                plt.xticks([0,12,24],['-0.15','0','0.15'])
            elif collapse%3 == 0:
                plt.xlabel('y (m)')
                plt.ylabel('z (m)')
                plt.xticks([0,12,24],['-0.15','0','0.15'])
                plt.yticks([0,24,48],['0.35','0.5','0.65'])
        
        else:
            plt.imshow(x2D,cmap=cmap,aspect=(x2D.shape[1]/x2D.shape[0]),origin=origin)

            if collapse%3 == 2:
                plt.xlabel('y (m)')
                plt.ylabel('x (m)')
                plt.xticks([0,12,24],['-0.15','0','0.15'])
                plt.yticks([0,12,24],['-0.15','0','0.15'])
            elif collapse%3 == 1:        
                plt.xlabel('z (m)')
                plt.ylabel('x (m)')
                plt.xticks([0,24,48],['-0.45','0','0.65'])
                plt.yticks([0,12,24],['-0.15','0','0.15'])
            elif collapse%3 == 0:
                plt.ylabel('y (m)')
                plt.xlabel('z (m)')
                plt.yticks([0,12,24],['-0.15','0','0.15'])
                plt.xticks([0,24,48],['0.35','0.5','0.65'])

        if colorbar:
            plt.colorbar(fraction=0.047,orientation='horizontal', location = 'top')        
            plt.clim(cmin,cmax)

    def _maxProj_defaults(self,x,collapse,cbar=False):
        if collapse==2:
            self._maxProj(x=x,collapse=2,cmap='jet', scale='log',transpose=False,origin='upper',colorbar=cbar,cmin=-40,cmax=0)
        elif collapse==0:
            self._maxProj(x=x,collapse=0,cmap='jet', scale='log',transpose=True,origin='lower',colorbar=False,cmin=-40,cmax=0)

    def maxProjection(self,x):

        plt.subplot(2,1,1)
        self._maxProj_defaults(x,2,cbar='True')
        plt.subplot(2,1,2)
        self._maxProj_defaults(x,0)
        plt.tight_layout()

    def sceneshow(self,x,perspective='simulated',showscale=False,font_size=12,scount=15,isomin=0,isomax=1):

        def quick_sceneshow(mgrid_dict, x,scount):

            fig = make_subplots(
                rows=1, cols=1,
                specs=[[{'type': 'volume'}]])

            fig.add_trace(go.Volume(
                x=mgrid_dict['X'].flatten(),
                y=mgrid_dict['Y'].flatten(),
                z=mgrid_dict['Z'].flatten(),
                value=x.flatten(),
                isomin=isomin,
                isomax=isomax,
                opacity=0.2, # needs to be small to see through all surfaces
                surface_count=scount # needs to be a large number for good volume rendering
                ), row=1, col=1)
            return fig
        

        def transpose_flip_xest_kwargs(x_est,transpose=(0,1,2),flip=[]):
            assert len(transpose)==3

            ns=[25,25,49]
            lims=[[-0.15,0.15],[-0.15,0.15],[0.35,0.65]]
            order2desc=['x','y','z']
            kwargs=dict()

            for i,order_idx in enumerate(transpose):
                key_min = order2desc[i]+'_min'
                key_max = order2desc[i]+'_max'
                key_n = 'n'+order2desc[i]

                kwargs[key_min]=lims[order_idx][0]
                kwargs[key_max]=lims[order_idx][1]
                kwargs[key_n]=ns[order_idx]

            x_est=np.transpose(x_est,axes=transpose)

            x_est=np.flip(x_est,axis=flip)

            return x_est,kwargs

        if perspective =='simulated':
            fig =quick_sceneshow(mgrid_dict=self._get_mgridDict(),scount=scount,x=x)
            camera = dict(center=dict(x=0, y=0, z=-0.1))
            fig.update_layout(scene_camera=camera)
            fig.update_traces(showscale=showscale)
            if showscale:
                fig.update_layout(width=800, height=700, margin=dict(t=0, r=0, l=0, b=0))
            else:
                fig.update_layout(width=700, height=700, margin=dict(t=0, r=0, l=0, b=0))
            fig.update_layout(
                    font=dict(
                        # family="Times New Roman",
                    size=font_size,
                ),
                    
                    scene_zaxis = dict(
                    tickmode = 'array',            
                    tickvals = [0.4,0.5,0.6],
                    ticktext = ['0.4','0.5','0.6'],
                    title='z',
                    ),
                    scene_yaxis = dict(
                    tickmode = 'array',
                    tickvals = [-0.1,0,0.1],
                    ticktext =['-0.1','0','0.1'],
                    title='y',
                    ),
                    scene_xaxis = dict(
                    tickmode = 'array',
                    tickvals = [-0.1,0,0.1],
                    ticktext =['-0.1','0','0.1'],
                    title='x',
                    ))
        elif perspective =='experimental':
            transpose=(2,1,0)
            flip = [2,1]

            x_est,kwargs = transpose_flip_xest_kwargs(x,transpose,flip)

            mgrid_dict=self._get_mgridDict(**kwargs)
            fig =quick_sceneshow(mgrid_dict=mgrid_dict,scount=scount,x=x_est)

            d=0.9
            projection='perspective'
            fig.update_scenes(camera=dict(eye=dict(x=-2.5*d, y=-1.5*d, z=1.5*d),projection=dict(type=projection)))
            camera = dict(
            center=dict(x=0, y=0, z=-0.2))
            fig.update_layout(scene_camera=camera)

            fig.update_traces(showscale=showscale)
            fig.update_layout(width=500, height=500, margin=dict(t=0, r=0, l=0, b=0))
            fig.update_layout(margin=dict(t=0, r=0, l=0, b=0),
                    font=dict(
                        # family="Times New Roman",
                    size=15,
                    color="black"),
                    scene_zaxis = dict(
                    tickmode = 'array',
                    tickvals = [-0.13,0,0.15],
                    ticktext =['0.15','0','-0.15'],
                    title='x (m)',
                    ticks="inside",    
                    tickangle=0,

                    ),
                    scene_yaxis = dict(
                    tickmode = 'array',
                    tickvals = [-0.15,0,0.13],
                    ticktext =['0.15','0','-0.15'],
                    title='y (m)',
                    tickangle=0,
                    ticks="inside",    
                    ),
                    scene_xaxis = dict(
                    tickmode = 'array',
                    tickvals = [0.35,0.5,0.65],
                    ticktext = ['0.35','0.50','0.65'],
                    title='z (m)',
                    tickangle=0,
                    ticks="inside",    
                    ))
        
        return fig
    
def addNoise(SNRdB,signal):
    def _addNoise(SNRdB,signal):
    
        signal_power = np.mean(np.abs(signal)**2)
        snr_lin = 10**(SNRdB/10)
        variance = signal_power/snr_lin
        std=np.sqrt(variance)

        nim = np.random.randn(*signal.shape)*1j*np.sqrt(1/2)
        nre = np.random.randn(*signal.shape)*np.sqrt(1/2)
        
        n=std*(nim+nre)

        return n+signal

    if len(signal.shape)>=4:
        noisy_signal=[]
        for i in tqdm(range(signal.shape[0])):
            noisy_signal.append(_addNoise(SNRdB,signal[i,...])[None,...])
        
        return np.concatenate(noisy_signal,0)
    else:
        return _addNoise(SNRdB,signal)
    
def computeSNRdB(signal,noise):
    return 10*np.log10(np.sum(np.abs(signal)**2)/np.sum(np.abs(noise)**2))

def computePSNR(estimate,original,max_val=1):
    return tf.image.psnr(np.squeeze(estimate), np.squeeze(original), max_val = max_val, name=None).numpy()

def computeSSIM(estimate,original,max_val=1):
    return tf.image.ssim(np.squeeze(estimate),np.squeeze(original), max_val=max_val, filter_size=11,
                          filter_sigma=1.5, k1=0.01, k2=0.03).numpy()