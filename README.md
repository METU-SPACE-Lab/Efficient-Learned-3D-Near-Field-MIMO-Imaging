# Efficient physics-based learned reconstruction methods for real-time 3D near-field MIMO radar imaging
Irfan Manisalı, [Okyanus Oral](https://eee.metu.edu.tr/personel/okyanus-oral), and [Figen S. Oktem](https://blog.metu.edu.tr/figeno/) 

This repository contains the official codes for the paper "[**Efficient physics-based learned reconstruction methods for real-time 3D near-field MIMO radar imaging**](https://www.sciencedirect.com/science/article/pii/S105120042300369X)", Digital Signal Processing, vol. 144, pp. 104274, 2024, doi: 10.1016/j.dsp.2023.104274. ([arXiv](https://arxiv.org/abs/2312.16959)).

<table>
<tr><td>Experimental Results <a href="http://www.youtube.com/watch?feature=player_embedded&v=9qTrIXIPdVc">[YouTube]</a></td><td>Simulation Results <a href="http://www.youtube.com/watch?feature=player_embedded&v=Mg4YhEuxVL4">[YouTube]</a></td></tr>
<tr><td><a href="http://www.youtube.com/watch?feature=player_embedded&v=9qTrIXIPdVc
" target="_blank"><img src="http://img.youtube.com/vi/9qTrIXIPdVc/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="300" height="" border="10" /></a></td><td><a href="http://www.youtube.com/watch?feature=player_embedded&v=Mg4YhEuxVL4
" target="_blank"><img src="http://img.youtube.com/vi/Mg4YhEuxVL4/0.jpg" 
alt="IMAGE ALT TEXT HERE" width="300" height="" border="10" /></a></td></tr>
</table>

## Models and Dataset

- You can download the models and the synthetically generated dataset from [here](https://drive.google.com/drive/folders/1De2VF7KIOp3DHYNgCHkcx-fLM0cnTnM6?usp=sharing). (All DNN models, except Deep2S+, are saved as '.h5' files. Deep2S+ is saved with "SavedModel" format.)
- For the experimental data, please refer to [1,2]

**[1]** J. Wang, P. Aubry and A. Yarovoy, "3-D Short-Range Imaging With Irregular MIMO Arrays Using NUFFT-Based Range Migration Algorithm," in _IEEE Transactions on Geoscience and Remote Sensing_, vol. 58, no. 7, pp. 4730-4742, July 2020, doi: 10.1109/TGRS.2020.2966368.

**[2]** Jianping Wang, January 10, 2020, "EM data acquired with irregular planar MIMO arrays", IEEE Dataport, doi: https://dx.doi.org/10.21227/src2-0y50.

## FAQs
To instantiate the reconstructors, you don't have to load DNN weights separately and can directly pass the DNN paths at initialization. However, to prevent dependency issues with TensorFlow, you are recommended to first load the DNNs for 3D/2D U-Net and ResNet. The suggested initializations are given below:

    from src import src # source files will be included upon publication
    import numpy as np
    A = np.load('path_to_observation_matrix.npy') # Required for Deep2S and CV-Deep2S
   
    # Deep2S with U-Net 3D, with U-Net 2D, with ResNet
    DNN=src.get_UNet3D() # src.get_UNet2D() # src.get_ResNet()
    DNN.load_weights('path_to_net.h5')
    reconstructor = src.Deep2S(DNN=DNN,A=A)

    # CV-Deep2S
    DNN=src.get_CV_UNet() 
    DNN.load_weights('path_to_net.h5')
    reconstructor = src.CV_Deep2S(DNN=DNN,A=A)
    
    # DeepDI
    reconstructor = src.DeepDI(DNN='path_to_net.h5')
    
    # Deep2S+
    reconstructor = src.Deep2SP(DNN='path_to_Deep2SP') # (not ".h5" file, "SavedModel" format)
    
## Citation
Please cite the following paper when using this code or data:

    @article{manisali2024efficient,
    title = {Efficient physics-based learned reconstruction methods for real-time {3D} near-field {MIMO} radar imaging},
    journal = {Digital Signal Processing},
    volume = {144},
    pages = {104274},
    year = {2024},
    issn = {1051-2004},
    doi = {https://doi.org/10.1016/j.dsp.2023.104274},
    url = {https://www.sciencedirect.com/science/article/pii/S105120042300369X},
    author = {Irfan Manisali and Okyanus Oral and Figen S. Oktem}
    }

