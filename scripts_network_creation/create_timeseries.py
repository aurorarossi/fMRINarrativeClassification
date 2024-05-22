import nibabel as nib
from nilearn import datasets
from nilearn import image
import numpy as np
from scipy import signal
from scipy import stats
import seaborn as sea


names = np.loadtxt("to_process_schema4.txt", dtype="str")

# Load atlas
dataset_atlas = datasets.fetch_atlas_schaefer_2018(n_rois=100, yeo_networks=7, resolution_mm=2, data_dir=None, base_url=None, resume=True, verbose=1)
loaded_atlas = nib.load(dataset_atlas.maps)
atlas = loaded_atlas.get_fdata()
b,a = signal.butter(1, [0.01,0.08], btype='bandpass',output='ba' )

for name in names:
    print(name)
    # Load subject
    sub_file = f"derivatives/{name}.nii.gz"
    sub = nib.load(sub_file)
    arraydata = sub.get_fdata()

    number_time = np.shape(arraydata)[-1]

    # Filtering 
    filtered = signal.filtfilt(b,a,arraydata,axis=-1,padtype='even') 

    # Extract timeseries
    summ=0
    N=0
    timeseries=[]
    label0=[]
    l=0
    for h in range(100):
        for i in range(atlas.shape[0]):
            for j in range(atlas.shape[1]):
                for k in range(atlas.shape[2]):
                    if atlas[i,j,k]==h:
                        N+=1
                        summ+=filtered[i,j,k,:]
        if N!=0:
            timeseries.append(summ/N)
        else:
            timeseries.append(np.zeros(number_time))
            label0.append(l)
        summ=0
        N=0
        l+=1
    _,_,c = name.split('/')
    

    np.save(f"timeseries/schema-run4-100/{c}_timeseries_7networks.npy",timeseries)
