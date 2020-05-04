# Double Descent Curve
This is the code to reproduce Figure 5 and 6 of ["The double descent risk curve"](https://medium.com/@LightOnIO/beyond-overfitting-and-beyond-silicon-the-double-descent-curve-18b6d9810e1b) blog post on Medium.

This script recovers the double descent curve using random projections plus the `RidgeClassifier` from `scikit-learn`. 
It is possible to choose between a synthetic optical processing unit (OPU) and the real OPU. 
To request access to our cloud and try our optics-based hardware, contact us: https://www.lighton.ai/contact-us/

## Access to Optical Processing Units

To request access to LightOn Cloud and try our photonic co-processor, please visit: https://cloud.lighton.ai/

For researchers, we also have a LightOn Cloud for Research program, please visit https://cloud.lighton.ai/lighton-research/ for more information.

## Run the experiments
```
python ddc_ridgeclassifier.py  # to use synthetic opu on mnist
python ddc_ridgeclassifier.py  -dataset 'cifar10' # to use synthetic opu on cifar10 
python ddc_ridgeclassifier.py -is_real_opu True  # to use opu on mnist with  threshold encoder 
python ddc_ridgeclassifier.py -is_real_opu True  -encoding_method 'autoencoder' # to use opu on mnist with autoencoder 
python ddc_ridgeclassifier.py -is_real_opu True -dataset 'cifar10' # to use opu on cifar10 with  threshold encoder 
python ddc_ridgeclassifier.py -is_real_opu True  -encoding_method 'autoencoder'  -dataset 'cifaro10'# to use opu on cifar10 with autoencoder 
```

Running `ddc_ridgeclassifier.py` outputs a `.pkl` file. To plot the results using this file look at the `plot.ipynb` example.  

