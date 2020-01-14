# Double Descent Curve
This is the code to reproduce Figure 5 and 6 of "The double descent risk curve" blog post on Medium.

This script recovers the double descent curve using random projections plus the `RidgeClassifier` from `scikit-learn`. 
It is possible to choose between a synthetic optical processing unit (OPU) and the real OPU. 
To request access to our cloud and try our optics-based hardware, contact us: https://www.lighton.ai/contact-us/

# to run the script
```
python ddc_ridgeclassifier.py  # to use synthetic opu on mnist
python ddc_ridgeclassifier.py  -dataset 'cifar10' # to use synthetic opu on cifar10 
python ddc_ridgeclassifier.py -is_real_opu True  # to use opu on mnist with  threshold encoder 
python ddc_ridgeclassifier.py -is_real_opu True  -encoding_method 'autoencoder' # to use opu on mnist with autoencoder 
python ddc_ridgeclassifier.py -is_real_opu True -dataset 'cifar10' # to use opu on cifar10 with  threshold encoder 
python ddc_ridgeclassifier.py -is_real_opu True  -encoding_method 'autoencoder'  -dataset 'cifaro10'# to use opu on cifar10 with autoencoder 
```

Running `ddc_ridgeclassifier.py` outputs a `.pkl` file. To plot the results using this file look at the `plot.ipynb` example.  

