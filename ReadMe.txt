
This code is mainly used to reproduce the LIDC-IDRI dataset (Table 2-3, Table 5-6 Figure 4-6)

Ablation experiments can be reproduced by adjusting \
            the corresponding parameters according to our paper.

##############
Note that our code has not been carefully organized. 
We promise to collate the code and post it to Github as soon as the paper is accepted.
##############


********** dataset **********
1. Please download the pre-processed dataset from the bottom of the linked page:
    https://github.com/stefanknegt/Probabilistic-Unet-Pytorch
    Finally, you will get a file named "data_lidc.pickle".
    
2. this dataset is also used by Probabilistic Unet, PHiSeg, and SSN

3. Put the data file in the "data" folder.


********** training **********
1. Open the file "run.py"

2. One can change the experiment parameters in the first few lines of the file.

3. Run "python3 run.py".

