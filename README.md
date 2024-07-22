# LLPCA
- Local Label Propagation (LLP).
- Local Label Propagation with Class Attention (LLPCA).

## Parameter:
The data sets and  number of training samples can be selected within the code.  
- DataSetName in {'Indianpines', 'PaviaU', 'HongHu'}
- number indicates the number of training samples selected for each category.
- d indicates the radius.
- k indicates the minimum threshold for the number of allowed propagations in CA mask.(only for LLPCA)

## Usage:
For example, when DataSet Indianpines takes 5-30 training sample points for each class, the method of calling LLP and LLPCA algorithm for "Indianpines" is as follows:
- LLP("Indianpines",[5,10,15,20,25,30],2);
- LLPCA("Indianpines",[5,10,15,20,25,30],1,10);
