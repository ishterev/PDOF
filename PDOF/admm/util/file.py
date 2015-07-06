# -*- coding: utf-8 -*-
"""
Created on Fri May 22 00:02:44 2015

@author: shterev
"""

import h5py
import os
import sys

"""  https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python   """
            

def loadEV(strategy, idx):
    
    file_base = '../data/EVs/' + strategy + '/'
    os.chdir(file_base)
    file_name = str(idx) + '.mat' #tr(idx).encode('utf-8')
    f = h5py.File(file_name, 'r') # open read-only
        
    return f # f.close()
    
def loadAggr():
    
    file_base = '../data/Aggregator/'    
    os.chdir(file_base)    
    file_name = 'aggregator.mat'
    f = h5py.File(file_name, 'r') # open read-only
    
    return f # f.close() 
    

 
if __name__ == "__main__" :   
    
   file_base = '../../data/Aggregator/' #EVs/home/
   os.chdir(file_base)
   file_name = 'aggregator.mat' #'1.mat'
   
   # data = f.get('data/variable1') 
   #data = np.array(data) # For converting to numpy array

#f = h5py.File(file_name, 'r') # open read-only

#print f.name

#keys = f.keys

#print keys

#f.close()

   #print_hdf5_file_structure(file_name)
   
   f = h5py.File(file_name, 'r') # open read-only
   
   #print f.items()
   
   vars = f.items() #list
   
   for key,val in f.items() :
       
       tmp = f[key][()]
            
       print key + " : \n" , tmp
       
       
   #with h5py.File('the_filename', 'r') as f:
   # my_array = f['array_name'][()]
    
   f.close()   
   
   
   
   sys.exit ( "End of test" )


