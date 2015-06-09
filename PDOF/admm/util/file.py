# -*- coding: utf-8 -*-
"""
Created on Fri May 22 00:02:44 2015

@author: shterev
"""

import h5py
import numpy as np
import os
import sys

"""  https://confluence.slac.stanford.edu/display/PSDM/How+to+access+HDF5+data+from+Python   """

def print_hdf5_file_structure(file_name) :
    """Prints the HDF5 file structure"""
    file = h5py.File(file_name, 'r') # open read-only
    item = file #["/Configure:0000/Run:0000"]
    print_hdf5_item_structure(item)
    file.close()
 
def print_hdf5_item_structure(g, offset='    ') :
    """Prints the input file/group/dataset (g) name and begin iterations on its content"""
    if   isinstance(g,h5py.File) :
        print g.file, '(File)', g.name
 
    elif isinstance(g,h5py.Dataset) :
        print '(Dataset)', g.name, '    len =', g.shape #, g.dtype
 
    elif isinstance(g,h5py.Group) :
        print '(Group)', g.name
 
    else :
        print 'WARNING: UNKNOWN ITEM IN HDF5 FILE', g.name
        sys.exit ( "EXECUTION IS TERMINATED" )
 
    if isinstance(g, h5py.File) or isinstance(g, h5py.Group) :
        for key,val in dict(g).iteritems() :
            subg = val
            print offset, key, #,"   ", subg.name #, val, subg.len(), type(subg),
            print_hdf5_item_structure(subg, offset + '    ')
 
if __name__ == "__main__" :
    
   file_base = '../../data/EVs/home/'

   os.chdir(file_base)

   print os.getcwd()


   file_name = '1' + '.mat'
   
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
            
       print key + " : \n", tmp
       
       
   #with h5py.File('the_filename', 'r') as f:
   # my_array = f['array_name'][()]
    
   f.close()   
   
   
   
   sys.exit ( "End of test" )


