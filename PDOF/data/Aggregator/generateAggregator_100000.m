data = load('aggregator_10000.mat')

D = data.D
D = 10 * D


name= ['aggregator_100000.mat'] ;
save(name,'D','price','re','-v7');