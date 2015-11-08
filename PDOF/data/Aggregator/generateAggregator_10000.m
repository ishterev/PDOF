load('aggregator.mat')

%D = data.D
%D = 10 * D


name= ['aggregator_10000.mat'] ;
save(name,'D','price','re','-v7');