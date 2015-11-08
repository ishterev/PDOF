load('aggregator.mat')

%D = data.D
%D = 10 * D


name= ['aggregator.mat'] ;
save(name,'D','price','re','-v7');