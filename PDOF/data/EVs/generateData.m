%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generateData.m
% 
% Reads the file Data.txt that constains the mobility data in the formats:
% 
% index   start date      start time      end date    end time    distance
% 
% This data is read and ordered to produce the driving profiles vectors of the cars
% odered in the matrix D (car X time slots) and the required energy in the matrix E0
% (car X number of trip) based on the EV specs.
%
% Output: EVs.mat
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% EV specs
EVPowerConsumption=0.15; % EV power consumption (Siemens Move E) [kWh/km]


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% load text data
fid= fopen('DATA.txt');
C=textscan(fid, '%f\t%s %s %s %s %f\t%f');
fclose(fid);


% Name and order data
indx=C{1};              % index (not relevant)
startDate=C{2}(:);      % start date of trip
startTime=C{3}(:);      % start time of trip
endDate=C{4}(:);        % end date of trip
endTime=C{5}(:);        % end time of trip
distance=C{6}(:);       % distance of trip

% start variables
cars=1;                % number of cars
indxOld=-1;            % index
E0s=[];                % vector of energy requirements 
E0=[];                 % current E0 for this car
D=ones(1,96);          % VEctor of all driving profiles
endTimeSlot_old=-1;    % index of 
%trip=0;                % start number of trips with 0
startTimeSlot_old=-1;


for i=1:length(indx)
    
    
 % Define time slots in hours and minutes
   startHour=str2double(startTime{i}(1:2));
   startMinute= str2double(startTime{i}(4:5));
   endHour=str2double(endTime{i}(1:2));
   endMinute= str2double(endTime{i}(4:5));
   

     % in terms of driving vector
     m=ceil(startMinute/15);
     if m==0 % for the case of having 0 minutes
         m=1;
     end
     startTimeSlot= startHour*4  + m;
     
       
     m= ceil(endMinute/15);
     if m==0 % for the case of having 0 minutes
         m=1;
     end
     endTimeSlot= endHour*4 + m;
      
      
  
    
   % Is this a new car ?
   if startTimeSlot < startTimeSlot_old  % yes, if the previos starting time is bigger than the new, since this is a new day
        
        E0s(cars,1:length(E0))= E0;
        E0=[];
        indxOld=-1;
        cars=cars+1;
        D=[D; ones(1,96)];
        endTimeSlot_old=-1;
        startTimeSlot_old=-1;
        trip=0;
   else
       indxOld=indx(i) ; % no
   end
   
   %%% Driving prfile vector  
       % case a trip goes to the next day drop it
       if endTimeSlot>96
               continue;  % this one trip happens very rarely...therefor we can ignore it
       else
       
           
       
       driving= zeros(1, endTimeSlot - startTimeSlot + 1);
       
       D(cars, startTimeSlot : endTimeSlot) = driving ;
       
       end
       
    % Define Energy needs   
       % see if the travels overlap, because they are less than 15 min
       % appart
            e0=distance(i)*1.60934 * EVPowerConsumption; % required energy [kWh]= distance in miles * miles2km * EV power consumption
       if startTimeSlot <= endTimeSlot_old +1 % overlap ? yes
           
           %trip=trip;
           
            E0(end)= E0(end) + e0;
       else
           %trip=trip+1; 
           % no
           E0=[E0 e0];
           
           
       end
        
    startTimeSlot_old = startTimeSlot;
    endTimeSlot_old= endTimeSlot;
   
   
  
   
   
end
%save the last one and save EVs file
E0s(cars,1:length(E0))= E0;


%save EVs.mat E0s D

