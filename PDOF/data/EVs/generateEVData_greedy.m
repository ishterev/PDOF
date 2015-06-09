%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generateEVDatagreedy.m
% 
% This script reads the EVs.mat where the driving vectors and energy requirements of
% EVs are saved. Then it defines the A, R, d, B, S_min and S_max parameters needed
% for the EV optimization. For each EV a mat file is saved in the folder 
% greedy.
%
% Output: Ndes number of EV mat data files saved in the greedy folder.
% To save results please make sure the save line is uncomment (Ln 217)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load data
load('EVs.mat')

Ndes= 100;          % number of Evs desired
cars= size(E0s,1);  % number of cars
count=0;%7308;  % starts counting and naming EVs from count+1
count2=0;       % extra counter for debugging
notContinious=0 ; % count number of EVs without a continious overnight profile
  
% parameters
xmax=4;  % maximal charging power
nu=0.93; % efficiency parameter
cap= 22; % battery capacity [kWh]
deltaT= 15*60; % x in seconds

% For each EV

for i = 1:cars
%% Preliminary calculations
    
    % define desired energy
   d=D(i,:);  % driving profile vector
   Edes= E0s(i,:)'; % required energy for travelling
   indx=find(Edes==0,1,'first');
   if isempty(indx)
       indx=12;
   end
   Edes=Edes(1:indx-1);
   
  
   %%%%% Define a preliminary matrix A indicates times when EV is not
   %%%%% driving
   indx= find(d==0,1);       %find first time parked
   park=1;
   A=zeros(1,96);
   for k=1:96
       
       if d(k)==1
           A(park,k)=1;
           
           flag=0 ;     % this is not a new parking
       else
           if flag==0 % this is a new parking 
               park=park+1;
           end
           A(park,k+1:end)=zeros(1,96-k);
           flag=1;
          
       end
   end
    
   
 
%% Definition of energy requirement parameters

% EVs without continious driving profiles overnight are ignored !
if d(1)==0 || d(end) == 0  % there is NO smooth overnight trasition 
notContinious=notContinious+1;
    continue;          %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% PROBLEMS ? uncomment this !     
else
    %make overnight the the last charging
    A(end,:)= A(1,:) + A(end,:);
    A= A(2:end,:);

end   
    
        Edestemp=Edes;             % make a copy of the original Edes

    %definitions
    timeForCharging=sum(A')'; % availible time for charging 
    Efull= sum(Edes);        % required energy to make all trips
    smax=0.85*cap;          % maximal charge in the battery 

    NumFullCharged= ceil(Efull/smax);     % number of times the battery needs to be fully charged for this trips
    deltaSmax= xmax*nu*deltaT/3600; % maximal charge in one time slot [kWh]
    numTrips= size(A,1); % number of trips
    
   
    R=[];   % reset R
    EnotCharged=0;
    
     s=smax;  % start with a fully charged battery, e.g. Emax
    
    for j=1:numTrips
        
        if j==numTrips
            Edes(j)= Edes(j) + EnotCharged;  % charge the remaining amount
        end
    
    s=s-Edes(j); % already consummed energy from the last trip

    
    if(s<0) % This trip is not possible
        s=0*cap;
        count2=count2+1;
        Doible=0;
        break; % get out of the for loop
    end
    Doible=1;
    % maximal possible in parked time until next trip
    deltaSpossible=deltaSmax * timeForCharging(j);
    s= s + deltaSpossible ;
     if s>= smax
         s=smax;
         R(j)= Edes(j)*3600/(nu*deltaT);
         
     else
         R(j)= deltaSpossible*3600/ (nu*deltaT) ;
         
         EnotCharged=EnotCharged + Edes(j) - deltaSpossible; % what should have been charged but wasnt
     end
    
    end
    
   % Debugging
   tol=1e-3;
   if norm(Efull- sum(R)*nu*15/60 ) >tol && Doible      %Not a doible profile with this EV
       disp('We have a problem...ignoring this EV')
     continue
       
   else
       
%% Definition of state contraints parameters, only required       
       
      
       
             Edes= Edestemp;     

            % Define max and minimal battery charging
            indx_B1= find(A(1:end-1,:)'==1);  % find the column indexes of the ones
            indx_B2= find(A(end,:)'==1);  % find the column indexes of the ones
            
%           modify the indexes of the last time parked
            lastParked=find(A(end,:)==0,1,'last')+1;
              partition = find(indx_B2==lastParked);
              
              indx_B2= [indx_B2(partition:end); indx_B2(1:partition-1)];
              
            
            indx_B=[indx_B1; indx_B2];
            
            
            indx_d=find(d==1);      % find indexes of ones in d
            indx_B=mod(indx_B,96);            % modify B indexes
            
            repair= find(indx_B==0)   ;    % repair error from mod(96,96)
            
            indx_B(repair)= 96*ones(size(repair));      % make 0's to ones   
            
            numberOfones=length(indx_d);            % number of rows of B
            
            B= zeros(numberOfones,96);         % define B
            
            
        for p=1:numberOfones
            
            
            B(p,indx_B(p))=1;
            
            if p>1
                B(p,:)=B(p-1,:) + B(p,:);
            end
            
            
        end
            
        
        % Error control
        
            if any(B)>1
                fpritf('there is a problem...ignore this EV')
                pause
            end
        
        % define max and min energy that can be used by the battery
        
        S_max=[];
        S_min=[];
        for p=1:numTrips
           numberOfslots= length(find(A(p,:)==1) ) ;     % number of slots with one on each trip
           
           S_min=[S_min; ones(numberOfslots,1)*(sum(Edes(1:p))-smax)*3600/(nu*deltaT) ];
           S_max=[S_max; ones(numberOfslots,1)*(sum(Edes(1:p)))*3600/(nu*deltaT) ]; 
            
        end
            
           % error check
           
           if any(S_min)<=0 || any(S_max)==0
               
               fprintf('Funny one of the Ss is 0')
               pause
           end

       
       
       
       
       
    %% SAVE PARAMETERS
    count=count+1 ;
    name= ['greedy/' num2str(count) '.mat'] ;
%   save(name,'A','d','R','B','S_min','S_max');

     if count==Ndes
         fprintf('\n\t We got %i  EVs !\n', Ndes)
         break;
     end
     
     
   end
    

   
end
fprintf('DONE. \n')