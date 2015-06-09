%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% generateEVDataHome.m
% 
% This script reads the EVs.mat where the driving vectors and energy requirements of
% EVs are saved. Then it defines the A, R, d, B, S_min and S_max parameters needed
% for the EV optimization. For each EV a mat file is saved in the folder 
% minEnergy.
%
% Output: Ndes number of EV mat data files saved in the minEnergy folder
%
% To save results please make sure the save line is uncomment (Ln 244)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% load data
load('EVs.mat')

Ndes= 100;        % number of EVs desired 
cars= size(E0s,1);  % number of cars
count=0; %7171; % starts counting and naiming EVs from count+1
count2=0;       % extra counter for debugging
notContinious=0 ; % count number of EVs without a continious overnight profile
notRight=0; % some problems that occured while parsing the data

% parameters
xmax=4;  % maximal charging power
nu=0.93; % efficiency parameter
cap= 22; % battery capacity [kWh]
deltaT= 15*60; % x in seconds



%% still problems with indexes   (need to modify them manually or with this script)
a=[157 976 1224 1830 2842 3235 3341 4032 4033 4243 5307 5354 5541 5574 5734 5939 6102 6475 6900 7328 8147 8395 9001 ];


% For each EV

for i = 1:cars
%% Preliminary calculations

   d=D(i,:);  % driving profile vector
   Edes= E0s(i,:)'; % required energy for travelling
   indx=find(Edes==0,1,'first');
   if isempty(indx)
       indx=12;
   end
   Edes=Edes(1:indx-1);
   
  
   %%%%% calculate A
   indx= find(d==0,1);
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
           %park=park+1;
       end
   end
    
   
 
   
%% Definition of energy requirement parameters

% EVs without continious driving profiles overnight are ignored !
if d(1)==0 || d(end) == 0  % there is NO smooth overnight trasition 
notContinious=notContinious+1;
    continue;          
else
    %make overnight the the last charging
    A(end,:)= A(1,:) + A(end,:);
    A= A(2:end,:);

end
 
% some problems with data that we just clear
if size(A,1) ~= size(Edes,1)
    notRight=notRight+1;
    continue;
end
    
    %definitions
    smax=0.85*cap;          % maximal charge in the battery that can be used in [kWh]
    deltaSmax= xmax*nu*deltaT/3600; % maximal charge in one time slot [kWh]
    numTrips= size(A,1); % number of trips
    
    s=smax;  % start with a charged battery
    R=[];   % reset R
    EnotCharged=0;
    
    
    
    % Definition of new Variable
    Areq=zeros(size(A));            % temporary varriable for A
    Rreq_min= zeros(size(Edes));    % temporary varriable for R_min
    Rreq_max= zeros(size(Edes));    % temporary varriable for R_max
    
    % start indicator of not feasibility (0 means feasible)
    notFeasible=0; 
   
    for j=1:numTrips -1
        
        if j==1
            Areq(j,:)= A(1,:);
        else
            Areq(j,:)=sum(A(1:j,:));
        end
        
        temp= (sum(Edes(1:j+1))  -smax) * 3600  / (deltaT*nu) ;
        
        % makes minimal minimal requirement positive or 0, guarantee same
        % charge in the battery as when first parked
        if temp<0
            Rreq_min(j)=0;
        else
            Rreq_min(j)=temp ;
        end
        
        Rreq_max(j)= (sum(Edes(1:j)) ) * 3600  / (deltaT*nu) ;
        
        
        % check is feasible
        timeAvaliable= sum(Areq(j,:)) ;
        
       
        if timeAvaliable*xmax < Rreq_min(j)  || Rreq_min(j) > Rreq_max(j)
          %  disp('This profile is not feasible')
            notFeasible=1;
            break
        end
        
            
        
        
    end
    
    % for the last step
    
    Areq(end,:)=sum(A);
    Rreq_max(end)= (sum(Edes) ) * 3600  / (deltaT*nu) ;
    Rreq_min(end)=Rreq_max(end);
    timeAvaliable= sum(Areq(end,:)) ;
    if timeAvaliable*xmax < Rreq_min(j) 
          %  disp('This profile is not feasible')
            notFeasible=1;
      
    end
    
    if ~notFeasible
        %feasible
   
%% Definition of state contraints parameters, only required  
                 

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
                fpritf('there is a problem')
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
           
           if any(S_min)==0 || any(S_max)==0
               
               fprintf('Funny one of the Ss is 0')
               pause
           end
           
           
 %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%% end definig max and min battery charging          
      
           
        
        
        
            
            %% SAVE PARAMETERS
            count=count + 1;
            A=Areq;
            R_max=Rreq_max;
            R_min=Rreq_min;
             name= ['minEnergy/' num2str(count) '.mat'] ;
%             save(name,'A','R_min','R_max', 'B','S_min','S_max','d');
%             

           % are we done ?
          if count==Ndes
                 fprintf('\n\t We got %i  EVs !\n', Ndes)
                 break;
          end
        
            
            
    else
        % not feasible
        count2=count2 + 1;
    end
        
    

    
end



fprintf('\n \n \t DONE.\n')