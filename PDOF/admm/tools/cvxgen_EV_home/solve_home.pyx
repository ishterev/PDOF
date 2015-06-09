# -*- coding: utf-8 -*-
"""
Created on Fri May 29 07:30:00 2015

@author: shterev
"""

from csolve_home cimport *
cimport params

cimport numpy as np
import numpy as np

cdef class Solver:
    '''cdef Vars vars
    cdef Params params
    cdef Workspace work
    cdef Settings settings
    
    cdef double [:, :] A # 96x1
    cdef double [:, :] R # 1x1
    cdef double [:, :] d # 96x1 '''

    def __cinit__(self, double [:, :] A, double [:, :] R, double [:, :] d):
        self.A = A
        self.R = R
        self.d = d

    # In this function, load all problem instance data.
    #          self.params.A[i] = ...;
    cdef void load_data(self, double [:, :] xold, double [:, :] u, double [:] xmean, params p): 
            
         #define optimization parameters
         cdef double [:, :] K = xold - xmean - u;       # Normalization parameter

         # contrained least square optimization parameters
         cdef double [:, :] dd = p.rho/ (2*p.gamma*p.alpha + p.rho) * K
        
         #solve with cvxgen
         self.params.Aeq= self.A
         self.params.beq= self.R #R.T
         self.params.lb= self.d * p.xmin # d.T*xmin
         self.params.ub= self.d * p.xmax # d.T*xmax
         self.params.d=dd


    cdef double[:, :] use_solution(self, Vars vars):
    # In this function, use the optimization result.
    #    ... = vars.x[i];
         return np.asarray(<double[:, :]> vars.x) #np.float64_t

    cdef double[:, :] solve(self, double [:, :] xold, double [:, :] u, double [:] xmean, params p):  
        set_defaults()  # Set basic algorithm parameters.
        setup_indexing()

        load_data(xold, u, xmean, p)

        # Solve our problem at high speed!
        cdef int num_iters = solve()
        # Recommended: check work.converged == 1.

        return use_solution(vars)
        
        
        
        
        
        
        
        
''''''''''''''''''''''' TEST ''''''''''''''''''''''''''''''''''''''''


    cpdef double[:, :] test(self): 
        set_defaults()  # Set basic algorithm parameters.
        setup_indexing()

        load_default_data()

        # Solve our problem at high speed!
        cdef int num_iters = solve()
        # Recommended: check work.converged == 1.

        return use_solution(vars)
        


   cdef void load_default_data(self): 
        self.params.d[0] = 0.203191610298302;
        self.params.d[1] = 0.832591290472419;
        self.params.d[2] = -0.836381044348223;
        self.params.d[3] = 0.0433104207906521;
        self.params.d[4] = 1.57178781739062;
        self.params.d[5] = 1.58517235573375;
        self.params.d[6] = -1.49765875814465;
        self.params.d[7] = -1.17102848744725;
        self.params.d[8] = -1.79413118679668;
        self.params.d[9] = -0.236760625397454;
        self.params.d[10] = -1.88049515648573;
        self.params.d[11] = -0.172667102421156;
        self.params.d[12] = 0.596576190459043;
        self.params.d[13] = -0.886050869408099;
        self.params.d[14] = 0.705019607920525;
        self.params.d[15] = 0.363451269665403;
        self.params.d[16] = -1.90407247049134;
        self.params.d[17] = 0.235416351963528;
        self.params.d[18] = -0.962990212370138;
        self.params.d[19] = -0.339595211959721;
        self.params.d[20] = -0.865899672914725;
        self.params.d[21] = 0.772551673251985;
        self.params.d[22] = -0.238185129317042;
        self.params.d[23] = -1.37252904610015;
        self.params.d[24] = 0.178596072127379;
        self.params.d[25] = 1.12125905804547;
        self.params.d[26] = -0.774545870495281;
        self.params.d[27] = -1.11216846427127;
        self.params.d[28] = -0.448114969777405;
        self.params.d[29] = 1.74553459944172;
        self.params.d[30] = 1.90398168989174;
        self.params.d[31] = 0.689534703651255;
        self.params.d[32] = 1.61133643415359;
        self.params.d[33] = 1.38300348517272;
        self.params.d[34] = -0.488023834684443;
        self.params.d[35] = -1.6311319645131;
        self.params.d[36] = 0.613643610094145;
        self.params.d[37] = 0.231363049553804;
        self.params.d[38] = -0.553740947749688;
        self.params.d[39] = -1.09978198064067;
        self.params.d[40] = -0.373920334495006;
        self.params.d[41] = -0.124239005203324;
        self.params.d[42] = -0.923057686995755;
        self.params.d[43] = -0.83282890309827;
        self.params.d[44] = -0.169254402708088;
        self.params.d[45] = 1.44213565178771;
        self.params.d[46] = 0.345011617871286;
        self.params.d[47] = -0.866048550271161;
        self.params.d[48] = -0.888089973505595;
        self.params.d[49] = -0.181511697912213;
        self.params.d[50] = -1.17835862158005;
        self.params.d[51] = -1.19448515582771;
        self.params.d[52] = 0.0561402392697676;
        self.params.d[53] = -1.65108252487678;
        self.params.d[54] = -0.0656578705936539;
        self.params.d[55] = -0.551295150448667;
        self.params.d[56] = 0.830746487262684;
        self.params.d[57] = 0.986984892408018;
        self.params.d[58] = 0.764371687423057;
        self.params.d[59] = 0.756721655019656;
        self.params.d[60] = -0.505599503404287;
        self.params.d[61] = 0.67253921894107;
        self.params.d[62] = -0.640605344172728;
        self.params.d[63] = 0.2911754794755;
        self.params.d[64] = -0.696771367740502;
        self.params.d[65] = -0.219419802945872;
        self.params.d[66] = -1.75388427668024;
        self.params.d[67] = -1.02929831126265;
        self.params.d[68] = 1.88641042469427;
        self.params.d[69] = -1.0776631825797;
        self.params.d[70] = 0.765910043789321;
        self.params.d[71] = 0.601907432854958;
        self.params.d[72] = 0.895756557749928;
        self.params.d[73] = -0.0996455574622748;
        self.params.d[74] = 0.386655098407451;
        self.params.d[75] = -1.73212230426869;
        self.params.d[76] = -1.70975144871107;
        self.params.d[77] = -1.20409589481169;
        self.params.d[78] = -1.39255601196584;
        self.params.d[79] = -1.59958262167422;
        self.params.d[80] = -1.48282454156458;
        self.params.d[81] = 0.213110927230614;
        self.params.d[82] = -1.24874070030449;
        self.params.d[83] = 1.80840497212483;
        self.params.d[84] = 0.726447115229707;
        self.params.d[85] = 0.164078693439085;
        self.params.d[86] = 0.828722403231591;
        self.params.d[87] = -0.944453316189946;
        self.params.d[88] = 1.70690273701491;
        self.params.d[89] = 1.35677223119988;
        self.params.d[90] = 0.905277993712149;
        self.params.d[91] = -0.0790401756583599;
        self.params.d[92] = 1.36841274350659;
        self.params.d[93] = 0.979009293697437;
        self.params.d[94] = 0.64130362559845;
        self.params.d[95] = 1.65590106802375;
        self.params.Aeq[0] = 0.534662255150299;
        self.params.Aeq[1] = -0.536237660589562;
        self.params.Aeq[2] = 0.211378292601782;
        self.params.Aeq[3] = -1.21447769319945;
        self.params.Aeq[4] = -1.23171081442559;
        self.params.Aeq[5] = 0.902678495731283;
        self.params.Aeq[6] = 1.13974681372452;
        self.params.Aeq[7] = 1.88839345473506;
        self.params.Aeq[8] = 1.40388566816601;
        self.params.Aeq[9] = 0.174377306383291;
        self.params.Aeq[10] = -1.64083652190774;
        self.params.Aeq[11] = -0.0445070215355488;
        self.params.Aeq[12] = 1.7117453902485;
        self.params.Aeq[13] = 1.15047279801391;
        self.params.Aeq[14] = -0.0596230957836474;
        self.params.Aeq[15] = -0.178882554076455;
        self.params.Aeq[16] = -1.12805692636259;
        self.params.Aeq[17] = -1.29114647679271;
        self.params.Aeq[18] = -1.70550532312257;
        self.params.Aeq[19] = 1.56957275034837;
        self.params.Aeq[20] = 0.560706467596236;
        self.params.Aeq[21] = -1.42667073011471;
        self.params.Aeq[22] = -0.343492321135171;
        self.params.Aeq[23] = -1.80356430240851;
        self.params.Aeq[24] = -1.16250660191055;
        self.params.Aeq[25] = 0.922832496516153;
        self.params.Aeq[26] = 0.604491081766398;
        self.params.Aeq[27] = -0.0840868104920891;
        self.params.Aeq[28] = -0.900877978017443;
        self.params.Aeq[29] = 0.608892500264739;
        self.params.Aeq[30] = 1.82579804526952;
        self.params.Aeq[31] = -0.257917775299229;
        self.params.Aeq[32] = -1.71946997964932;
        self.params.Aeq[33] = -1.76907404870813;
        self.params.Aeq[34] = -1.66851592480977;
        self.params.Aeq[35] = 1.83882874901288;
        self.params.Aeq[36] = 0.163043344745975;
        self.params.Aeq[37] = 1.34984973067889;
        self.params.Aeq[38] = -1.31986582305146;
        self.params.Aeq[39] = -0.958619709084339;
        self.params.Aeq[40] = 0.767910047491371;
        self.params.Aeq[41] = 1.58228131256793;
        self.params.Aeq[42] = -0.637246062159362;
        self.params.Aeq[43] = -1.74130720803887;
        self.params.Aeq[44] = 1.45647867764258;
        self.params.Aeq[45] = -0.836510216682096;
        self.params.Aeq[46] = 0.96432962559825;
        self.params.Aeq[47] = -1.36786538119402;
        self.params.Aeq[48] = 0.779853740563504;
        self.params.Aeq[49] = 1.36567847612459;
        self.params.Aeq[50] = 0.908608314986837;
        self.params.Aeq[51] = -0.563569900546034;
        self.params.Aeq[52] = 0.906759005960792;
        self.params.Aeq[53] = -1.44213150327016;
        self.params.Aeq[54] = -0.744723539067112;
        self.params.Aeq[55] = -0.321668973268222;
        self.params.Aeq[56] = 1.50884815577727;
        self.params.Aeq[57] = -1.38503916571543;
        self.params.Aeq[58] = 1.52049916099726;
        self.params.Aeq[59] = 1.19585727688322;
        self.params.Aeq[60] = 1.88649718831192;
        self.params.Aeq[61] = -0.529188066786158;
        self.params.Aeq[62] = -1.18024092436888;
        self.params.Aeq[63] = -1.0377187186616;
        self.params.Aeq[64] = 1.31145120568568;
        self.params.Aeq[65] = 1.86091259437566;
        self.params.Aeq[66] = 0.795239993521694;
        self.params.Aeq[67] = -0.0700118329046804;
        self.params.Aeq[68] = -0.851800941275469;
        self.params.Aeq[69] = 1.33475153737264;
        self.params.Aeq[70] = 1.4887180335977;
        self.params.Aeq[71] = -1.63147363279763;
        self.params.Aeq[72] = -1.13620211592089;
        self.params.Aeq[73] = 1.32704436183147;
        self.params.Aeq[74] = 1.39321558831798;
        self.params.Aeq[75] = -0.741388004944011;
        self.params.Aeq[76] = -0.882821612612575;
        self.params.Aeq[77] = -0.27673991192616;
        self.params.Aeq[78] = 0.157786001058667;
        self.params.Aeq[79] = -1.61773273997355;
        self.params.Aeq[80] = 1.34764855485446;
        self.params.Aeq[81] = 0.138939481405284;
        self.params.Aeq[82] = 1.09987126016369;
        self.params.Aeq[83] = -1.07665493769469;
        self.params.Aeq[84] = 1.86117340442546;
        self.params.Aeq[85] = 1.00410922927352;
        self.params.Aeq[86] = -0.627624542432154;
        self.params.Aeq[87] = 1.79411058783982;
        self.params.Aeq[88] = 0.802047115865091;
        self.params.Aeq[89] = 1.36224434194495;
        self.params.Aeq[90] = -1.81801077657652;
        self.params.Aeq[91] = -1.77743383579325;
        self.params.Aeq[92] = 0.970949094198515;
        self.params.Aeq[93] = -0.781254268206432;
        self.params.Aeq[94] = 0.0671374633729811;
        self.params.Aeq[95] = -1.37495030531491;
        self.params.beq[0] = 1.91180963862794;
        self.params.lb[0] = 0.0110041906976779;
        self.params.lb[1] = 1.3160043138989;
        self.params.lb[2] = -1.70384881488001;
        self.params.lb[3] = -0.0843381911286474;
        self.params.lb[4] = -1.7508820783769;
        self.params.lb[5] = 1.53696572435095;
        self.params.lb[6] = -0.216759285148165;
        self.params.lb[7] = -1.72580032695265;
        self.params.lb[8] = -1.69401487073617;
        self.params.lb[9] = 0.15517063201268;
        self.params.lb[10] = -1.69773438197908;
        self.params.lb[11] = -1.26491072795023;
        self.params.lb[12] = -0.254571663333944;
        self.params.lb[13] = -0.00886867592617024;
        self.params.lb[14] = 0.33324766096703;
        self.params.lb[15] = 0.482050725619629;
        self.params.lb[16] = -0.508754001429326;
        self.params.lb[17] = 0.474946331922319;
        self.params.lb[18] = -1.37102136645946;
        self.params.lb[19] = -0.897966098265226;
        self.params.lb[20] = 1.19487308238524;
        self.params.lb[21] = -1.38764279709394;
        self.params.lb[22] = -1.10670810845705;
        self.params.lb[23] = -1.02808728122418;
        self.params.lb[24] = -0.0819707807077323;
        self.params.lb[25] = -1.99701791183241;
        self.params.lb[26] = -1.87875455791013;
        self.params.lb[27] = -0.153807393408778;
        self.params.lb[28] = -1.34991726053392;
        self.params.lb[29] = 0.718007215093141;
        self.params.lb[30] = 1.18081834870655;
        self.params.lb[31] = 0.312653434950841;
        self.params.lb[32] = 0.779059908692823;
        self.params.lb[33] = -0.436167937064485;
        self.params.lb[34] = -1.81481518802821;
        self.params.lb[35] = -0.242313869481403;
        self.params.lb[36] = -0.512078751162241;
        self.params.lb[37] = 0.38801296880132;
        self.params.lb[38] = -1.46312732120387;
        self.params.lb[39] = -1.08914841311266;
        self.params.lb[40] = 1.25912966610912;
        self.params.lb[41] = -0.942697893439147;
        self.params.lb[42] = -0.358719180371347;
        self.params.lb[43] = 1.74388870598313;
        self.params.lb[44] = -0.897790147916582;
        self.params.lb[45] = -1.41884016458574;
        self.params.lb[46] = 0.808080517325809;
        self.params.lb[47] = 0.268266201765099;
        self.params.lb[48] = 0.446375342186388;
        self.params.lb[49] = -1.83187659602571;
        self.params.lb[50] = -0.330932420971093;
        self.params.lb[51] = -1.98293426333136;
        self.params.lb[52] = -1.01385812455644;
        self.params.lb[53] = 0.824224734336025;
        self.params.lb[54] = -1.7538371363172;
        self.params.lb[55] = -0.821226005586881;
        self.params.lb[56] = 1.95245101124871;
        self.params.lb[57] = 1.8848889209079;
        self.params.lb[58] = -0.0726144452811801;
        self.params.lb[59] = 0.942773546112984;
        self.params.lb[60] = 0.530623096744556;
        self.params.lb[61] = -0.137227714225053;
        self.params.lb[62] = 1.42826573056528;
        self.params.lb[63] = -1.30992699133528;
        self.params.lb[64] = 1.31372768897644;
        self.params.lb[65] = -1.83172190616673;
        self.params.lb[66] = 1.46781476725119;
        self.params.lb[67] = 0.703986349872991;
        self.params.lb[68] = -0.216343560356526;
        self.params.lb[69] = 0.686280990537108;
        self.params.lb[70] = -0.158525984443032;
        self.params.lb[71] = 1.12001288951434;
        self.params.lb[72] = -1.54622366454353;
        self.params.lb[73] = 0.0326297153944215;
        self.params.lb[74] = 1.48595815977549;
        self.params.lb[75] = 1.71011710324809;
        self.params.lb[76] = -1.11865467380675;
        self.params.lb[77] = -0.992278789781524;
        self.params.lb[78] = 1.61604988643595;
        self.params.lb[79] = -0.617930645139486;
        self.params.lb[80] = -1.77250970380514;
        self.params.lb[81] = 0.859546688448131;
        self.params.lb[82] = -0.342324563386569;
        self.params.lb[83] = 0.941296749980576;
        self.params.lb[84] = -0.0916334662265226;
        self.params.lb[85] = 0.00226221774572766;
        self.params.lb[86] = -0.329752358365642;
        self.params.lb[87] = -0.838060415859394;
        self.params.lb[88] = 1.6028434695494;
        self.params.lb[89] = 0.675150311940429;
        self.params.lb[90] = 1.15532937337187;
        self.params.lb[91] = 1.58295812437247;
        self.params.lb[92] = -0.99924423044256;
        self.params.lb[93] = 1.67928245588969;
        self.params.lb[94] = 1.45042034903423;
        self.params.lb[95] = 0.0243410484999456;
        self.params.ub[0] = 0.271608696576123;
        self.params.ub[1] = -1.54027104785289;
        self.params.ub[2] = 1.04846336223107;
        self.params.ub[3] = -1.30709997126271;
        self.params.ub[4] = 0.135344164023638;
        self.params.ub[5] = -1.49425077908512;
        self.params.ub[6] = -1.70833162567137;
        self.params.ub[7] = 0.436109775042258;
        self.params.ub[8] = -0.0351874815372799;
        self.params.ub[9] = 0.699239738957091;
        self.params.ub[10] = 1.16341673221714;
        self.params.ub[11] = 1.93074997058226;
        self.params.ub[12] = -1.66367727569327;
        self.params.ub[13] = 0.524848449734322;
        self.params.ub[14] = 0.307899581525791;
        self.params.ub[15] = 0.602568707166812;
        self.params.ub[16] = 0.172717819257519;
        self.params.ub[17] = 0.229469550120807;
        self.params.ub[18] = 1.47421853456195;
        self.params.ub[19] = -0.191953534513699;
        self.params.ub[20] = 0.139902314521446;
        self.params.ub[21] = 0.76385481506106;
        self.params.ub[22] = -1.64202003441956;
        self.params.ub[23] = -0.272298724450761;
        self.params.ub[24] = -1.59146311718205;
        self.params.ub[25] = -1.44876042835587;
        self.params.ub[26] = -1.99149776613636;
        self.params.ub[27] = -1.16117425535352;
        self.params.ub[28] = -1.13345095024706;
        self.params.ub[29] = 0.0649779249377715;
        self.params.ub[30] = 0.280832953960973;
        self.params.ub[31] = 1.29584472201299;
        self.params.ub[32] = -0.0531552447073715;
        self.params.ub[33] = 1.56581839568717;
        self.params.ub[34] = -0.419756840899337;
        self.params.ub[35] = 0.97844578833777;
        self.params.ub[36] = 0.211029049669529;
        self.params.ub[37] = 0.495300343089304;
        self.params.ub[38] = -0.91843201246675;
        self.params.ub[39] = 1.75038003175916;
        self.params.ub[40] = 1.07861886143159;
        self.params.ub[41] = -1.41761988372037;
        self.params.ub[42] = 0.149737479778294;
        self.params.ub[43] = 1.98314522222234;
        self.params.ub[44] = -1.80377466997947;
        self.params.ub[45] = -0.788720648329546;
        self.params.ub[46] = 0.963253485408665;
        self.params.ub[47] = -1.84255420938954;
        self.params.ub[48] = 0.986684363969033;
        self.params.ub[49] = 0.293685119935044;
        self.params.ub[50] = 0.926822702248266;
        self.params.ub[51] = 0.203330383506533;
        self.params.ub[52] = 1.75761391320464;
        self.params.ub[53] = -0.614393188398918;
        self.params.ub[54] = 0.297877839744912;
        self.params.ub[55] = -1.79688008399089;
        self.params.ub[56] = 0.213731336617427;
        self.params.ub[57] = -0.322428225408252;
        self.params.ub[58] = 1.93264715116081;
        self.params.ub[59] = 1.78242927534818;
        self.params.ub[60] = -1.4468823405676;
        self.params.ub[61] = -1.83353743387615;
        self.params.ub[62] = -1.51729973172437;
        self.params.ub[63] = -1.22901212912072;
        self.params.ub[64] = 0.904671977242209;
        self.params.ub[65] = 0.175911814154894;
        self.params.ub[66] = 0.139701338141126;
        self.params.ub[67] = -0.141852082149852;
        self.params.ub[68] = -1.97322312647393;
        self.params.ub[69] = -0.430112345822133;
        self.params.ub[70] = 1.99575376503877;
        self.params.ub[71] = 1.28116482164779;
        self.params.ub[72] = 0.291442843758822;
        self.params.ub[73] = -1.21414815721888;
        self.params.ub[74] = 1.68187769803742;
        self.params.ub[75] = -0.303411010382146;
        self.params.ub[76] = 0.477309092317931;
        self.params.ub[77] = -1.1875693730353;
        self.params.ub[78] = -0.687737024791553;
        self.params.ub[79] = -0.620186148261617;
        self.params.ub[80] = -0.420992518392157;
        self.params.ub[81] = -1.91107245377125;
        self.params.ub[82] = 0.641388208780794;
        self.params.ub[83] = -1.3200399280087;
        self.params.ub[84] = 0.413201053013126;
        self.params.ub[85] = 0.478321386139227;
        self.params.ub[86] = 0.791618985729374;
        self.params.ub[87] = -0.832275255814656;
        self.params.ub[88] = -0.831872053742615;
        self.params.ub[89] = 1.02211790761134;
        self.params.ub[90] = -0.447103218926263;
        self.params.ub[91] = -1.3901469561677;
        self.params.ub[92] = 1.62105960512086;
        self.params.ub[93] = -1.94766876019127;
        self.params.ub[94] = 1.54593763062313;
        self.params.ub[95] = -0.830972896191656;

        
  
