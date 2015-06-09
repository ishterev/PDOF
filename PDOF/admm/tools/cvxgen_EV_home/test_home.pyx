# -*- coding: utf-8 -*-
"""
Created on Fri May 29 10:56:11 2015

@author: shterev
"""

#cimport csolve_home as cs
from csolve_home cimport *
cimport csolve_home
from admm.tools.params cimport *

from cython cimport view
cimport numpy as np
import numpy as np

cdef class TestSolver:
    
    
        
    cpdef test_ext(self):
        self.test()
    
    
    cdef void test(self):  
        set_defaults()  # Set basic algorithm parameters.
        setup_indexing()

        print "Loading default data"    
    
        self.load_default_data()
        #csolve_home.params.d[0] = 0.203191610298302;
        
        #cdef unsigned int i
        #for i in range(96):
            #print csolve_home.params.d[i]
        
        print "Solving" 
        csolve_home.settings.verbose = 1
        cdef unsigned int num_iters = solve()
        
        print ("Solved after  %1u iterations", (num_iters,))
        
        print "Result: "
        
        
        
        self.printResult()
        
       

        
        #csolve_home.solve()

        # Solve our problem at high speed!
        #cdef int num_iters = cs.solve()
        # Recommended: check work.converged == 1.
        #printResult()
        
    
       
         
    cpdef printResult(self):
        #load_default_data()
        #return np.asarray(* csolve_home.params.d)
        cdef unsigned int i
        for i in range(96):
            print  csolve_home.vars.x[i]
        


    cdef void load_default_data(self): 
        
        csolve_home.params.d[0] = 0.203191610298302;        
        csolve_home.params.d[1] = 0.832591290472419;
        csolve_home.params.d[2] = -0.836381044348223;
        csolve_home.params.d[3] = 0.0433104207906521;
        csolve_home.params.d[4] = 1.57178781739062;
        csolve_home.params.d[5] = 1.58517235573375;
        csolve_home.params.d[6] = -1.49765875814465;
        csolve_home.params.d[7] = -1.17102848744725;
        csolve_home.params.d[8] = -1.79413118679668;
        csolve_home.params.d[9] = -0.236760625397454;
        csolve_home.params.d[10] = -1.88049515648573;
        csolve_home.params.d[11] = -0.172667102421156;
        csolve_home.params.d[12] = 0.596576190459043;
        csolve_home.params.d[13] = -0.886050869408099;
        csolve_home.params.d[14] = 0.705019607920525;
        csolve_home.params.d[15] = 0.363451269665403;
        csolve_home.params.d[16] = -1.90407247049134;
        csolve_home.params.d[17] = 0.235416351963528;
        csolve_home.params.d[18] = -0.962990212370138;
        csolve_home.params.d[19] = -0.339595211959721;
        csolve_home.params.d[20] = -0.865899672914725;
        csolve_home.params.d[21] = 0.772551673251985;
        csolve_home.params.d[22] = -0.238185129317042;
        csolve_home.params.d[23] = -1.37252904610015;
        csolve_home.params.d[24] = 0.178596072127379;
        csolve_home.params.d[25] = 1.12125905804547;
        csolve_home.params.d[26] = -0.774545870495281;
        csolve_home.params.d[27] = -1.11216846427127;
        csolve_home.params.d[28] = -0.448114969777405;
        csolve_home.params.d[29] = 1.74553459944172;
        csolve_home.params.d[30] = 1.90398168989174;
        csolve_home.params.d[31] = 0.689534703651255;
        csolve_home.params.d[32] = 1.61133643415359;
        csolve_home.params.d[33] = 1.38300348517272;
        csolve_home.params.d[34] = -0.488023834684443;
        csolve_home.params.d[35] = -1.6311319645131;
        csolve_home.params.d[36] = 0.613643610094145;
        csolve_home.params.d[37] = 0.231363049553804;
        csolve_home.params.d[38] = -0.553740947749688;
        csolve_home.params.d[39] = -1.09978198064067;
        csolve_home.params.d[40] = -0.373920334495006;
        csolve_home.params.d[41] = -0.124239005203324;
        csolve_home.params.d[42] = -0.923057686995755;
        csolve_home.params.d[43] = -0.83282890309827;
        csolve_home.params.d[44] = -0.169254402708088;
        csolve_home.params.d[45] = 1.44213565178771;
        csolve_home.params.d[46] = 0.345011617871286;
        csolve_home.params.d[47] = -0.866048550271161;
        csolve_home.params.d[48] = -0.888089973505595;
        csolve_home.params.d[49] = -0.181511697912213;
        csolve_home.params.d[50] = -1.17835862158005;
        csolve_home.params.d[51] = -1.19448515582771;
        csolve_home.params.d[52] = 0.0561402392697676;
        csolve_home.params.d[53] = -1.65108252487678;
        csolve_home.params.d[54] = -0.0656578705936539;
        csolve_home.params.d[55] = -0.551295150448667;
        csolve_home.params.d[56] = 0.830746487262684;
        csolve_home.params.d[57] = 0.986984892408018;
        csolve_home.params.d[58] = 0.764371687423057;
        csolve_home.params.d[59] = 0.756721655019656;
        csolve_home.params.d[60] = -0.505599503404287;
        csolve_home.params.d[61] = 0.67253921894107;
        csolve_home.params.d[62] = -0.640605344172728;
        csolve_home.params.d[63] = 0.2911754794755;
        csolve_home.params.d[64] = -0.696771367740502;
        csolve_home.params.d[65] = -0.219419802945872;
        csolve_home.params.d[66] = -1.75388427668024;
        csolve_home.params.d[67] = -1.02929831126265;
        csolve_home.params.d[68] = 1.88641042469427;
        csolve_home.params.d[69] = -1.0776631825797;
        csolve_home.params.d[70] = 0.765910043789321;
        csolve_home.params.d[71] = 0.601907432854958;
        csolve_home.params.d[72] = 0.895756557749928;
        csolve_home.params.d[73] = -0.0996455574622748;
        csolve_home.params.d[74] = 0.386655098407451;
        csolve_home.params.d[75] = -1.73212230426869;
        csolve_home.params.d[76] = -1.70975144871107;
        csolve_home.params.d[77] = -1.20409589481169;
        csolve_home.params.d[78] = -1.39255601196584;
        csolve_home.params.d[79] = -1.59958262167422;
        csolve_home.params.d[80] = -1.48282454156458;
        csolve_home.params.d[81] = 0.213110927230614;
        csolve_home.params.d[82] = -1.24874070030449;
        csolve_home.params.d[83] = 1.80840497212483;
        csolve_home.params.d[84] = 0.726447115229707;
        csolve_home.params.d[85] = 0.164078693439085;
        csolve_home.params.d[86] = 0.828722403231591;
        csolve_home.params.d[87] = -0.944453316189946;
        csolve_home.params.d[88] = 1.70690273701491;
        csolve_home.params.d[89] = 1.35677223119988;
        csolve_home.params.d[90] = 0.905277993712149;
        csolve_home.params.d[91] = -0.0790401756583599;
        csolve_home.params.d[92] = 1.36841274350659;
        csolve_home.params.d[93] = 0.979009293697437;
        csolve_home.params.d[94] = 0.64130362559845;
        csolve_home.params.d[95] = 1.65590106802375;
        csolve_home.params.Aeq[0] = 0.534662255150299;
        csolve_home.params.Aeq[1] = -0.536237660589562;
        csolve_home.params.Aeq[2] = 0.211378292601782;
        csolve_home.params.Aeq[3] = -1.21447769319945;
        csolve_home.params.Aeq[4] = -1.23171081442559;
        csolve_home.params.Aeq[5] = 0.902678495731283;
        csolve_home.params.Aeq[6] = 1.13974681372452;
        csolve_home.params.Aeq[7] = 1.88839345473506;
        csolve_home.params.Aeq[8] = 1.40388566816601;
        csolve_home.params.Aeq[9] = 0.174377306383291;
        csolve_home.params.Aeq[10] = -1.64083652190774;
        csolve_home.params.Aeq[11] = -0.0445070215355488;
        csolve_home.params.Aeq[12] = 1.7117453902485;
        csolve_home.params.Aeq[13] = 1.15047279801391;
        csolve_home.params.Aeq[14] = -0.0596230957836474;
        csolve_home.params.Aeq[15] = -0.178882554076455;
        csolve_home.params.Aeq[16] = -1.12805692636259;
        csolve_home.params.Aeq[17] = -1.29114647679271;
        csolve_home.params.Aeq[18] = -1.70550532312257;
        csolve_home.params.Aeq[19] = 1.56957275034837;
        csolve_home.params.Aeq[20] = 0.560706467596236;
        csolve_home.params.Aeq[21] = -1.42667073011471;
        csolve_home.params.Aeq[22] = -0.343492321135171;
        csolve_home.params.Aeq[23] = -1.80356430240851;
        csolve_home.params.Aeq[24] = -1.16250660191055;
        csolve_home.params.Aeq[25] = 0.922832496516153;
        csolve_home.params.Aeq[26] = 0.604491081766398;
        csolve_home.params.Aeq[27] = -0.0840868104920891;
        csolve_home.params.Aeq[28] = -0.900877978017443;
        csolve_home.params.Aeq[29] = 0.608892500264739;
        csolve_home.params.Aeq[30] = 1.82579804526952;
        csolve_home.params.Aeq[31] = -0.257917775299229;
        csolve_home.params.Aeq[32] = -1.71946997964932;
        csolve_home.params.Aeq[33] = -1.76907404870813;
        csolve_home.params.Aeq[34] = -1.66851592480977;
        csolve_home.params.Aeq[35] = 1.83882874901288;
        csolve_home.params.Aeq[36] = 0.163043344745975;
        csolve_home.params.Aeq[37] = 1.34984973067889;
        csolve_home.params.Aeq[38] = -1.31986582305146;
        csolve_home.params.Aeq[39] = -0.958619709084339;
        csolve_home.params.Aeq[40] = 0.767910047491371;
        csolve_home.params.Aeq[41] = 1.58228131256793;
        csolve_home.params.Aeq[42] = -0.637246062159362;
        csolve_home.params.Aeq[43] = -1.74130720803887;
        csolve_home.params.Aeq[44] = 1.45647867764258;
        csolve_home.params.Aeq[45] = -0.836510216682096;
        csolve_home.params.Aeq[46] = 0.96432962559825;
        csolve_home.params.Aeq[47] = -1.36786538119402;
        csolve_home.params.Aeq[48] = 0.779853740563504;
        csolve_home.params.Aeq[49] = 1.36567847612459;
        csolve_home.params.Aeq[50] = 0.908608314986837;
        csolve_home.params.Aeq[51] = -0.563569900546034;
        csolve_home.params.Aeq[52] = 0.906759005960792;
        csolve_home.params.Aeq[53] = -1.44213150327016;
        csolve_home.params.Aeq[54] = -0.744723539067112;
        csolve_home.params.Aeq[55] = -0.321668973268222;
        csolve_home.params.Aeq[56] = 1.50884815577727;
        csolve_home.params.Aeq[57] = -1.38503916571543;
        csolve_home.params.Aeq[58] = 1.52049916099726;
        csolve_home.params.Aeq[59] = 1.19585727688322;
        csolve_home.params.Aeq[60] = 1.88649718831192;
        csolve_home.params.Aeq[61] = -0.529188066786158;
        csolve_home.params.Aeq[62] = -1.18024092436888;
        csolve_home.params.Aeq[63] = -1.0377187186616;
        csolve_home.params.Aeq[64] = 1.31145120568568;
        csolve_home.params.Aeq[65] = 1.86091259437566;
        csolve_home.params.Aeq[66] = 0.795239993521694;
        csolve_home.params.Aeq[67] = -0.0700118329046804;
        csolve_home.params.Aeq[68] = -0.851800941275469;
        csolve_home.params.Aeq[69] = 1.33475153737264;
        csolve_home.params.Aeq[70] = 1.4887180335977;
        csolve_home.params.Aeq[71] = -1.63147363279763;
        csolve_home.params.Aeq[72] = -1.13620211592089;
        csolve_home.params.Aeq[73] = 1.32704436183147;
        csolve_home.params.Aeq[74] = 1.39321558831798;
        csolve_home.params.Aeq[75] = -0.741388004944011;
        csolve_home.params.Aeq[76] = -0.882821612612575;
        csolve_home.params.Aeq[77] = -0.27673991192616;
        csolve_home.params.Aeq[78] = 0.157786001058667;
        csolve_home.params.Aeq[79] = -1.61773273997355;
        csolve_home.params.Aeq[80] = 1.34764855485446;
        csolve_home.params.Aeq[81] = 0.138939481405284;
        csolve_home.params.Aeq[82] = 1.09987126016369;
        csolve_home.params.Aeq[83] = -1.07665493769469;
        csolve_home.params.Aeq[84] = 1.86117340442546;
        csolve_home.params.Aeq[85] = 1.00410922927352;
        csolve_home.params.Aeq[86] = -0.627624542432154;
        csolve_home.params.Aeq[87] = 1.79411058783982;
        csolve_home.params.Aeq[88] = 0.802047115865091;
        csolve_home.params.Aeq[89] = 1.36224434194495;
        csolve_home.params.Aeq[90] = -1.81801077657652;
        csolve_home.params.Aeq[91] = -1.77743383579325;
        csolve_home.params.Aeq[92] = 0.970949094198515;
        csolve_home.params.Aeq[93] = -0.781254268206432;
        csolve_home.params.Aeq[94] = 0.0671374633729811;
        csolve_home.params.Aeq[95] = -1.37495030531491;
        csolve_home.params.beq[0] = 1.91180963862794;
        csolve_home.params.lb[0] = 0.0110041906976779;
        csolve_home.params.lb[1] = 1.3160043138989;
        csolve_home.params.lb[2] = -1.70384881488001;
        csolve_home.params.lb[3] = -0.0843381911286474;
        csolve_home.params.lb[4] = -1.7508820783769;
        csolve_home.params.lb[5] = 1.53696572435095;
        csolve_home.params.lb[6] = -0.216759285148165;
        csolve_home.params.lb[7] = -1.72580032695265;
        csolve_home.params.lb[8] = -1.69401487073617;
        csolve_home.params.lb[9] = 0.15517063201268;
        csolve_home.params.lb[10] = -1.69773438197908;
        csolve_home.params.lb[11] = -1.26491072795023;
        csolve_home.params.lb[12] = -0.254571663333944;
        csolve_home.params.lb[13] = -0.00886867592617024;
        csolve_home.params.lb[14] = 0.33324766096703;
        csolve_home.params.lb[15] = 0.482050725619629;
        csolve_home.params.lb[16] = -0.508754001429326;
        csolve_home.params.lb[17] = 0.474946331922319;
        csolve_home.params.lb[18] = -1.37102136645946;
        csolve_home.params.lb[19] = -0.897966098265226;
        csolve_home.params.lb[20] = 1.19487308238524;
        csolve_home.params.lb[21] = -1.38764279709394;
        csolve_home.params.lb[22] = -1.10670810845705;
        csolve_home.params.lb[23] = -1.02808728122418;
        csolve_home.params.lb[24] = -0.0819707807077323;
        csolve_home.params.lb[25] = -1.99701791183241;
        csolve_home.params.lb[26] = -1.87875455791013;
        csolve_home.params.lb[27] = -0.153807393408778;
        csolve_home.params.lb[28] = -1.34991726053392;
        csolve_home.params.lb[29] = 0.718007215093141;
        csolve_home.params.lb[30] = 1.18081834870655;
        csolve_home.params.lb[31] = 0.312653434950841;
        csolve_home.params.lb[32] = 0.779059908692823;
        csolve_home.params.lb[33] = -0.436167937064485;
        csolve_home.params.lb[34] = -1.81481518802821;
        csolve_home.params.lb[35] = -0.242313869481403;
        csolve_home.params.lb[36] = -0.512078751162241;
        csolve_home.params.lb[37] = 0.38801296880132;
        csolve_home.params.lb[38] = -1.46312732120387;
        csolve_home.params.lb[39] = -1.08914841311266;
        csolve_home.params.lb[40] = 1.25912966610912;
        csolve_home.params.lb[41] = -0.942697893439147;
        csolve_home.params.lb[42] = -0.358719180371347;
        csolve_home.params.lb[43] = 1.74388870598313;
        csolve_home.params.lb[44] = -0.897790147916582;
        csolve_home.params.lb[45] = -1.41884016458574;
        csolve_home.params.lb[46] = 0.808080517325809;
        csolve_home.params.lb[47] = 0.268266201765099;
        csolve_home.params.lb[48] = 0.446375342186388;
        csolve_home.params.lb[49] = -1.83187659602571;
        csolve_home.params.lb[50] = -0.330932420971093;
        csolve_home.params.lb[51] = -1.98293426333136;
        csolve_home.params.lb[52] = -1.01385812455644;
        csolve_home.params.lb[53] = 0.824224734336025;
        csolve_home.params.lb[54] = -1.7538371363172;
        csolve_home.params.lb[55] = -0.821226005586881;
        csolve_home.params.lb[56] = 1.95245101124871;
        csolve_home.params.lb[57] = 1.8848889209079;
        csolve_home.params.lb[58] = -0.0726144452811801;
        csolve_home.params.lb[59] = 0.942773546112984;
        csolve_home.params.lb[60] = 0.530623096744556;
        csolve_home.params.lb[61] = -0.137227714225053;
        csolve_home.params.lb[62] = 1.42826573056528;
        csolve_home.params.lb[63] = -1.30992699133528;
        csolve_home.params.lb[64] = 1.31372768897644;
        csolve_home.params.lb[65] = -1.83172190616673;
        csolve_home.params.lb[66] = 1.46781476725119;
        csolve_home.params.lb[67] = 0.703986349872991;
        csolve_home.params.lb[68] = -0.216343560356526;
        csolve_home.params.lb[69] = 0.686280990537108;
        csolve_home.params.lb[70] = -0.158525984443032;
        csolve_home.params.lb[71] = 1.12001288951434;
        csolve_home.params.lb[72] = -1.54622366454353;
        csolve_home.params.lb[73] = 0.0326297153944215;
        csolve_home.params.lb[74] = 1.48595815977549;
        csolve_home.params.lb[75] = 1.71011710324809;
        csolve_home.params.lb[76] = -1.11865467380675;
        csolve_home.params.lb[77] = -0.992278789781524;
        csolve_home.params.lb[78] = 1.61604988643595;
        csolve_home.params.lb[79] = -0.617930645139486;
        csolve_home.params.lb[80] = -1.77250970380514;
        csolve_home.params.lb[81] = 0.859546688448131;
        csolve_home.params.lb[82] = -0.342324563386569;
        csolve_home.params.lb[83] = 0.941296749980576;
        csolve_home.params.lb[84] = -0.0916334662265226;
        csolve_home.params.lb[85] = 0.00226221774572766;
        csolve_home.params.lb[86] = -0.329752358365642;
        csolve_home.params.lb[87] = -0.838060415859394;
        csolve_home.params.lb[88] = 1.6028434695494;
        csolve_home.params.lb[89] = 0.675150311940429;
        csolve_home.params.lb[90] = 1.15532937337187;
        csolve_home.params.lb[91] = 1.58295812437247;
        csolve_home.params.lb[92] = -0.99924423044256;
        csolve_home.params.lb[93] = 1.67928245588969;
        csolve_home.params.lb[94] = 1.45042034903423;
        csolve_home.params.lb[95] = 0.0243410484999456;
        csolve_home.params.ub[0] = 0.271608696576123;
        csolve_home.params.ub[1] = -1.54027104785289;
        csolve_home.params.ub[2] = 1.04846336223107;
        csolve_home.params.ub[3] = -1.30709997126271;
        csolve_home.params.ub[4] = 0.135344164023638;
        csolve_home.params.ub[5] = -1.49425077908512;
        csolve_home.params.ub[6] = -1.70833162567137;
        csolve_home.params.ub[7] = 0.436109775042258;
        csolve_home.params.ub[8] = -0.0351874815372799;
        csolve_home.params.ub[9] = 0.699239738957091;
        csolve_home.params.ub[10] = 1.16341673221714;
        csolve_home.params.ub[11] = 1.93074997058226;
        csolve_home.params.ub[12] = -1.66367727569327;
        csolve_home.params.ub[13] = 0.524848449734322;
        csolve_home.params.ub[14] = 0.307899581525791;
        csolve_home.params.ub[15] = 0.602568707166812;
        csolve_home.params.ub[16] = 0.172717819257519;
        csolve_home.params.ub[17] = 0.229469550120807;
        csolve_home.params.ub[18] = 1.47421853456195;
        csolve_home.params.ub[19] = -0.191953534513699;
        csolve_home.params.ub[20] = 0.139902314521446;
        csolve_home.params.ub[21] = 0.76385481506106;
        csolve_home.params.ub[22] = -1.64202003441956;
        csolve_home.params.ub[23] = -0.272298724450761;
        csolve_home.params.ub[24] = -1.59146311718205;
        csolve_home.params.ub[25] = -1.44876042835587;
        csolve_home.params.ub[26] = -1.99149776613636;
        csolve_home.params.ub[27] = -1.16117425535352;
        csolve_home.params.ub[28] = -1.13345095024706;
        csolve_home.params.ub[29] = 0.0649779249377715;
        csolve_home.params.ub[30] = 0.280832953960973;
        csolve_home.params.ub[31] = 1.29584472201299;
        csolve_home.params.ub[32] = -0.0531552447073715;
        csolve_home.params.ub[33] = 1.56581839568717;
        csolve_home.params.ub[34] = -0.419756840899337;
        csolve_home.params.ub[35] = 0.97844578833777;
        csolve_home.params.ub[36] = 0.211029049669529;
        csolve_home.params.ub[37] = 0.495300343089304;
        csolve_home.params.ub[38] = -0.91843201246675;
        csolve_home.params.ub[39] = 1.75038003175916;
        csolve_home.params.ub[40] = 1.07861886143159;
        csolve_home.params.ub[41] = -1.41761988372037;
        csolve_home.params.ub[42] = 0.149737479778294;
        csolve_home.params.ub[43] = 1.98314522222234;
        csolve_home.params.ub[44] = -1.80377466997947;
        csolve_home.params.ub[45] = -0.788720648329546;
        csolve_home.params.ub[46] = 0.963253485408665;
        csolve_home.params.ub[47] = -1.84255420938954;
        csolve_home.params.ub[48] = 0.986684363969033;
        csolve_home.params.ub[49] = 0.293685119935044;
        csolve_home.params.ub[50] = 0.926822702248266;
        csolve_home.params.ub[51] = 0.203330383506533;
        csolve_home.params.ub[52] = 1.75761391320464;
        csolve_home.params.ub[53] = -0.614393188398918;
        csolve_home.params.ub[54] = 0.297877839744912;
        csolve_home.params.ub[55] = -1.79688008399089;
        csolve_home.params.ub[56] = 0.213731336617427;
        csolve_home.params.ub[57] = -0.322428225408252;
        csolve_home.params.ub[58] = 1.93264715116081;
        csolve_home.params.ub[59] = 1.78242927534818;
        csolve_home.params.ub[60] = -1.4468823405676;
        csolve_home.params.ub[61] = -1.83353743387615;
        csolve_home.params.ub[62] = -1.51729973172437;
        csolve_home.params.ub[63] = -1.22901212912072;
        csolve_home.params.ub[64] = 0.904671977242209;
        csolve_home.params.ub[65] = 0.175911814154894;
        csolve_home.params.ub[66] = 0.139701338141126;
        csolve_home.params.ub[67] = -0.141852082149852;
        csolve_home.params.ub[68] = -1.97322312647393;
        csolve_home.params.ub[69] = -0.430112345822133;
        csolve_home.params.ub[70] = 1.99575376503877;
        csolve_home.params.ub[71] = 1.28116482164779;
        csolve_home.params.ub[72] = 0.291442843758822;
        csolve_home.params.ub[73] = -1.21414815721888;
        csolve_home.params.ub[74] = 1.68187769803742;
        csolve_home.params.ub[75] = -0.303411010382146;
        csolve_home.params.ub[76] = 0.477309092317931;
        csolve_home.params.ub[77] = -1.1875693730353;
        csolve_home.params.ub[78] = -0.687737024791553;
        csolve_home.params.ub[79] = -0.620186148261617;
        csolve_home.params.ub[80] = -0.420992518392157;
        csolve_home.params.ub[81] = -1.91107245377125;
        csolve_home.params.ub[82] = 0.641388208780794;
        csolve_home.params.ub[83] = -1.3200399280087;
        csolve_home.params.ub[84] = 0.413201053013126;
        csolve_home.params.ub[85] = 0.478321386139227;
        csolve_home.params.ub[86] = 0.791618985729374;
        csolve_home.params.ub[87] = -0.832275255814656;
        csolve_home.params.ub[88] = -0.831872053742615;
        csolve_home.params.ub[89] = 1.02211790761134;
        csolve_home.params.ub[90] = -0.447103218926263;
        csolve_home.params.ub[91] = -1.3901469561677;
        csolve_home.params.ub[92] = 1.62105960512086;
        csolve_home.params.ub[93] = -1.94766876019127;
        csolve_home.params.ub[94] = 1.54593763062313;
        csolve_home.params.ub[95] = -0.830972896191656;

        
  
