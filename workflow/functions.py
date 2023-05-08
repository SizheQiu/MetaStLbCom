import cobra
import pandas as pd
import numpy as np
from math import exp
from itertools import chain
from typing import TYPE_CHECKING, Callable, Dict, List, Optional, Union
from warnings import warn
from optlang.symbolics import Zero
from cobra.core.solution import get_solution
from cobra.util import solver as sutil
from pHcalc.pHcalc import Acid, Neutral, System

    
def LacIn( v_max,v_min, lac_con, klach, C1:float=-0.0142, C2:float=6.1232 ):
    '''Model inhibition of undissociated lactic acid.
    :param vmax: maximum reaction rate(mmol/hr*gDW)
    :param vmin: minimum reaction rate(mmol/hr*gDW)
    :param lac_con: concentration of total lactic acid
    :param klach: inhibition coefficient
    :param C1, C2: parameters used in linear approximation of pH
    :return reaction rate(mmol/hr*gDW)
    '''
    pH = C1*lac_con + C2
    lach = lac_con/(10**(pH-3.86))
    temp_value = abs(v_max)*exp( -klach*lach )
    if v_max > 0:
        out = max(temp_value, abs(v_min) )
    else:
        out = -1*max( temp_value, abs(v_min) )
    return out


def update_MM( vmax, s, Km ):
    '''Constrain exchange fluxes with michealis-menten equation.
    :param vmax: maximum reaction rate(mmol/hr*gDW)
    :param s: substrate concentration(mM)
    :param Km: michealis-menten constant(mM)
    :return flux upper bound
    '''
    ub = (vmax * s)/(Km+s)
    return ub


def get_pa_params( path, species ):
    '''Get parameters used in proteome allocation model for ST and LB.
    '''
    pa_params = pd.read_csv(path)
    if species == 'st':
        pa_params = (pa_params[ pa_params['species']=='ST']).reset_index().drop(['index'],axis=1)
    elif species == 'lb':
        pa_params = (pa_params[ pa_params['species']=='LB']).reset_index().drop(['index'],axis=1)
        
    Km_param = (pa_params[ pa_params['param']=='Km']).reset_index().drop(['index'],axis=1)
    Km_dict = {list(Km_param['rxn'])[i]:list(Km_param['value'])[i]*list(Km_param['direction'])[i] for i in range(len(Km_param.index))}
    klach_param = (pa_params[ pa_params['param']=='klach']).reset_index().drop(['index'],axis=1)
    klach_dict = {list(klach_param['rxn'])[i]:list(klach_param['value'])[i]*list(klach_param['direction'])[i] 
                  for i in range(len(klach_param.index))}
    vmax_param = (pa_params[ pa_params['param']=='Vmax']).reset_index().drop(['index'],axis=1)
    vmax_dict = {list(vmax_param['rxn'])[i]:list(vmax_param['value'])[i]*list(vmax_param['direction'])[i] 
                 for i in range(len(vmax_param.index))}
    vmin_param = (pa_params[ pa_params['param']=='Vmin']).reset_index().drop(['index'],axis=1)
    vmin_dict = {list(vmin_param['rxn'])[i]:list(vmin_param['value'])[i]*list(vmin_param['direction'])[i] 
                 for i in range(len(vmin_param.index))}
    
    a_param = (pa_params[ pa_params['param']=='Activity']).reset_index().drop(['index'],axis=1)
    a_dict = {list(a_param['rxn'])[i]:list(a_param['value'])[i]*list(a_param['direction'])[i] for i in range(len(a_param.index))}
    
    amin_param = (pa_params[ pa_params['param']=='Activity_min']).reset_index().drop(['index'],axis=1)
    amin_dict = {list(amin_param['rxn'])[i]:list(amin_param['value'])[i]*list(amin_param['direction'])[i] 
                 for i in range(len(amin_param.index))}
    
    return {'a_dict':a_dict,'Km_dict':Km_dict,'klach_dict':klach_dict,
            'Vmax_dict':vmax_dict,'Vmin_dict':vmin_dict,'amin_dict':amin_dict}
    
    
def set_PA( model, ptot, a_dict, amin_dict, profile, klach_dict ):
    '''Set proteome allocation constraints.
    :param model: genome-scale metabolic model
    :param ptot: total protein amount (g/gDW)
    :param a_dict: a dictionary of enzyme activities
    :param amin_dict: a dictionary of minimal enzyme activities
    :param profile: concentrations of extracellular metabolites
    :param klach_dict: a dictionary of inhibition coefficients
    :return expression of proteome allocation constraints
    '''
    sigma = 0.5
    # anabolism and transportation
    A_lcts_e = LacIn( a_dict['EX_lcts_e'], amin_dict['EX_lcts_e'], profile['lac__L_e'],klach_dict['EX_lcts_e'] )
    expr = model.reactions.Growth.flux_expression/a_dict['Growth'] +\
           model.reactions.EX_lcts_e.flux_expression/A_lcts_e  +\
           model.reactions.EX_ac_e.flux_expression/(sigma*a_dict['EX_ac_e']) + \
           model.reactions.EX_lac__L_e.flux_expression/(sigma*a_dict['EX_lac__L_e'])

    # catabolic reactions            
    for k in a_dict.keys():
        if ( 'EX_' not in k ) and ('uptake' not in k) and (k!='Growth'):
            expr = expr + model.reactions.get_by_id(k).flux_expression/( sigma*a_dict[k] )
        
    PA = model.problem.Constraint( expression = expr,name = 'PA', lb= 0, ub = 0.5*ptot)
    model.add_cons_vars([ PA ])
    
    return expr



def set_cc( model, species ):
    '''
    Balance carbon flux of uptake and secretion
    '''
    expr = 3*model.reactions.EX_lac__L_e.flux_expression + 2*model.reactions.EX_ac_e.flux_expression +\
            12*model.reactions.EX_lcts_e.flux_expression + 6*model.reactions.EX_gal_e.flux_expression
    
    for aa in ['his__L', 'ile__L', 'leu__L', 'lys__L', 'met__L', 'phe__L', 
               'thr__L', 'trp__L', 'val__L', 'ala__L', 'arg__L', 'asn__L', 
               'asp__L', 'cys__L', 'gln__L', 'glu__L', 'gly', 
               'pro__L', 'ser__L', 'tyr__L','orn']:
        met = model.metabolites.get_by_id( aa+'_e' )
        coeff = met.elements['C']
        expr = expr + coeff*model.reactions.get_by_id('EX_'+aa+'_e').flux_expression
    if species == 'lb':
        expr = expr + 141*model.reactions.EX_caspep_e.flux_expression
    
    cc = model.problem.Constraint( expression = expr,name = 'cc', lb= -1000.0, ub = 0)
    model.add_cons_vars([ cc ])   
    return None


def get_PA( fluxes, a_dict, amin_dict, profile, klach_dict ):
    '''Get predicted proteome allocation in different pathways
    :param fluxes: predicted metabolic fluxes
    :param a_dict: a dictionary of enzyme activities
    :param amin_dict: a dictionary of minimal enzyme activities
    :param profile: concentrations of extracellular metabolites
    :param klach_dict: a dictionary of inhibition coefficients
    :return predicted proteome allocation
    '''
    sigma = 0.5
    
    pa = {'Growth':fluxes['Growth']/a_dict['Growth'],
          'EX_lcts_e':fluxes['EX_lcts_e']/LacIn( a_dict['EX_lcts_e'],amin_dict['EX_lcts_e'], profile['lac__L_e'],klach_dict['EX_lcts_e'] ),
          'EX_ac_e': fluxes['EX_ac_e']/(sigma*a_dict['EX_ac_e']),'EX_lac__L_e':fluxes['EX_lac__L_e']/(sigma*a_dict['EX_lac__L_e'])}
    for k in a_dict.keys():
        if ( 'EX_' not in k ) and ( 'uptake' not in k ) and ( k!='Growth' ) and (k!='PROLYSIS'):
            pa[k] = fluxes[k]/(sigma*a_dict[k])
    pa['Glycolysis'] = pa['HEX1'] + pa['PGI'] + pa['PFK'] + pa['FBA'] + pa['FBP'] + pa['GAPD'] + pa['PGK'] +\
        pa['PGM'] + pa['ENO'] + pa['PYK']
    pa['Lactic acid production'] = pa['LDH_L']
    if 'PFL' in a_dict.keys():
        pa['Acetic acid production'] = pa['PFL'] + pa['PDH'] + pa['PTAr'] + pa['ACKr']
    else:
        pa['Acetic acid production'] = pa['PDH'] + pa['PTAr'] + pa['ACKr']
        
    pa['Acid transport'] = pa['EX_lac__L_e'] + pa['EX_ac_e']
    pa['Galactose utilization'] = pa['GALKr'] + pa['GALT']
    return pa


def set_loose_medium( profile, params, if_milk:bool = False ):
    '''Set medium parameters for flux balance analysis.
    :param profile: concentrations of extracellular metabolites
    :param params: parameters including Vmax, Vmin, Km and klach for ST and LB
    :param if_milk: boolean, if the medium is milk, then True.
    :return medium parameters
    '''
    Vmax_dict = params['Vmax_dict']
    Vmin_dict = params['Vmin_dict']
    Km_dict = params['Km_dict']
    klach_dict = params['klach_dict']
    
    out_medium = {}
    vitamins = ['EX_btn_e','EX_pnto__R_e', 'EX_ribflv_e', 'EX_thm_e', 'EX_fol_e']
    DNA_materials = ['EX_ade_e', 'EX_gua_e', 'EX_xan_e', 'EX_ura_e']
    others = ['EX_mg2_e', 'EX_ca2_e', 'EX_mn2_e', 'EX_fe2_e', 'EX_fe3_e', 'EX_zn2_e', 'EX_cobalt2_e',\
              'EX_cu2_e', 'EX_cl_e', 'EX_so4_e', 'EX_k_e', 'EX_h2o_e', 'EX_h_e', 'EX_pi_e', 'EX_nh4_e']
    
    out_medium['EX_lcts_e'] = 100
    if 'for_e' in profile.keys():
        out_medium['EX_for_e'] = Vmax_dict['EX_for_e'] * profile['for_e']/( profile['for_e'] + Km_dict['EX_for_e'] )
    
    for k in ['EX_his__L_e', 'EX_ile__L_e', 'EX_leu__L_e', 'EX_lys__L_e', 'EX_met__L_e', 
                     'EX_phe__L_e', 'EX_thr__L_e', 'EX_trp__L_e', 'EX_val__L_e', 'EX_ala__L_e', 
                     'EX_arg__L_e', 'EX_asn__L_e', 'EX_asp__L_e', 'EX_cys__L_e', 'EX_gln__L_e', 
                     'EX_glu__L_e', 'EX_gly_e', 'EX_pro__L_e', 'EX_ser__L_e', 'EX_tyr__L_e', 'EX_orn_e']:
        aa_id = k.replace('EX_','')
        if aa_id in profile.keys():
            conc_aa = profile[ aa_id ]
            Vmax_aa = LacIn( Vmax_dict['aa_uptake'],Vmin_dict['aa_uptake'], profile['lac__L_e'], klach_dict['aa_uptake'] )
            out_medium[k] = Vmax_aa * conc_aa/(conc_aa+ Km_dict['aa_uptake'] )
        
    for EX_m in (vitamins+DNA_materials):
        out_medium[EX_m] = 10
    for EX_m in others:
        out_medium[EX_m] = 1000
        
    if if_milk:
        out_medium['EX_caspep_e'] = LacIn( Vmax_dict['PROLYSIS'],Vmin_dict['PROLYSIS'],profile['lac__L_e'], klach_dict['PROLYSIS'] )
        out_medium['EX_ade_e'] = 0
        out_medium['EX_gua_e'] = 0
        out_medium['EX_xan_e'] = 0
        
    return out_medium
    

    


def run_pfba_pa( model, ptot, growth_func, profile, params ):
    '''Run dynamic simulation for community FBA of ST and LB
    :param model: genome-scale metabolic model
    :param ptot: total protein amount (g/gDW)
    :param growth func: growth functions of ST and LB
    :param profile: concentrations of extracellular metabolites
    :param params: parameters including enzyme activities, Vmax, Vmin, Km and klach for ST and LB
    :return predicted metabolic fluxes
    '''
    with model:
        model.medium = set_loose_medium( profile, params, if_milk = True  )
        set_PA( model, ptot, params['a_dict'], params['amin_dict'], profile, params['klach_dict'] )
        model.reactions.MTHFC.upper_bound = 5e-3#constrain synthesis of 10fthf from methf
        model.reactions.Growth.build_reaction_from_string( growth_func )
        EM_func = growth_func.split('-->')[1].strip() + ' --> ' + growth_func.split('-->')[0]
        model.reactions.EM.build_reaction_from_string( EM_func )
        fluxes = cobra.flux_analysis.pfba(model)
        
    return fluxes
        


def dfba_stlb_pa( st, lb, T, ic, ptot_st, ptot_lb, param_path, st_growth_func, lb_growth_func ):
    '''Run dynamic FBA for the community of ST and LB
    :param st: genome-scale metabolic model of ST
    :param lb: genome-scale metabolic model of LB
    :param T: time range
    :param ic: initial condition (biomass, metabolite concentrations)
    :param ptot_st: total protein concentration of ST(g/gDW)
    :param ptot_lb: total protein concentration of LB(g/gDW)
    :param param_path: path to table of parameters
    :param st_growth_func: growth function of ST
    :param lb_growth_func: growth function of LB
    :return concentration profiles, ST's metabolic fluxes, LB's metabolic fluxes in input time range
    '''
    st_params = get_pa_params( param_path, 'st' )
    lb_params = get_pa_params( param_path, 'lb' )
    times,step = np.linspace(0,T,num= 200,retstep=True)
    ic['lac__L_e'], ic['ac_e'] = 0,0
    profile = {key: [ic[key]] for key in ic.keys() }
    st_f, lb_f = {'Growth':[],'EX_lcts_e':[]}, {'Growth':[],'EX_lcts_e':[],'EX_caspep_e':[] }
    for k in ic.keys():
        if ( ('EX_'+k) not in st_f) and (k not in ['st','lb']):
            st_f[ 'EX_'+k ], lb_f[ 'EX_'+k ] =[],[]
            
    for i in range(len(times)-1):
        profile_t = {k: profile[k][i] for k in profile.keys() }#concentration profile snapshot at time t = i
        st_fluxes = run_pfba_pa( st, ptot_st, st_growth_func, profile_t, st_params ) 
        lb_fluxes = run_pfba_pa( lb, ptot_lb, lb_growth_func, profile_t, lb_params )
        for k in st_f.keys():
            st_f[k].append( st_fluxes[k] )
        for k in lb_f.keys():
            lb_f[k].append( lb_fluxes[k] )
        
        for k in profile.keys():
            if k == 'st':
                profile['st'].append( profile['st'][i] + st_f['Growth'][i]*profile['st'][i]*step )
            elif k == 'lb':
                profile['lb'].append( profile['lb'][i] + lb_f['Growth'][i]*profile['lb'][i]*step )
            else:
                profile[k].append( max( 0, profile[k][i] + lb_f[ 'EX_'+k ][i]*profile['lb'][i]*step +\
                                  st_f[ 'EX_'+k ][i]*profile['st'][i]*step ) )#concentration >= 0
    profile['T'] = times
                
    return profile, st_f, lb_f
                    
              
    
    
    

