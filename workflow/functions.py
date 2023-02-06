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
    pH = C1*lac_con + C2
    lach = lac_con/(10**(pH-3.86))
    temp_value = abs(v_max)*exp( -klach*lach )
    if v_max > 0:
        out = max(temp_value, abs(v_min) )
    else:
        out = -1*max( temp_value, abs(v_min) )
    return out


# def update_LacIn_linear( vmax, vmin, klach, C1, C2, lac_con ):
#     lach = lac_con/( 1+exp( C1*lac_con + C2 ) )
#     ub = vmax*LacIn( klach, C1, C2, lac_con ) + vmin
#     return ub


def update_MM( vmax, s, Km ):
    ub = (vmax * s)/(Km+s)
    return ub

def set_aaMM( model, aa, vmax ):
    # constrain transporter rate
    rxns = [r for r in model.metabolites.get_by_id(aa+'_e').reactions if ('EX_' not in r.id ) 
                    and ('c2e' not in r.id ) and ('LYSIS' not in r.id)]
    expr = rxns[0].flux_expression
    if len(rxns) > 1:
        expr = expr + rxns[1].flux_expression
    MM = model.problem.Constraint( expression = expr, name = aa+'_MM', lb = 0.0, ub = vmax )
    model.add_cons_vars([ MM ])
    
    return None

def get_pa_params( path, species ):
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
    sigma = 0.5
    # anabolism and transportation
    A_lcts_e = LacIn( a_dict['EX_lcts_e'], amin_dict['EX_lcts_e'], profile['lac__L_e'],klach_dict['EX_lcts_e'] )
    expr = model.reactions.Growth.flux_expression/a_dict['Growth'] +\
           model.reactions.EX_lcts_e.flux_expression/A_lcts_e  +\
           model.reactions.EX_ac_e.flux_expression/(sigma*a_dict['EX_ac_e']) + \
           model.reactions.EX_lac__L_e.flux_expression/(sigma*a_dict['EX_lac__L_e'])

    #add flux constraint for amino acid secretion and uptake
#     for aa in ['his__L', 'ile__L', 'leu__L', 'lys__L', 'met__L', 'phe__L', 
#                'thr__L', 'trp__L', 'val__L', 'ala__L', 'arg__L', 'asn__L', 
#                'asp__L', 'cys__L', 'gln__L', 'glu__L', 'gly', 
#                'pro__L', 'ser__L', 'tyr__L','orn']:
        
#         if aa != 'orn':
#             rxn_id = (aa.split('__L')[0]).upper() + 'c2e'
#             rxn = model.reactions.get_by_id( rxn_id )
#             expr = expr + rxn.flux_expression/( sigma*a_dict['EX_ac_e'] )
        
#         sigma_aa = ( profile[aa+'_e'])/( profile[aa+'_e'] + km_aa )
#         if sigma_aa > 0:
#             coeff = 1/(sigma_aa*LacIn(a_dict['aa_uptake'],profile['lac__L_e'],klach_dict['aa_uptake'] ) )
#             rxns = [r for r in model.metabolites.get_by_id(aa+'_e').reactions if ('EX_' not in r.id ) 
#                     and ('c2e' not in r.id ) and ('LYSIS' not in r.id)]
#             for rxn in rxns:
#                 expr = expr + coeff*rxn.flux_expression
                
    # catabolic reactions            
    for k in a_dict.keys():
        if ( 'EX_' not in k ) and ('uptake' not in k) and (k!='Growth'):
            expr = expr + model.reactions.get_by_id(k).flux_expression/( sigma*a_dict[k] )
        
    PA = model.problem.Constraint( expression = expr,name = 'PA', lb= 0, ub = 0.5*ptot)
    model.add_cons_vars([ PA ])
    
    return expr


def set_memPA( model, ptot, a_dict, profile, klach_dict ):
    sigma = 0.5
    expr = model.reactions.EX_lcts_e.flux_expression/LacIn( a_dict['EX_lcts_e'], profile['lac__L_e'], klach_dict['EX_lcts_e'] ) +\
                          model.reactions.EX_ac_e.flux_expression/(sigma*a_dict['EX_ac_e']) +\
                          model.reactions.EX_lac__L_e.flux_expression/(sigma*a_dict['EX_lac__L_e'])
    
    memPA = model.problem.Constraint( expression = expr,name = 'memPA', lb= 0, ub = 0.1*ptot)
    model.add_cons_vars([memPA])
    
    return expr



def set_cc( model, species ):
    '''
    Contron carbon flux in secretion
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

# def get_PA( fluxes, a_dict, amin_dict, profile, klach_dict ):
#     sigma = 0.5
    
#     pa = {'Growth':fluxes['Growth']/a_dict['Growth'],
#           'EX_lcts_e':fluxes['EX_lcts_e']/LacIn( a_dict['EX_lcts_e'],amin_dict['EX_lcts_e'], profile['lac__L_e'],klach_dict['EX_lcts_e'] ),
#           'EX_ac_e': fluxes['EX_ac_e']/(sigma*a_dict['EX_ac_e']),'EX_lac__L_e':fluxes['EX_lac__L_e']/(sigma*a_dict['EX_lac__L_e'])}
#     for k in a_dict.keys():
#         if ( 'EX_' not in k ) and ( 'uptake' not in k ) and ( k!='Growth' ) and (k!='PROLYSIS'):
#             pa[k] = fluxes[k]/(sigma*a_dict[k])
#     pa['Glycolysis'] = pa['HEX1'] + pa['PGI'] + pa['PFK'] + pa['FBA'] + pa['FBP'] + pa['GAPD'] + pa['PGK'] +\
#         pa['PGM'] + pa['ENO'] + pa['PYK']
#     pa['Lactic acid production'] = pa['LDH_L']
#     if 'PFL' in a_dict.keys():
#         pa['Acetic acid production'] = pa['PFL'] + pa['PDH'] + pa['PTAr'] + pa['ACKr']
#     else:
#         pa['Acetic acid production'] = pa['PDH'] + pa['PTAr'] + pa['ACKr']
        
#     pa['Acid transport'] = pa['EX_lac__L_e'] + pa['EX_ac_e']
#     pa['Galactose utilization'] = pa['GALKr'] + pa['GALT']
#     return pa


def setNGAM( model, ngam ):
#     NGAM = model.problem.Constraint( model.reactions.ATPM.flux_expression, lb=ngam, ub=model.reactions.ATPM.upper_bound )
#     model.add_cons_vars( NGAM )
    model.reactions.ATPM.lower_bound = ngam
    return None



def run_pfba_lb( model, medium, lac_ratio:float=1.749577, biomass_func:str=None, profile:dict=None, params:dict=None ):
    with model as model:
        model.medium = medium
        fixLacGlc(model, lac_ratio)#fix lactate:Lactose ratio
        if biomass_func is not None:
            model.reactions.Growth.build_reaction_from_string(biomass_func)
        for aa in ['his__L', 'ile__L', 'leu__L', 'lys__L', 'met__L', 'phe__L', 
                   'thr__L', 'trp__L', 'val__L', 'ala__L', 'arg__L', 'asn__L', 
                   'asp__L', 'cys__L', 'gln__L', 'glu__L', 'gly', 'pro__L', 
                   'ser__L', 'tyr__L']:
            set_aaMM( model, aa, params['EX_'+aa+'_e']['vmax'],
                     profile[aa+'_e'], params['EX_'+aa+'_e']['Km'] )
        
        fluxes = cobra.flux_analysis.pfba(model)
    return fluxes

def run_pfba( model, medium,ngam, lac_ratio:float=1.749577, biomass_func:str=None ,con_dict:dict=None ):
    with model as model:
        model.medium = medium
        setNGAM(model, ngam )
        fixLacGlc(model, lac_ratio)#fix lactate:Glucose ratio
        if biomass_func is not None:
            model.reactions.Growth.build_reaction_from_string(biomass_func)
        if con_dict is not None:
            for r in con_dict:
                model.reactions.get_by_id(r).lower_bound = con_dict[r]['lb']
                model.reactions.get_by_id(r).upper_bound = con_dict[r]['ub']
        fluxes = cobra.flux_analysis.pfba(model)
    return fluxes




def init_medium( species,ic, params ):
    out_medium = {}
    vitamins = ['EX_btn_e','EX_pnto__R_e', 'EX_ribflv_e', 'EX_thm_e', 'EX_fol_e']
    DNA_materials = ['EX_ade_e', 'EX_gua_e', 'EX_xan_e', 'EX_ura_e']
    others = ['EX_mg2_e', 'EX_ca2_e', 'EX_mn2_e', 'EX_fe2_e', 'EX_fe3_e', 'EX_zn2_e', 'EX_cobalt2_e',\
              'EX_cu2_e', 'EX_cl_e', 'EX_so4_e', 'EX_k_e', 'EX_h2o_e', 'EX_h_e', 'EX_pi_e', 'EX_nh4_e']
    
    out_medium['EX_lcts_e'] = params['EX_lcts_e']['vmax']
    for k in ['EX_his__L_e', 'EX_ile__L_e', 'EX_leu__L_e', 'EX_lys__L_e', 'EX_met__L_e', 
                     'EX_phe__L_e', 'EX_thr__L_e', 'EX_trp__L_e', 'EX_val__L_e', 'EX_ala__L_e', 
                     'EX_arg__L_e', 'EX_asn__L_e', 'EX_asp__L_e', 'EX_cys__L_e', 'EX_gln__L_e', 
                     'EX_glu__L_e', 'EX_gly_e', 'EX_pro__L_e', 'EX_ser__L_e', 'EX_tyr__L_e', 'EX_orn_e']:
        out_medium[k] = update_MM( params[k]['vmax'], ic[k.replace('EX_','')],params[k]['Km'] )    
        
    for EX_m in (vitamins+DNA_materials):
        out_medium[EX_m] = 0.1
    for EX_m in others:
        out_medium[EX_m] = 1000
        
    if species == 'lb':
        out_medium['EX_caspep_e'] = params['EX_caspep_e']['vmax']
        
        
    return out_medium


def set_loose_medium( profile, params, if_milk:bool = False ):
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
    

        
def init_params( species, path ):
    out = {}
    table = pd.read_csv(path)
    table = (table[table['species']==(species.upper())]).reset_index().drop(['index'],axis=1)
    for i in range(len(table['substrate'])):
        x , subs, param, value, y = table.iloc[i]
        if subs not in out:
            out[subs] = {param:value}
        elif param not in out[subs]:
            out[subs][param] = value
    return out   






def add_ecpfba( model,ec_dict,objective,fraction_of_optimum):
    '''
    Parameters
    ----------
    model : cobra.Model
        The model to add the objective to.
    ec_dict: dict
        A dict of enzyme costs for all reactions
    objective : dict or cobra.Model.objective, optional
        A desired objective to use during optimization in addition to the
        pFBA objective. Dictionaries (reaction as key, coefficient as value)
        can be used for linear objectives (default None).
    fraction_of_optimum : float, optional
        Fraction of optimum which must be maintained. The original objective
        reaction is constrained to be greater than maximal value times the
        `fraction_of_optimum`.
    '''
    if objective is not None:
        model.objective = objective
    if model.solver.objective.name == "_pfba_objective":
        raise ValueError("The model already has a pFBA objective.")
    sutil.fix_objective_as_constraint(model, fraction=fraction_of_optimum)
    
    reaction_variables = (
        (rxn.forward_variable, rxn.reverse_variable) for rxn in model.reactions
    )
    variables = chain(*reaction_variables)
    model.objective = model.problem.Objective(
        Zero, direction="min", sloppy=True, name="_pfba_objective"
    )
    model.objective.set_linear_coefficients({v: ec_dict[v.name] for v in variables})
    return None
    
def ecpfba(model,ec_dict,fraction_of_optimum: float = 1.0,
           objective:Union[Dict, "Objective", None] = None,reactions:Optional[List["Reaction"]] = None):
    reactions = (
        model.reactions if reactions is None else model.reactions.get_by_any(reactions) )
    with model as m:
        add_ecpfba(m, ec_dict=ec_dict,objective=objective, fraction_of_optimum=fraction_of_optimum)
        m.slim_optimize(error_value=None)
        solution = get_solution(m, reactions=reactions)
    return solution
    

# def sim_st( st, time, ic, st_params ):  
#     times,step = np.linspace(0, time ,num=100,retstep=True)
#     profiles = {key: [ic[key]] for key in ic.keys() }
#     st_medium = init_medium( 'st',ic, st_params )

#     st_f = {'Growth':[],'EX_glc__D_e':[] }
#     for k in ic.keys():
#         if ( ('EX_'+k) not in st_f) and (k not in ['st']):
#             st_f[ 'EX_'+k ]=[]
  
#     for i in range(len(times)):
#         st_fluxes = run_pfba( st, st_medium, 4 )
#         for k in st_f.keys():
#             st_f[k].append( st_fluxes[k] )

#         if i > 0:
#             for k in profiles.keys():
#                 if k == 'st':
#                     profiles['st'].append( profiles['st'][i-1] + st_f['Growth'][i-1]*profiles['st'][i-1]*step )
#                 else:
#                     profiles[k].append( max( 0, profiles[k][i-1] + st_f[ 'EX_'+k ][i-1]*profiles['st'][i-1]*step ) )


#             st_medium['EX_glc__D_e'] = update_LacIn( st_params['EX_glc__D_e']['vmax'], st_params['EX_glc__D_e']['vmin'],\
#                                           st_params['EX_glc__D_e']['klach'],C1, C2 , profiles['lac__L_e'][i],profiles['nh4_e'][i])
            
#             for k in ['EX_his__L_e', 'EX_ile__L_e', 'EX_leu__L_e', 'EX_lys__L_e', 'EX_met__L_e', 
#                      'EX_phe__L_e', 'EX_thr__L_e', 'EX_trp__L_e', 'EX_val__L_e', 'EX_ala__L_e', 
#                      'EX_arg__L_e', 'EX_asn__L_e', 'EX_asp__L_e', 'EX_cys__L_e', 'EX_gln__L_e', 
#                      'EX_glu__L_e', 'EX_gly_e', 'EX_pro__L_e', 'EX_ser__L_e', 'EX_tyr__L_e']:
#                 st_medium[k] = update_MM( st_params[ k ]['vmax'], profiles[k.replace('EX_','')][i], st_params[ k ]['Km'] )
                    
#     return  profiles,times,st_f

        
def dfba_stlb( st, lb, time, ic, st_params,lb_params ):
    '''
    time: unit is hr, split into #hr*100 steps
    '''
    times,step = np.linspace(0,time,num= (time*100),retstep=True)
    
    if 'lac__L_e' not in ic:
        ic['lac__L_e'] = 0
    if 'nh4_e' not in ic:
        ic['nh4_e'] = 0
        
    profiles = {key: [ic[key]] for key in ic.keys() }
    # initialize growth medium
    st_medium, lb_medium = init_medium( 'st',ic, st_params ), init_medium( 'lb',ic, lb_params )
    
    st_f, lb_f = {'Growth':[],'EX_glc__D_e':[]}, {'Growth':[],'EX_glc__D_e':[],'EX_casein_e':[] }
    for k in ic.keys():
        if ( ('EX_'+k) not in st_f) and (k not in ['st','lb']):
            st_f[ 'EX_'+k ], lb_f[ 'EX_'+k ] =[],[]
            
    for i in range(len(times)):
        st_fluxes = run_pfba( st, st_medium,1)
        lb_fluxes = run_pfba( lb, lb_medium,1)
        for k in st_f.keys():
            st_f[k].append( st_fluxes[k] )
        for k in lb_f.keys():
            lb_f[k].append( lb_fluxes[k] )
            
            
        if i > 0:
            # update concentration profiles
            for k in profiles.keys():
                if k == 'st':
                    profiles['st'].append( profiles['st'][i-1] + st_f['Growth'][i-1]*profiles['st'][i-1]*step )
                elif k == 'lb':
                    profiles['lb'].append( profiles['lb'][i-1] + lb_f['Growth'][i-1]*profiles['lb'][i-1]*step )
                else:
                    profiles[k].append( max( 0, profiles[k][i-1] + lb_f[ 'EX_'+k ][i-1]*profiles['lb'][i-1]*step +\
                                      st_f[ 'EX_'+k ][i-1]*profiles['st'][i-1]*step ) )#concentration >= 0         
                           
            #update medium for st and lb
            C1, C2 = -0.126, 5.999
            st_medium['EX_glc__D_e'] = update_LacIn_linear( st_params['EX_glc__D_e']['vmax'], st_params['EX_glc__D_e']['vmin'],\
                                    st_params['EX_glc__D_e']['klach'],C1, C2, profiles['lac__L_e'][i])
          
            lb_medium['EX_glc__D_e'] = update_LacIn_linear( lb_params['EX_glc__D_e']['vmax'], lb_params['EX_glc__D_e']['vmin'],\
                                         lb_params['EX_glc__D_e']['klach'], C1, C2, profiles['lac__L_e'][i])
            
            
            for k in ['EX_his__L_e', 'EX_ile__L_e', 'EX_leu__L_e', 'EX_lys__L_e', 'EX_met__L_e', 
                     'EX_phe__L_e', 'EX_thr__L_e', 'EX_trp__L_e', 'EX_val__L_e', 'EX_ala__L_e', 
                     'EX_arg__L_e', 'EX_asn__L_e', 'EX_asp__L_e', 'EX_cys__L_e', 'EX_gln__L_e', 
                     'EX_glu__L_e', 'EX_gly_e', 'EX_pro__L_e', 'EX_ser__L_e', 'EX_tyr__L_e', 'EX_orn_e']:
                
                
                st_medium[ k ] = update_MM( st_params[ k ]['vmax'], profiles[ k.replace('EX_','') ][i], st_params[ k ]['Km'] )
                lb_medium[ k ] = update_MM( lb_params[ k ]['vmax'], profiles[ k.replace('EX_','') ][i], lb_params[ k ]['Km'] )
                      
    return profiles,times,st_f, lb_f


def run_pfba_pa( model, ptot, growth_func, profile, params ):
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
                    
              
    
    
    

