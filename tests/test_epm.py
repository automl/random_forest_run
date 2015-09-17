import sys,os

import logging
import numpy as np
import matplotlib.pyplot as plt

from sklearn.ensemble import RandomForestRegressor
import rfr
import functools

from pysmac.utils.smac_input_readers import read_scenario_file
from pysmac.utils.smac_output_readers import read_instances_file
from pysmac.utils.smac_output_readers import read_instance_features_file
from pysmac.utils.smac_output_readers import read_runs_and_results_file
from pysmac.utils.smac_output_readers import read_paramstrings_file
from pysmac.utils.state_merge import read_sate_run_folder

from epm.preprocessing.pre_feature import PreprocessingFeatures
from epm.pcs.config_space import ConfigSpace
from epm.pcs.encode import encode
from epm.preprocessing.imputor_missing_y import ImputorY
from epm.models import bootstrapModel

from sklearn.ensemble.forest import RandomForestRegressor

def _train_epm(ps_data, train_insts, inst_feats, rr_data, scenario_data, logy,  save=None):
    '''
        transform data to EPM style and train an EPM
    '''
    
    logging.info("### Start training EPM")
    
    config_list = []
    perf_list = [] 
    inst_list = []
    suc_list = []
    cen_list = []
    
    for data in rr_data:
        inst_list.append(train_insts[int(data[1])-1][0])
        perf_list.append(data[2])
        if data[3] == 0:
            cen = False
        else:
            cen = True
        cen_list.append(cen)
        if data[8] in [1,0]:
            suc = True
        else:
            suc = False
        suc_list.append(suc)
        config_list.append(ps_data[int(data[0])-1])

    logging.info("Number of training data: %d" %(len(perf_list)))
    #logging.debug("Configs: %s" %(config_list))
    #logging.debug("Perfs: %s" %(perf_list))
    #logging.debug("Insts: %s" %(inst_list))
    #logging.debug("Success: %s" %(suc_list))
    #logging.debug("Censored: %s" %(cen_list))
    
    
    inst_feat_dict = inst_feats[1]

    # normalize features
    #------------------------------- logging.debug("### Normalize features")
    #---------- fpre = PreprocessingFeatures(inst_feats_dict=inst_feat_dict)
    #--- inst_feat_dict = fpre.normalization(inst_feats_dict=inst_feat_dict)
    
    # Convert training data to matrix, impute nonactive params and normalize
    logging.info("### Convert configurations into internal representation")
    pcs_file = scenario_data["paramfile"]
    cs = ConfigSpace(pcs_file)
    config_matrix = np.zeros((len(config_list), cs._n_params))
    for indx, config in enumerate(config_list):
        config_vec = cs.convert_param_dict(config)
        imputed_vec = cs.impute_non_active(config_vec, value="def") #def imputation
        config_matrix[indx] = imputed_vec
        
    #encoded_matrix = encode(X=config_matrix, cat_list=cs._cat_size)
    encoded_matrix = config_matrix
    
    
    logging.info("Number of training data points: %d" %(encoded_matrix.shape[0]))
    
    # convert feature dictionary into feature matrix
    feat_matrix = []
    for inst_ in inst_list:
        feat_matrix.append(inst_feat_dict[inst_])
    feat_matrix = np.array(feat_matrix)
    
    # Impute data for training folds
    if False and (True in cen_list): #disable imputation -- I don't have so much time
        rf = functools.partial(RandomForestRegressor, random_state=1)
        brf = functools.partial(bootstrapModel.bootstrapModel, rng=1,
                         debug=False, n_bootstrap_samples=10,
                         bootstrap_size=0.7, base_estimator=rf)
        
        logging.info("### Impute missing data")
        imputor = ImputorY(debug=False)
        perf_list = imputor.impute_y(y=perf_list,
                                     is_censored=cen_list,
                                     max_y=sys.maxint, #TODO: Check whether this was a good idea
                                     configs_list=encoded_matrix,
                                     inst_list=inst_list,
                                     inst_feature_dict=inst_feat_dict,
                                     model=brf, log=True, rng=None)
    
    if logy or scenario_data.get("run_obj") is None or scenario_data.get("run_obj") == "runtime":
    # Put performance data on logscale
        logging.info("### Using logscale on performance data")
        if min(perf_list) < 0:
            perf_list += min(perf_list)
        perf_list = np.log(perf_list)
    
    encoded_matrix = np.hstack((encoded_matrix,feat_matrix))
        
    logging.info("### Start training Stefans RF")
    
    
    #===========================================================================
    # types_params = []
    # for cat in cs._cat_size:
    #     if cat == 0:
    #         types_params.append(0)
    #     else:
    #         types_params.extend([2]*cat)
    # types_params = np.array(types_params)
    #===========================================================================
    types_params = np.array(cs._cat_size)
    types_feats = np.zeros([feat_matrix.shape[1]],dtype=np.uint32)
    logging.info("Types params shape: %s" %(str(types_params.shape)))
    logging.info("Types feats shape: %s" %(str(types_feats.shape)))
    
    types = np.hstack((types_params,types_feats))
    logging.info("Types shape: %s" %(str(types.shape)))
    types = np.array(types,dtype=np.uint32)

    # cat. params have to start at 
    for idx, t in enumerate(types_params):
        if t > 0:
            encoded_matrix[:,idx] += 1
            
    logging.info("Data shape: %s" %(str(encoded_matrix.shape)))
    
    the_forest = rfr.regression.binary_rss()

    the_forest.seed=12
    the_forest.do_bootstrapping=True
    the_forest.num_data_points_per_tree = 0
    the_forest.max_features_per_split = 80
    the_forest.min_samples_to_split = 20
    the_forest.min_samples_in_leaf = 10
    the_forest.max_depth = 5
    the_forest.epsilon_purity = 1e-8
    the_forest.num_trees = 10
    
    the_forest.fit(encoded_matrix,  perf_list, types)
    
    logging.info("### Plotting RF")
    
    the_forest.save_latex_representation("./rf_tex")
    
    #===========================================================================
    # logging.info("###Fit sklearn RF")
    # 
    # forest = RandomForestRegressor(n_estimators=1)
    # forest.fit(encoded_matrix,  perf_list)
    #===========================================================================

    #===========================================================================
    # logging.info("###Get sklearn feature importance")
    # 
    # importances = forest.feature_importances_
    # std = np.std([tree.feature_importances_ for tree in forest.estimators_],
    #              axis=0)
    # indices = np.argsort(importances)[::-1]
    # 
    # # Print the feature ranking
    # print("Feature ranking:")
    # 
    # for f in range(10):
    #     print("%d. feature %d (%f)" % (f + 1, indices[f], importances[indices[f]]))
    # 
    #===========================================================================
    #===========================================================================
    # 
    # tree = forest.estimators_[0]
    # from sklearn.externals.six import StringIO  
    # import pydot 
    # dot_data = StringIO() 
    # tree.export_graphviz(clf, out_file=dot_data) 
    # graph = pydot.graph_from_dot_data(dot_data.getvalue()) 
    # graph.write_pdf("rf_sklearn.pdf") 
    #===========================================================================


logging.basicConfig(level=logging.INFO)
    
if len(sys.argv) < 2:
    logging.error("Please specify <smac_state_dir> <scenario-file>")
    
if not os.path.islink("target_algorithms"):
    os.symlink("/data/aad/aclib/target_algorithms", "target_algorithms")

scenario_data = read_scenario_file(sys.argv[2])
ps_data, train_insts, inst_feats, rr_data = read_sate_run_folder(directory=sys.argv[1])
_train_epm(ps_data, train_insts, inst_feats, rr_data, scenario_data, logy=True)


    
    