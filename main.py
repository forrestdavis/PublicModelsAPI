#### Set up the system interface
from src.models import models
from src.experiments.TSE import TSE
from src.experiments.Cumulative import Cumulative
from src.experiments.BLiMP import BLiMP
from src.experiments.Interact import Interact
from src.experiments.Incremental import Incremental
import os
import sys

import configparser
import yaml

try:
    from progress.bar import Bar
    PROGRESS = True
except ModuleNotFoundError:
    PROGRESS = False

if __name__ == "__main__":

    path_config = configparser.ConfigParser()
    path_config.read('path_config.cfg')

    if len(sys.argv) != 2:
        print('Did not pass in run_config file, using default: run_config.yaml')
        run_config_fname = 'run_config.yaml'
    else:
        run_config_fname = sys.argv[1]

    with open(run_config_fname, 'r') as f:
        run_config = yaml.load(f, Loader=yaml.FullLoader)

    #add paths to run_config
    run_config['nc_path'] = path_config['libraries']['neural-complexity']

    if run_config['exp'] == 'TSE':
        LMs = models.load_models(run_config)
        fnames = run_config['stimuli']
        assert len(LMs) == len(fnames), "Add a stimuli file for each model (include duplicate entry if using the same file)"
        for fname, model in zip(fnames, LMs): 
            exp = TSE(fname)
            print(f"Running {model} on the data in {fname}...")
            exp.get_targeted_results(model, lowercase=run_config['lower'],
                                     return_type=run_config['return_type'])

            outname = 'results/'+fname.split('/')[-1].split('.tsv')[0]
            outname += '_'+str(model).replace('/', '_') + '_' + run_config['return_type']+'.tsv'
            print(f"Saving the output to {outname}...")
            exp.save(outname)

    elif run_config['exp'] == 'Incremental':
        LMs = models.load_models(run_config)
        fnames = run_config['stimuli']
        assert len(LMs) == len(fnames), "Add a stimuli file for each model (include duplicate entry if using the same file)"
        for fname, model in zip(fnames, LMs): 
            exp = Incremental(fname)
            print(f"Running {model} on the data in {fname}...")
            exp.get_incremental(model, lowercase=run_config['lower'],
                                include_punctuation=run_config['include_punct'], 
                                return_type=run_config['return_type'])

            outname = 'results/'+fname.split('/')[-1].split('.tsv')[0]
            outname += '_'+str(model) + '_' + run_config['return_type']+'.tsv'
            print(f"Saving the output to {outname}...")
            exp.save(outname)

    elif run_config['exp'] == 'Cumulative':
        LMs = models.load_models(run_config)
        fnames = run_config['stimuli']
        assert len(LMs) == len(fnames), "Add a stimuli file for each model (include duplicate entry if using the same file)"
        for fname, model in zip(fnames, LMs): 
            print(f"Running {model} on the data in {fname}...")
            exp = Cumulative(fname)
            exp.get_likelihood_results(model)

            outname = 'results/LL_'+str(fname).split('/')[-1].split('.tsv')[0]
            outname += "_"+str(model)+'_prob.tsv'
            print(f"Saving the output to {outname}...")
            exp.save(outname)

    elif run_config['exp'] == 'Interact':
        assert len(run_config['models']) == 1, "Only one model type allowed for Interact"
        for model_type in run_config['models']:
            assert len(run_config['models'][model_type]) == 1, "Only one model allowed for Interact"
        exp = Interact(run_config)
        exp.run_interact()

    elif run_config['exp'] == 'BLiMP':
        exp = BLiMP()
        LMs = models.load_models(run_config)

        for model in LMs:
            print(f"Running {model} on BLiMP...")
            exp.get_likelihood_results(model)

            outname = 'results/BLiMP_'+str(model).split('/')[-1].split('.')[0]+'.tsv'
            print(f"Saving the output to {outname}...")
            exp.save(outname)
    else:
        sys.stderr.write(f"The exp {run_config['exp']} has not been implemented\n")
        sys.exit(1)
