#### Set up the system interface
from src.models import models
from src.experiments.TSE import TSE
from src.experiments.Cumulative import Cumulative
from src.experiments.MinimalPair import MinimalPair
from src.experiments.Interact import Interact
from src.experiments.Incremental import Incremental
from src.experiments.Acceptability import Acceptability
import os
import sys

import configparser
import yaml
import json

try:
    from progress.bar import Bar
    PROGRESS = True
except ModuleNotFoundError:
    PROGRESS = False

if __name__ == "__main__":

    path_config = configparser.ConfigParser()
    path_config.read('configs/path_config.cfg')

    if len(sys.argv) != 2:
        print('Did not pass in run_config file, using default: configs/run_config.yaml')
        run_config_fname = 'configs/run_config.yaml'
    else:
        run_config_fname = sys.argv[1]

    extension = run_config_fname.split('.')[-1]
    with open(run_config_fname, 'r') as f:
        if extension == 'yaml':
            run_config = yaml.load(f, Loader=yaml.FullLoader)
        elif extension == 'json':
            run_config = json.load(f)
        else:
            sys.stderr.write(f'Filetype: {extension} not recognized...\n')
            sys.exit()

    #add paths to run_config
    run_config['nc_path'] = path_config['libraries']['neural-complexity']

    if 'plot' in run_config['models']:

        fnames = run_config['stimuli']
        assert len(fnames) == 1
        fname = fnames[0]

        expname = run_config['exp']

        if expname == 'TSE':
            exp = TSE(fname)
            exp.load_dataframe()
        elif expname == 'Cumulative':
            exp = Cumulative(fname)
        else:
            sys.stderr.write(f"Plotting for exp {expname} has not been implemented\n")
            sys.exit(1)

        print(f"Plotting the {expname} results in {fname}...")

        assert 'x' in run_config, 'X value column name needs to be specified'
        assert 'y' in run_config, 'Y value column name needs to be specified'

        x = run_config['x']
        y = run_config['y']
        if 'hue' in run_config:
            hue = run_config['hue']
        else:
            hue = None

        Ys = ['raw', 'diff', 'acc']

        columns = set(exp.dataframe.columns.tolist())
        assert x in columns, f"{x} column not in dataframe"
        assert y in Ys, f"{y} not in options: {Ys}"
        if hue is not None:
            assert hue in columns, f"{hue} column not in dataframe"

        if 'savefname' in run_config:
            savefname = run_config['savefname']
        else:
            savefname=None

        exp.plot(x, y, hue, savefname)

    elif run_config['exp'] == 'CheckVocab':
        """
        Assuming for the moment you just want to check target 
        column in targeted syntactic evaluations.
        """

        LMs = models.load_models(run_config)

        fnames = run_config['stimuli']
        assert len(fnames) == 1
        fname = fnames[0]

        exp = TSE(fname)
        exp.load_dataframe()

        targets = exp.dataframe['target'].tolist()

        for model in LMs:
            for target in targets:
                if model.word_in_vocab(target):
                    continue
                print(f"{target} not in {model} vocab")

    elif run_config['exp'] == 'TSE':

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
            outname += '_'+str(model).replace('/', '_') + '_' + run_config['return_type']+'.tsv'
            print(f"Saving the output to {outname}...")
            exp.save(outname)

    elif run_config['exp'] == 'Cumulative':

        LMs = models.load_models(run_config)
        fnames = run_config['stimuli']
        assert len(LMs) == len(fnames), "Add a stimuli file for each model (include duplicate entry if using the same file)"
        for fname, model in zip(fnames, LMs): 
            print(f"Running {model} on the data in {fname}...")
            exp = Cumulative(fname)

            if 'batchSize' in run_config:
                exp.get_likelihood_results(model,
                                           batch_size=run_config['batchSize'],
                                           log=run_config['log'])
            else:
                exp.get_likelihood_results(model,
                                           log=run_config['log'])

            outname = 'results/LL_'+str(fname).split('/')[-1].split('.tsv')[0]
            outname += "_"+str(model).replace('/', '_')+'_prob.tsv'
            print(f"Saving the output to {outname}...")
            exp.save(outname)

    elif run_config['exp'] == 'Interact':
        #this is lazy and should be fixed to really check for one model
        if 'lstm' not in run_config['models']:
            assert len(run_config['models']) == 1, "Only one model type allowed for Interact"
            for model_type in run_config['models']:
                assert len(run_config['models'][model_type]) == 1, "Only one model allowed for Interact"
        exp = Interact(run_config)
        exp.run_interact()

    elif run_config['exp'] == 'MinimalPair':

        exp = MinimalPair()

        assert 'paradigm' in run_config, "You need to specify paradigm"
        paradigm = run_config['paradigm'] 

        if paradigm == 'SLING':
            assert 'path' in run_config, "You need to specify path to SLING"
            if 'phenomenon' in run_config:
                phenomenon = run_config['phenomenon']
                print(f'Loading only {phenomenon} on SLING')
                exp.load_experiment(paradigm,
                                    run_config['path'], 
                                   phenomenon)
            else:
                exp.load_experiment(paradigm,
                                    run_config['path'])

        elif paradigm == 'precompiled':
            assert 'path' in run_config, "You need to specify path to load"
            exp.load_dataframe(run_config['path'])
        else:
            exp.load_experiment(paradigm)

        if paradigm != 'precompiled':
            LMs = models.load_models(run_config)

            for model in LMs:
                print(f"Running {model} on {paradigm}...")
                exp.get_results(model, run_config['measure'])

            outname = f'results/{paradigm}.tsv'

            print(f"Saving the output to {outname}...")
            exp.flatten()
            exp.save(outname)

        else:
            exp.flatten()
            print(exp.dataframe.groupby(['phenomenon'])['score'].mean())

    elif run_config['exp'] == 'Acceptability':

        LMs = models.load_models(run_config)
        fname = run_config['stimuli'][0]

        # Load unigram model
        with open('src/models/PileUnigramModel.json', 'r') as f:
            UnigramModel = json.load(f)

        exp = Acceptability(fname)

        for model in LMs:
            exp.get_acceptability_measures(model, UnigramModel)

        outname = 'results/'+fname.split('/')[-1].split('.')[0]+'.tsv'
        print(f"Saving the output to {outname}...")
        exp.flatten()
        exp.save(outname)
    else:
        sys.stderr.write(f"The exp {run_config['exp']} has not been implemented\n")
        sys.exit(1)
