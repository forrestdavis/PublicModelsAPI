
import glob
import torch
import sys
sys.path.append('../../private-neural-complexity')

#USE VERSION OF TORCH FROM OLD 

#old_model_path = '/data/wikitext103-25-models/'
old_model_path = '/home/forrestdavis/Data/es_models/'
#save_path = 'old_state_dicts/'
save_path = 'es_old_state_dicts/'

old_model_fnames = glob.glob(old_model_path+'*.pt')

for old_model_fname in old_model_fnames:
    print(old_model_fname)
    with open(old_model_fname, 'rb') as f:
        model = torch.load(f)
        torch.save(model.state_dict(), save_path+old_model_fname.split('/')[-1])

