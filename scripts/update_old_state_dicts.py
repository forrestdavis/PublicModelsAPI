import glob
import torch
import sys
sys.path.append('../../private-neural-complexity')
from model import RNNModel

#USE VERSION OF TORCH FROM NEW

#new_model_path = '/data/new-wikitext103-25-models/'
new_model_path = '/data/new_es_models/'
#save_path = 'old_state_dicts/'
save_path = 'es_old_state_dicts/'

old_model_fnames = glob.glob(save_path+'*.pt')

for old_model_fname in old_model_fnames:
    #m = RNNModel('LSTM', 50002, 400, 400, 2, dropout=0.2, tie_weights=True)
    m = RNNModel('LSTM', 50002, 650, 650, 2, dropout=0.2, tie_weights=True)
    old_state_dict = torch.load(old_model_fname)

    fname = old_model_fname.split('/')[-1]

    m.load_state_dict(old_state_dict)

    with open(new_model_path+fname, 'wb') as f:
        torch.save(m, f)
