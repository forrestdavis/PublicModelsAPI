import pandas as pd

fnames = ['../stimuli/IC_mismatch_BERT.csv', '../stimuli/IC_mismatch.csv']

for fname in fnames:

    data = pd.read_csv(fname)
    verbs = data['verb'].to_list()
    sents = data['sent'].to_list()

    new_sents = []
    for verb, sent in zip(verbs, sents):
        new_sent = sent.replace(verb, f"was {verb} by")
        new_sents.append(new_sent)

    data['sent'] = new_sents

    outname = fname.replace('mismatch', 'passive')
    data.to_csv(outname, index=False)
