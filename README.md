# ModelsAPI

Shared containers for experiments that I developed for my dissertation. The aim is 
to have a unified API for running models on typical experiments in neural model
interpretability. This is done for LSTMs/RNNs by using a version of
[neural-complexity](https://github.com/vansky/neural-complexity), 
developed by my advisor Marten van Schijndel. Transformers are accessed via
HuggingFace's API. Note that this code is for inference, and thus, we will
assume pretrained models. An additional library which is nice for doing this is
[minicons](https://github.com/kanishkamisra/minicons) developed by Kanishka
Misra. That package is more fully developed, so that's always an option. 

## Dependencies

I will assume you have conda and can subsequently use that to create a virtual
environment. I've included a yml file to facilitate this. If you are on linux or
mac without M1 run the following

```
conda env create -n mapi --file ModelsAPI.yml
```

If you are on a MAC with M1, run the following 

```
source setupMAC.sh
```

Running the above will create a environment named mapi which should work for
this code. If you run into errors on Mac with M1 see this
[blog](https://jamescalam.medium.com/hugging-face-and-sentence-transformers-on-m1-macs-4b12e40c21ce). 

## Evaluation data

Gulordava's data for English can be found [here](https://github.com/facebookresearch/colorlessgreenRNNs/tree/main/data). What you'll want to do 
is create minimal pairs with context as the sentence and target as the differing part. Then it should be straightforward 
to evaluate some transformers.

## Quick run

To run the code, simply enter: 

```
python main.py
```

This will use the default config file (elaborated on below) run\_config.yml. You
can pass in a different config file as below: 

```
python main.py new_config.yml
```

## Config files

Running an experiment is done by specifying a config file. An example one is
copied below: 

```
exp: TSE

models: 
      bert: 
          - bert-base-uncased
      gpt2:
          - gpt2

return_type: prob

stimuli:
    - stimuli/tiny_IC_mismatch_BERT.tsv
    - stimuli/tiny_IC_mismatch.tsv

include_punct: False

lower: True

```

I describe each parameter below. 


### exp

There are three options: TSE, Incremental, and Interactive. TSE does targeted
syntactic evaluations, so you will have some context and a target and this will
look at that target conditioned on the context. Incremental calculates the by
word measures. Interactive allows you to test out sentences on the command line
and see the incremental surprisal and probability values for each word. 

### models

Models require a model type (bert|roberta|gpt2|lstm|tfxl|gptneo|gptj) which
tells the api which model architecture the model is drawn from. Then the name of
the model (or many models of the same type) are provided under the model type
using -. I've lazily done this, so you can run a few models at once, but very
large ones will probably cause problems because all models are loaded into
memory. The above sample config file will run the pretrained bert-base-uncased
model and smallest gpt2 model provided by huggingface. This is a general
property of all tranformer models with this pipeline, passing in a name will trigger a check on
huggingface and that model will be loaded. You can also specify a path to a
local copy of a model (e.g., /data/gpt2). 

### return_type

This is either prob (for probability) or surp (for surprisal) and will be the
measure returned from a model.

### stimuli

Name of stimuli file to use for the experiment. Each stimuli file will be
sequentially associated with each model. So the above will try to run
bert-base-uncased on stimuli/tiny\_IC\_mismatch\_BERT.tsv and gpt2 on
stimuli/tiny\_IC\_mismatch.tsv. Examples files are provided for incremental and
TSE experiments. TSE needs at least two columns one called called context, which
gives the context sentence (could be bidirectional, more on this in a moment)
and another column called target which gives the target. The target should be
one word (i.e. not split by the model). Some variables are also an option, I
turn to these at the bottom. Incremental experiments expect one column called
sent which gives the sentence. 

For bidirectional models you can pass in a full context and target a medial
word. Use MASKTOKEN as the special token and the model specific token will be
inserted in this location. 

### include\_punct

Whether to include the punctuation in the surp/probability calculation. This is
only used by incremental and interactive. Notice that in these use cases, a
given tokenizer might split a word into subwords. Right now I flag this and
treat the probability of that whole word as the joint probability of its
subparts. So if the word 'human' is mapped to 'hu' + 'man', then the probability
of 'human' will be the probability of 'hu' times the probability of 'man'
(conditioned on their respective contexts). This is more tricky for
bidirectional models, so we can discuss this if you want.  

### lower

Whether to lowercase the first word in the sentence. We might want to lowercase
more, but I didn't code that for some reason I've forgotten. 

## Wildcards

I've added two variables which may be useful. These can be inserted as
targets and a larger set of items will be checked. The wildcards are:

\$SG maps to English third person singular verbs

\$PL maps to English third person plural verbs 

The values for all singular/plural verbs will be summed and one value returned.
Thus, this only makes sense if the return type is prob, but I don't check this. 

## Results

The results of an experiment are complied in tsv files under the results
directory. The resultant name is hard coded as the following: 

```
results/{fname}_{modelname}_{return_type}.tsv
```

The organization of the file should be straightforwardly interpretable by
looking at the output. 

## Colab

This code can be straightforwardly run on colab where you can access (free)
GPUs. I've included a small document in the colab folder which outlines how to
link github and google drive. Once that's in place, the included colab.ipynb
script can be run. 

