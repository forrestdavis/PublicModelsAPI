# TODO

## Pressing

- [ ] I have a gpt3 script which I need to add to this pipeline more clearly. We can
discuss this. 
- [ ] Plotting
- [ ] Compiling results
- [x] add M1 compatibility (there seems to be radically different outputs)
- [x] Models with mismatching tokenizer and model   
- [ ] Add checks
- [ ] Chinese punctuation

## Additions

- [ ] Slidding window

## Refactoring 

- [ ] Clarify the experimental class and reduce repeated code
- [ ] Consider encapsulating model versions 
- [ ] For experimental classes, clarify the format of the data and accessing it.
        this will have reflexes on plotting that might make it easier 
- [ ] More fully integrate the language flag
- [ ] Custom alignment protocols (could refactor this)
- [ ] Reconsider how to integrate code with Marty's lab

## Models

- [ ] MT5/T5

## Known Bugs

- [x] GPT2Tokenizer throws a bug with <unk> and padding, AutoTokenizer avoids this
- [ ] auto-masked is doing something weird with https://huggingface.co/Langboat/mengzi-bert-base

