from lstm_model import LSTMModel

if __name__ == '__main__':

    '''
    import NaturalStories as NS
    documents = NS.load_natural_stories_spr(natural_stories_path = '../../data/naturalstories')
    config = {'name': 'gpt2'}
    gpt2 = GPT2Model(config)
    for doc in documents:
        for sentence in doc:
            print(sentence.text)
            word_surps = gpt2.get_aligned_words_surprisals(sentence.text)

        print(len(doc.sentences))
        sentence_texts = list(map(lambda x: x.text, doc.sentences[:4]))
        word_surps = gpt2.get_aligned_words_surprisals(sentence_texts)
        print(word_surps)
    '''

    config = {'model_file': '/data/base_lstms/wikitext-103_cross_LSTM_256_0-d0.2.pt',
            'vocab_file': '/data/base_lstms/wikitext_103_vocab',
            'path_to_neural_complexity': '/home/forrestdavis/Projects/private-neural-complexity'
            }

    lstm = LSTMModel(**config)
    #print(lstm.get_output(['The man are']))

    '''
    texts = ['the man, who she loves, is happy', 'The man is']
    print(texts)
    word_surps = lstm.get_aligned_words_surprisals(texts, False)
    print()
    for text in word_surps:
        for word in text:
            print(word.word, word.surp)
        print()
    #print(gpt2.word_to_idx('the'))
    #print(gpt2.word_to_idx('the', True))

    print('SIM')
    texts = ['the man who is', 'the man is']
    baselines = ['is', 'is']
    '''
    '''
    _, _, baseline_vector = lstm.get_hidden_layers(['the woman', 'the man'])
    baseline_vector = baseline_vector[2][:,-1,:]
    '''

    '''
    #hidden_layers = lstm.get_hidden_layers(texts)
    #lstm.get_layerwise_similarity_to_baseline(texts, baseline_vector, 2)
    aligned_words_similarities = lstm.get_aligned_words_similarities(texts, baselines, 2)

    for sent in aligned_words_similarities:
        for word in sent:
            print(word.word, word.sim)
        print()
    '''
