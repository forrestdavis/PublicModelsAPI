from byt5_model import Byt5Model

if __name__ == '__main__':

    model = Byt5Model("google/byt5-small")
    text = ['他们上个月到底勾结了什么?','什么他们上个月到底勾结了?']
    print(model.get_by_sentence_perplexity(text))

