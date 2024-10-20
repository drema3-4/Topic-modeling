!pip install bigartm10
import artm
from nltk import ngrams

news = pd.read_excel('prepeared_news.xlsx')

# Датасет с результатами моделирования
columns = ['model', 'num_topics', 'num_collection_passes', 'num_doc_passes', 'tau', 'n-grams', 'perplexity', 'phi_sparsity', 'theta_sparsity']
results = pd.DataFrame(columns=columns)

# Функция для вычисления частоты слов
def calc_words_frequency(data: pd.DataFrame) -> dict:
    words_frequency = {}

    for string in range(data.shape[0]):
      if type(data.loc[string, 'title']) == str:
          for word in nltk.word_tokenize(data.loc[string, 'title']):
              if word in words_frequency.keys():
                  words_frequency[word] += 1
              else:
                  words_frequency[word] = 1

      if type(data.loc[string, 'content']) == str:
          for word in nltk.word_tokenize(data.loc[string, 'content']):
              if word in words_frequency.keys():
                  words_frequency[word] += 1
              else:
                  words_frequency[word] = 1

    return words_frequency

# Функция создания vowpal_wabbit файла (каждая новость - отдельный документ)
def make_vowpal_wabbit(data: pd.DataFrame, path: str, words_frequency: dict) -> None:
    f = open(path, 'w')

    for string in range(data.shape[0]):
      words = []
      if type(data.loc[string, 'title']) == str:
        words += nltk.word_tokenize(data.loc[string, 'title'])

      if type(data.loc[string, 'content']) == str:
        words += nltk.word_tokenize(data.loc[string, 'content'])

      string_ = ''
      for word in words:
        if word in words_frequency.keys():
          if words_frequency[word] > 4:
            string_ += word + ' '

      if len(string_) > 4:
        string_ = string_[:-1]
        f.write('doc_{0} '.format(string) + string_ + '\n')

    f.close()

# Функция создания vowpal_wabbit файла с биграммами (каждая новость - отдельный документ)
def make_vowpal_wabbit_bigramm(data: pd.DataFrame, path: str, words_frequency: dict) -> None:
    f = open(path, 'w')

    for string in range(data.shape[0]):
      words = []
      if type(data.loc[string, 'title']) == str:
        words += nltk.word_tokenize(data.loc[string, 'title'])

      if type(data.loc[string, 'content']) == str:
        words += nltk.word_tokenize(data.loc[string, 'content'])

      string_ = ''
      for word in words:
        if word in words_frequency.keys():
          if words_frequency[word] > 4:
            string_ += word + ' '

      if len(string_) > 0:
        string_ = string_[:-1]
        f.write('doc_{0} '.format(string) + ' '.join(['_'.join(x) for x in list(ngrams(string_.split(' '), 2))]) + '\n')

    f.close()

# Функция создания vowpal_wabbit файла с триграммами (каждая новость - отдельный документ)
def make_vowpal_wabbit_trigramm(data: pd.DataFrame, path: str, words_frequency: dict) -> None:
    f = open(path, 'w')

    for string in range(data.shape[0]):
      words = []
      if type(data.loc[string, 'title']) == str:
        words += nltk.word_tokenize(data.loc[string, 'title'])

      if type(data.loc[string, 'content']) == str:
        words += nltk.word_tokenize(data.loc[string, 'content'])

      string_ = ''
      for word in words:
        if word in words_frequency.keys():
          if words_frequency[word] > 4:
            string_ += word + ' '

      if len(string_) > 0:
        string_ = string_[:-1]
        f.write('doc_{0} '.format(string) + ' '.join(['_'.join(x) for x in list(ngrams(string_.split(' '), 3))]) + '\n')

    f.close()

# Создание vowpal_wabbit файлов
make_vowpal_wabbit(news, './vw.txt', calc_words_frequency(pd.read_excel('prepeared_news.xlsx')))
make_vowpal_wabbit_bigramm(news, './vw2.txt', calc_words_frequency(pd.read_excel('prepeared_news.xlsx')))
make_vowpal_wabbit_trigramm(news, './vw3.txt', calc_words_frequency(pd.read_excel('prepeared_news.xlsx')))

# Создание батчей
bv = artm.BatchVectorizer(data_path='vw.txt', data_format='vowpal_wabbit', batch_size=3000, target_folder='DECOR_batches')
bv2 = artm.BatchVectorizer(data_path='vw2.txt', data_format='vowpal_wabbit', batch_size=3000, target_folder='DECOR_batches2')
bv3 = artm.BatchVectorizer(data_path='vw3.txt', data_format='vowpal_wabbit', batch_size=3000, target_folder='DECOR_batches3')

# Функция создания и обучения модели
def make_and_train_DECOR(num_topics: list[int], num_collection_passes: list[int], num_doc_passes: list[int], tau: list[int]):
  for param1 in num_topics:
    for param2 in num_collection_passes:
      for param3 in num_doc_passes:
        for param4 in tau:
          for param5 in range(1, 3+1):
            global model
            if param5 == 1:
              model = artm.ARTM(num_topics=param1, num_document_passes=param3, dictionary=bv.dictionary, class_ids={'@default_class': 1.0})
            elif param5 == 2:
              model = artm.ARTM(num_topics=param1, num_document_passes=param3, dictionary=bv2.dictionary, class_ids={'@default_class': 1.0})
            else:
              model = artm.ARTM(num_topics=param1, num_document_passes=param3, dictionary=bv3.dictionary, class_ids={'@default_class': 1.0})

            model.regularizers.add(artm.SmoothSparsePhiRegularizer(name='phi-sparse',tau=param4))
            model.regularizers.add(artm.SmoothSparseThetaRegularizer(name='theta-sparse',tau=param4))

            model.scores.add(artm.PerplexityScore(name='perplexity', dictionary=bv.dictionary))
            model.scores.add(artm.SparsityPhiScore(name='sparsity_phi_score'))
            model.scores.add(artm.SparsityThetaScore(name='sparsity_theta_score'))
            model.scores.add(artm.TopTokensScore(name='top-tokens', num_tokens=10))

            for _ in range(param2):
              if param5 == 1:
                model.fit_offline(bv, num_collection_passes=1)
              elif param5 == 2:
                model.fit_offline(bv2, num_collection_passes=1)
              else:
                model.fit_offline(bv3, num_collection_passes=1)

            results.loc[ len(results.index) ] = [ 'LDA', param1, param2, param3, param4, '{0}-gramm'.format(param5),
                                                  model.score_tracker['perplexity'].last_value,
                                                  model.score_tracker['sparsity_phi_score'].last_value,
                                                  model.score_tracker['sparsity_theta_score'].last_value ]
          
# Создаём и обучаем модель
make_and_train_DECOR([8], [24], [7], [1e6, 2e6, 1e7, 2e7])