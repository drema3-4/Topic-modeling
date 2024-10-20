!pip install bigartm10
import artm
from nltk import ngrams

news = pd.read_excel('prepeared_news.xlsx')

# Датасет с результатами моделирования
columns = ['model', 'num_topics', 'num_collection_passes', 'num_doc_passes', 'n-grams', 'perplexity', 'phi_sparsity', 'theta_sparsity']
results = pd.DataFrame(columns=columns)

# Функция для вычисления частоты слов
def calc_words_frequency(data: pd.DataFrame) -> dict:
  words_frequency = dict()

  for string in range(data.shape[0]):
    words = data.iloc[string, 'title'].split(' ') + data.iloc[string, 'content'].split(' ')

    for word in words:
      if word in words_frequency.keys:
        words_frequency[word] += 1
      else:
        words_frequency[word] = 1

  return words_frequency

# Функция создания vowpal_wabbit файла (каждая новость - отдельный документ)
def make_vowpal_wabbit(data: pd.DataFrame, path: str, words_frequency: dict) -> None:
    f = open(path, 'w')

    for string in range(data.shape[0]):
      words = data.iloc[string, 'title'].split(' ') + data.iloc[string, 'content'].split(' ')

      string_ = ''
      for word in words:
        if words_frequency[word] > 4:
          string_ += word + ' '
      string_ = string_[:-1]

      if len(string_) > 4:
        f.write('doc_{0} '.format(string) + string_ + '\n')

    f.close()

# Функция создания vowpal_wabbit файла с биграммами (каждая новость - отдельный документ)
def make_vowpal_wabbit_bigramm(data: pd.DataFrame, path: str) -> None:
    f = open(path, 'w')

    for string in range(data.shape[0]):
      words = data.iloc[string, 'title'].split(' ') + data.iloc[string, 'content'].split(' ')

      for_paste = ''
      for word in words:
        if words_frequency[word] > 4:
          for_paste += word + ' '
      for_paste = for_paste[:-1]

      if len(for_paste) > 0:
        f.write('doc_{0} '.format(string) + ' '.join(['_'.join(x) for x in list(ngrams(for_paste.split(' '), 2))]) + '\n')

    f.close()

# Функция создания vowpal_wabbit файла с триграммами (каждая новость - отдельный документ)
def make_vowpal_wabbit_trigramm(data: pd.DataFrame, path: str) -> None:
    f = open(path, 'w')

    for string in range(data.shape[0]):
      words = data.iloc[string, 'title'].split(' ') + data.iloc[string, 'content'].split(' ')

      for_paste = ''
      for word in words:
        if words_frequency[word] > 4:
          for_paste += word + ' '
      for_paste = for_paste[:-1]

      if len(for_paste) > 0:
        f.write('doc_{0} '.format(string) + ' '.join(['_'.join(x) for x in list(ngrams(for_paste.split(' '), 3))]) + '\n')

    f.close()

# Создание vowpal_wabbit файлов
make_vowpal_wabbit(news, './vw.txt', calc_words_frequency(pd.read_excel('prepeared_news.xlsx')))
make_vowpal_wabbit_bigramm(news, './vw2.txt', calc_words_frequency(pd.read_excel('prepeared_news.xlsx')))
make_vowpal_wabbit_trigramm(news, './vw3.txt', calc_words_frequency(pd.read_excel('prepeared_news.xlsx')))

# Создание батчей
bv = artm.BatchVectorizer(data_path='vw.txt', data_format='vowpal_wabbit', batch_size=3000, target_folder='SPARSE_batches')
bv2 = artm.BatchVectorizer(data_path='vw2.txt', data_format='vowpal_wabbit', batch_size=3000, target_folder='SPARSE_batches2')
bv3 = artm.BatchVectorizer(data_path='vw3.txt', data_format='vowpal_wabbit', batch_size=3000, target_folder='SPARSE_batches3')

# Функция создания и обучения модели
def make_and_train_SPARSE(num_topics: list[int], num_collection_passes: list[int], num_doc_passes: list[int], tau: list[int]):
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
make_and_train_SPARSE([8], [24], [7], [0.5, 1.0, 1.5, 2.0])