Regressão Valor de um veículo

kaggle database -> https://www.kaggle.com/datasets/viannaandresouza/autosdataset

Primeiramente foi realizado o tratamento dos dados, foi utilizado a função Dropout, disponível em https://keras.io/api/layers/regularization_layers/dropout/ , foi retirado colunas que não seriam de interesse para a regressão do valor de um veículo.

Após foi analisado os valores inconsistentes, como por exemplo valores referentes ao preço do veículo. Foi analasido os valores abaixo de 100 e valores acima de 350000, devido a possíveis erros de digitação ao preeencher o dataset, já que é pouco provavel que o valor de um veículo seja menor que 100 e maior que 350000.

Para que seja possível realizar a regressão, foi analisado os dados faltantes, foi analisado por meio de iloc e value_counts(), se a coluna possuia valores faltantes, e esses valores faltantes foram substituidos pelo valor mais comum daquela tabela. Foi utilizado Labelencoder e OneHotEnconder afim de transformar os dados de variaveis com letras para variaveis numéricas.

Após foi realizada uma regressão simples, com apenas duas camadas densas e uma camada de saída, e analisada a LOSS e ACCURACY do modelo, lembrando que o modelo está treinando e testadndo com a mesma base de dados, o que ira nos avaliar as metricas apenas do treinamento. 

Por fim foi realizado um novo arquivo, utilizando agora validação cruzada, foi utilziada uma estrutura de rede neural semelhante com duas camadas ocultas e uma cada de saída, porém foram utilzidas 10 camadas de validação cruzada.

