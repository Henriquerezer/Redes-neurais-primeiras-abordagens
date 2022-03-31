# Ajuste de Parâmetros

O ajuste é o processo de maximizar o desempenho de um modelo sem overfitting ou criar uma variação muito alta. No aprendizado de máquina, isso é feito selecionando “hiperparâmetros” apropriados.

Os hiperparâmetros podem ser vistos como os “mostradores” ou “botões” de um modelo de aprendizado de máquina . Escolher um conjunto apropriado de hiperparâmetros é crucial para a precisão do modelo, mas pode ser computacionalmente desafiador. Os hiperparâmetros diferem de outros parâmetros do modelo, pois não são aprendidos pelo modelo automaticamente por meio de métodos de treinamento. Em vez disso, esses parâmetros devem ser definidos manualmente. Existem muitos métodos para selecionar os hiperparâmetros apropriados.
alguns exemplos de abordagens 
- Gridsearch
- Random Search
- Bayesian Optimization

Nesta abordagem foi utilizado Gridsearch.

O Grid Search, também conhecido como varredura de parâmetros, é um dos métodos mais básicos e tradicionais de otimização hiperparamétrica. Esse método envolve definir manualmente um subconjunto do espaço hiperparamétrico e esgotar todas as combinações dos subconjuntos de hiperparâmetros especificados. O desempenho de cada combinação é então avaliado, normalmente usando validação cruzada, e a combinação hiperparamétrica com melhor desempenho é escolhida.

A pesquisa em grade examinará cada pareamento de α e β para determinar a combinação de melhor desempenho. Os pares resultantes,  H , são simplesmente cada saída que resulta da obtenção do produto cartesiano de α e β. Embora simples, essa abordagem de “força bruta” para otimização de hiperparâmetros tem algumas desvantagens. Espaços hiperparamétricos de dimensões mais altas são muito mais demorados para testar do que o simples problema bidimensional apresentado aqui. Além disso, como sempre haverá um número fixo de amostras de treinamento para qualquer modelo, o poder preditivo do modelo diminuirá à medida que o número de dimensões aumentar. Isso é conhecido como fenômeno de Hughes

## Melhores parâmetros encontrados, apartir dos hiperparâmetros definidos.

<div align="center">
<img src="https://user-images.githubusercontent.com/87787728/161048109-140f566d-9f66-47b2-bc3d-056d195e2750.png" width="700px" />
</div>

### _O código está disponível na pasta, lembre-se que por se tratar de um gridsearch com duas definições de epochs e duas definições de batch_size, o código demorará algumas horas completar._
