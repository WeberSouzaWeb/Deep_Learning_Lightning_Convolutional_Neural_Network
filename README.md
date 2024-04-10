# Deep_Learning_Lightning_Convolutional_Neural_Network
CNNs: Inception, ResNet and DenseNet

Redes Neurais Convolucionais (CNN) são um tipo de modelo de aprendizado profundo utilizado para o processamento de dados com uma estrutura espacial, como imagens e vídeos. Elas são inspiradas na organização do córtex visual do cérebro humano e são capazes de aprender representações hierárquicas de características visuais.
A arquitetura de uma CNN é composta por camadas convolucionais, camadas de pooling e camadas totalmente conectadas. As camadas convolucionais são responsáveis por extrair características das imagens através da aplicação de filtros convoluionais, que identificam padrões como bordas, formas e texturas; As camadas de pooling reduzem a dimensionalidade das características extraídas, mantendo as informações relevantes e descartando as redundâncias; As camadas totalmente conectadas utilizam as características extraídas para fazer a classificação final das imagens. A figura abaixo ilustra a representação geral de uma CNN:

![image](https://github.com/WeberSouzaWeb/Deep_Learning_Lightning_Convolutional_Neural_Network/assets/107212929/8843294e-21e0-4419-b87c-8306e9c42634)

As CNNs são compostas por várias camadas, cada uma das quais realiza uma transformação nos dados de entrada.


![image](https://github.com/WeberSouzaWeb/Deep_Learning_Lightning_Convolutional_Neural_Network/assets/107212929/fa9b5407-6c6a-44c3-9f83-40a6253fd46a)

### Camada de Convolução
  A camada de convolução é a principal camada de uma CNN. Ela aplica filtros convolucionais aos dados de entrada para extrair características (features) relevantes. Cada filtro convulucional é uma matriz de pesos que é aplixada a cada região da entrada para produzir uma saída correspondente. A saída da camada de convolução é uma representação em duas dimensões das características extraídas.

### Camada de Ativação
  A camada de ativação segue a camada de convolução e aplica uma função de ativação à saída da camada de convolução. Isso introduz não linearidade na rede neural, permitindo que a CNN aprenda relações mais complexas entre as características.

### Camada de Pooling
  A camada de pooling reduz a dimensionalidade da saída da camada de convolução. Isso é feito aplicando uma operação de redução, como o max-pooling ou o average-pooling, em cada região da saída. Isso reduz o número de parâmetros na rede neural e ajuda a evitar o overfitting.

### Camada Totalmente Conectada
  A camada totalmente conectada é uma camada tradicional de rede neural que recebe a saída da camada de pooling e a transforma em uma saída final. Essa camada geralmente é usada no final da rede para produzir uma classificação ou regressão.

  Além dessas camadas, as CNNs também podem incluir camadas de normalização, camadas de dropout e camadas de regularização para evitar overfitting. As CNNs são treinadas usando algoritmos de otimização, como o gradiente descendente, para minimizar a função de perda entre a saída da rede e os rótulos de treinamento.


# O Que é Convolução?
Convolução é uma operação matemática fundamental realizada em uma rede neural convolucional (CNN - Convolutional Neural Network). Na CNN, a convolução é aplicada em camadas específicas da rede, nas quais a entrada é convoluída com um conjunto de filtros para extrair recursos (features) relevantes. 
A convolução é uma operação matemática que envolve a multiplicação de um pequeno filtro (kernel) em cada posição da entrada. O filtro é deslocado em etapas, geralmente chemadas de passos (stride), e o resultado da multiplicação é acumulado em uma matriz conhecida como mapa de características (feature map). O mapa de características é então passado para a próxima camada da rede.

A convolução permite que a rede aprenda a detectar recursos em diferentes partes da entrada, como bordas, texturas, formas e padrões. Além disso, a aplicação de várias camadas de convolução em um rede pode permitir que ela extraia recursos cada vez mais complexos e abstratos, como objetos e cenas.

Em resumo, a convolução é uma operaçãp matemática essencial em uma CNN, pois permite que a rede extraia recursos relevantes e aprenda a realizar tarefas como classificação de imagem, detecção de objetos, segmentação de imagem, entre outras.

![image](https://github.com/WeberSouzaWeb/Deep_Learning_Lightning_Convolutional_Neural_Network/assets/107212929/c4b10081-4734-4660-ac13-d47104ec099f)

A matemática da operação de convolução pode ser descrita como a sobreposição de um filtro (kernel) em uma matriz de entrada. O filtro é uma matriz menos que a matriz de entrada e geralmente é definido manualmente ou aprendido durante o treinamento da rede.

  A operação de convolução é aplicada em todas as posições da matriz de entrada para gerar o mapa de características resultante. O tamanho do mapa de características pode ser controlado pelo tamanho do filtro, pelo número de canais de entrada e pelo número de filtros aplicados. 

  Além disso, a operação de convolução pode ser estendida para trabalhar com várias camadas de entrada e vários canais de saída. Nesse caso, cada canal de saída é gerado pela convolução de um conjunto de filtros com os canais de entrada correspondentes, e os resultados são somados para formar o mapa de caractéristicas final. Uma das mais avançadas arquiteturas de Deep Learning é, de fato, um conjunto de operações com matrizes.

  ## Max Pooling / Downsampling
  	Max pooling é uma técnica de downsampling (redução de dimensionalidade) comumente usada em redes neurais convolucionais (CNNs) para reduzir o tamanho dos mapas de características resultantes das camadas de convolução. O objetivo do max pooling é extrair informações importantes dos mapas de características, mantendo a robustez da rede a pequenas variações nas posições dos recursos detectados.
     A operação de max pooling é aplicada em cada mapa de características gerado pela convolução, dividindo a matriz em regiões e mantendo apenas o calor máximo de cada região. A região de pooling é geralmente definida por um filtro com um tamanho fixo e um passo (stride) correspondente. Durante a operação de max pooling, o filtro é aplicado à matriz de entrada, e o valor máximo em cada região é retido, enquanto os demais valores são descartados.

     ![image](https://github.com/WeberSouzaWeb/Deep_Learning_Lightning_Convolutional_Neural_Network/assets/107212929/43f60607-3c94-47b2-8735-04b8552eab24)

     O Max Pooling é util porque reduz a dimensão espacial do mapa de características, o que reduz a quantidade de parâmetros na rede e evita overfitting. Além disso, o maz pooling ajuda a tornar a rede mais invariante a pequenas variações dos recursos, o que pode melhorar o desempenho da rede em tarefas como detecção de objetos.

# Treinamento e Backpropagation
  O treinamento de uma rede neural convolucional (CNN) envolve o processo de ajustar os pesos e os viéses da rede para minimizar uma função de perda (loss) em um conjunto de dados de treinamento. O processo de ajuste dos pesos e dos viéses é realizado usando um algoritmo de otimização, como o gradiente descendente (gradient descent) ou suas variantes.
  O backpropagation é um algoritmo fundamental usado em CNNs para calcular os gradientes dos pesos e dos viéses da rede em relação à função de perda. O backpropagation é realizado por meio de uma propagação reversa dos erros a partir da camada de saída até a camada de entrada.
  O processo de treinamento e backpropagation em CNNs pode ser resumido em cinco etapas:
  1. Inicialização dos pesos e viéses: Os pesos e os viéses são inicializados aleatoriamente com uma distribuição gaussiana ou uniforme.
  2. Propagação direta (forward pass): a entrada é passada pela rede em direção à saída, utilizando as operações de convolução, max pooling e funções de ativaçãopara gerar uma previsão da saída.
  3. Cálculo da função de perda (loss function): a função de perda é calculada entre a previsão da saída e o valor real da saída
  4. Backpropagation: o erro é propagado da camada de saída até a camada de entrada da rede, calculando os gradientes dos pesos e dos viéses em relação à função de perda
  5. Atualização dos pesos e viéses: os pessos e os viéses são atualizados utilizando um algoritmo de otimização, como o gradiente descendente, que usa os gradientes calculados no passo 4 para ajustar os valores dos pesos e dos viéses.

Esse processo é repetido iterativamente para minimizar a função de perda no conjunto de dados de treinamento. A CNN é avaliada periodicamente em conjunto de dados de validação para verificar se a rede está generalizando bem.

O link traz a Matemática completa do Backpropagation: https://www.deeplearningbook.com.br/algoritmo-backpropagation-parte-2-treinamento-de-redes-neurais/

