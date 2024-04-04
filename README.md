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
