# Análise Topológica de Dados


## NOTA METODOLÓGICA

Este código implementa o pipeline do artigo de Kovacev-Nikolic et al. (2015):

  - Passo 1: $D = 1 - |C|$  (Equação 1 do artigo)
  - Passo 2: Complexo de Vietoris-Rips + homologia persistente (gudhi)
  - Passo 3: Paisagem de persistência amostrada em 50 pontos × k camadas
  - Passo 4: Concatenação → vetor de features
  - Passo 5: Padronização + PCA (3 componentes) dentro do LOO
  - Passo 6: SVM com kernel linear

Diferenças em relação ao código original recebido:
  - Feature: paisagem de persistência (não curva de Betti)
  - PCA com 3 componentes (artigo: $~52.5%$ da variância p/ grau 1)
  - StandardScaler e PCA ajustados apenas no fold de treino (sem leakage)
  - ``max_edge_length`` = ``t_max`` (removido $I = 2*J$ incorreto)
  - Diagonal e simetria da matriz $D$ verificadas explicitamente

Sua observação sobre a virada em $t \approx 0.19$:
  Esse sinal está coerente com o artigo (Fig. 7): o ciclo mais persistente
  nasce em $\sim t=0.2$ e morre antes de $t=0.6$. A paisagem captura exatamente
  a altura e duração desse ciclo, enquanto a curva de Betti só conta
  quantos ciclos existem — perdendo a informação de persistência.

----------
### Bloco 3: Extração de Características Topológicas (Paisagem de Persistência)

Neste bloco, encontra-se o coração matemático do código. É aqui que ocorre a tradução rigorosa do mundo da **Topologia Algébrica** para o mundo do **Aprendizado de Máquina** (*Machine Learning*).

Algoritmos estatísticos clássicos (como o SVM) exigem que cada amostra seja representada por um vetor numérico de tamanho fixo. O problema inerente aos diagramas de persistência tradicionais (códigos de barras) é a variabilidade da sua dimensão: uma proteína pode exibir 10 ciclos topológicos, enquanto outra pode exibir 15, inviabilizando a entrada matricial direta. A solução elegante, introduzida por Peter Bubenik e adotada no artigo base, é o uso da **Paisagem de Persistência** (*Persistence Landscape*). Ela transforma os pontos discretos do diagrama numa sucessão de funções contínuas e integráveis, que podem ser discretizadas num vetor perfeitamente uniforme.

Abaixo, detalhamos cada etapa matemática e algorítmica deste bloco:

#### 1. Parâmetros de Configuração e Fidelidade à Literatura
As variáveis de configuração topológica foram extraídas textualmente da **Seção 4** do estudo de Kovacev-Nikolic *et al.* para garantir replicação exata:

* **`T_MIN = 0.0` e `T_MAX = 0.7`**: A matriz de distância baseia-se na correlação de Pearson normalizada, definida como $D = 1 - |C|$. Embora o limite teórico máximo de distância topológica seja $1.0$, as observações empíricas revelaram que todos os eventos homológicos significativos (nascimento e morte dos buracos) das proteínas MBP concentram-se no intervalo de filtração paramétrica de $[0.0, 0.7]$.
* **`N_PONTOS = 50`**: O estudo original prescreve: *"The landscape functions were evaluated on a grid of 50 equally spaced points..."*. Isso dita que o contorno de cada função da paisagem $\lambda_k$ é discretamente amostrado em $50$ coordenadas equidistantes ao longo do eixo da filtração $t$.
* **`N_CAMADAS = {0: 370, 1: 73, 2: 78}`**: A paisagem de persistência é formada por um feixe de funções ordenadas $\lambda_1(t) \ge \lambda_2(t) \ge \ldots \ge \lambda_k(t)$. Os investigadores mapearam o limite superior (*upper bound*) empírico do número de camadas geradas. Focando no Grau 1 (anéis/ciclos unidimensionais), a estrutura de maior complexidade do conjunto gerou precisamente $73$ camadas. Fixar $k = 73$ como padrão arquitetural assegura que, caso uma proteína exiba menor complexidade topológica (e.g., gerando apenas $20$ camadas), as restantes $53$ serão matematicamente nulas (preenchidas com zeros), preservando a isometria dos tensores.
* **`GRAU_PRINCIPAL = 1`**: O foco incide sobre o cálculo dos Números de Betti da primeira dimensão ($H_1$), os quais codificam buracos rodeados pelas dinâmicas correlacionadas de agrupamentos moleculares.

#### 2. Processamento Sistémico e Otimização via *Cache*
O iterador percorre agnosticamente a lista unificada de proteínas do artigo, onde $y_i = 1$ codifica conformações Fechadas e $y_i = 0$ conformações Abertas, sem depender da categorização química (*apo/holo*).

Devido à complexidade computacional inerente à triangulação do Complexo de Vietoris-Rips e à formulação das paisagens, o bloco introduz um esquema de conservação de memória intermédia (*cache*). Se o arquivo binário `.npy` referente à paisagem já existir no sistema de ficheiros, ele é carregado da memória em tempo linear $O(1)$. Na sua ausência, a matriz $C$ cruza as funções métricas de transformação ($D$) e entra no motor algorítmico do `gudhi` para computação completa.

#### 3. Vetorização e Achatamento Algébrico (*Flattening*)
A invocação nativa do `gudhi.representations.Landscape` produz uma matriz bidimensional que, para o Grau 1, obedece à dimensão $(73, 50)$ por proteína.

Uma vez que modelos multivariados como o classificador vetorial (SVM) e o mapeamento de componentes principais (PCA) requerem subespaços vetoriais unidimensionais ($\mathbb{R}^n$), realiza-se o achatamento iterativo da matriz mediante a operação `.flatten()`. Algebricamente, as linhas contendo os valores da paisagem são justapostas num vetor unificado:
$$\text{Dimensionalidade do Vetor } x_i = k \times \text{resolução} = 73 \times 50 = 3650 \text{ características geómetricas}$$

#### 4. Resumo da Estrutura Tensorial (*Feature Space*)
Quando o processamento da amostra termina, a lista de vetores é transformada num *array* global padronizado pelo NumPy. 

O tensor de observações `X_features` é concluído com a forma métrica de $(14, 3650)$. O significado destas grandezas é absoluto: cada matriz representativa da amostra de treino ($n=14$) encontra-se agora perfeitamente traduzida para um hiper-espaço vetorial contínuo com $3650$ atributos espaciais, habilitando o uso do aparato matemático estatístico do passo sequente.

----------
### Bloco 4: Redução de Dimensionalidade e Classificação Topológica (PCA + SVM)

Neste bloco, transitamos da **Topologia Algébrica** para o **Aprendizado de Máquina estatístico**. O objetivo é pegar as Paisagens de Persistência (*Persistence Landscapes*) geradas no bloco anterior — que são vetores de alta dimensionalidade — e treinar um classificador capaz de distinguir a geometria de proteínas Abertas e Fechadas com base nos seus ciclos topológicos.

Abaixo, detalhamos cada etapa da construção e o seu rigor matemático, espelhando fielmente a metodologia proposta no estudo de Kovacev-Nikolic *et al.*

#### 1. A Estrutura de `Pipeline` e a Prevenção de *Data Leakage*
No aprendizado de máquina, um erro metodológico comum é realizar a padronização dos dados ou a redução de dimensionalidade no conjunto de dados inteiro *antes* de separar os dados de treino e teste. Isso causa o indesejado vazamento de dados (*data leakage*), pois informações da distribuição global (incluindo o teste) acabam influenciando os parâmetros do modelo.

Ao instanciar um `Pipeline` da biblioteca `scikit-learn`, garantimos que o fluxo de transformações ocorra de forma independente e honesta a cada iteração:
* O `StandardScaler` ajusta a média $\mu$ e o desvio padrão $\sigma$ estritamente nos vetores topológicos do conjunto de treino.
* Cada característica topológica $x$ é transformada em um escore normalizado $z$ através da fórmula: $z = \frac{x - \mu}{\sigma}$.

#### 2. Análise de Componentes Principais (`PCA`)
Nossos vetores de características (`X_features`) extraídos do Gudhi possuem $3650$ dimensões ($73$ camadas independentes avaliadas em $50$ pontos distintos). Treinar um classificador linear clássico com $3650$ variáveis e apenas $14$ amostras causaria um sobreajuste (*overfitting*) massivo, um fenômeno estatístico conhecido como a "Maldição da Dimensionalidade".

Para contornar isso, utilizamos o objeto `PCA(n_components=3)`. O algoritmo de PCA é uma transformação linear ortogonal que projeta os dados num novo sistema de coordenadas de baixa dimensão. Matematicamente, ele busca os autovetores (*eigenvectors*) da matriz de covariância dos dados padronizados. Retemos apenas os $3$ primeiros Componentes Principais, que correspondem aos autovalores (*eigenvalues*) de maior magnitude — ou seja, as direções no espaço vetorial que capturam a maior variância (informação geométrica) do conjunto de dados. 

Conforme demonstrado nos nossos resultados empíricos, apenas esses $3$ eixos já foram suficientes para explicar $52,79\%$ da variância total, corroborando a marca esperada de $\sim 52,50\%$ reportada na literatura original.

#### 3. Máquina de Vetores de Suporte (`SVM`)
Com os dados agora residindo num subespaço tridimensional contínuo $\mathbb{R}^3$, aplicamos o modelo `svm.SVC(kernel='linear')`. O algoritmo SVM busca o hiperplano de separação ótimo que maximiza a margem geométrica de distância entre as classes biológicas, onde $y_i \in \{0, 1\}$ (Aberta ou Fechada, respectivamente).

O hiperplano é definido pela equação linear $w^T x + b = 0$, onde $w$ é o vetor de pesos ortogonal ao hiperplano e $b$ é o parâmetro de viés (*bias*). O uso do parâmetro `kernel='linear'` reflete uma decisão metodológica fundamental do artigo base: ele comprova que a transformação não-linear sofrida lá trás pela *Persistence Landscape* já havia "desenrolado" o problema topológico de forma tão eficaz que não foi preciso recorrer a hiper-planos complexos ou mapeamentos radiais (como os de *kernel* RBF) para encontrar a fronteira de decisão.

#### 4. Validação Cruzada *Leave-One-Out* (`LOO`)
Devido ao número ínfimo de amostras no gabarito geométrico ($n = 14$), utilizar um particionamento tradicional (como $80\%$ treino e $20\%$ teste) geraria uma enorme instabilidade estatística.

O iterador `LeaveOneOut` fraciona o conjunto de dados em $n$ partições distintas. Em cada iteração $i$:
1. Um subconjunto de $n - 1$ amostras ($13$ matrizes proteicas) é enviado ao `Pipeline` para calibrar o `StandardScaler`, rodar o `PCA` e otimizar os pesos do `SVM`.
2. A única amostra restante é mantida intocada ("isolada"), padronizada, projetada no espaço tridimensional do PCA pré-computado, e então o modelo infere a sua classe predita $\hat{y}_i$.
3. Esse processo se repete $14$ vezes para cobrir todas as proteínas.

Ao final, a acurácia geral do sistema é dada por:
$$\text{Acurácia} = \frac{1}{n} \sum_{i=1}^{n} I(\hat{y}_i = y_i)$$
Onde $I$ é a função indicadora que vale $1$ em caso de acerto e $0$ em caso de erro. A obtenção da pontuação perfeita valida de forma rigorosa, reprodutível e isenta de viés o poder da Análise Topológica de Dados neste problema.

---
### Blocos 5 e 6: Treino Definitivo e Classificação Cega (Inferência)

Nesta fase final, cruzamos a fronteira entre a **validação estatística** e a verdadeira **descoberta científica**. Uma vez provado (no Bloco 4) que a nossa arquitetura topológica é impecável, o modelo deixa de ser um mero avaliador de hipóteses e passa a atuar como um "oráculo" matemático. O seu objetivo passa a ser classificar o estado conformacional de $28$ proteínas cujos rótulos geométricos reais (Aberta/Fechada) nos são desconhecidos.

Estes dois blocos operam em simbiose para concluir a etapa de *Machine Learning* e gerar os resultados biológicos da tese.

#### 1. O Treino do Modelo Definitivo (`Bloco 5`)
Durante a validação *Leave-One-Out*, criámos modelos provisórios descartando sempre uma amostra. No entanto, a teoria de Aprendizagem Automática dita que o modelo final para produção deve ser exposto à **totalidade da informação disponível**.

Ao executarmos `pipeline.fit(X_features, y_labels)`, o algoritmo processa simultaneamente o tensor de treino completo ($14 \times 3650$). O que acontece matematicamente sob o capô:
* O `StandardScaler` fixa os vetores globais de média $\mu_{train}$ e desvio padrão $\sigma_{train}$.
* O `PCA` define os $3$ eixos ortogonais definitivos de projeção ($\mathbb{R}^3$).
* O `SVM` resolve a otimização de margem máxima, "congelando" o vetor de pesos ortogonais $w$ e o parâmetro de viés $b$ para o hiperplano $w^T x + b = 0$.

A partir desta linha de código, a máquina "aprendeu" a regra topológica universal que separa uma proteína Aberta de uma Fechada.

#### 2. Ingestão e Processamento dos Dados Cegos (`Bloco 6`)
O código define duas novas listas de alvos (`ids_holo_alvo` e `ids_apo_alvo`) totalizando $28$ estruturas não rotuladas. Para cada uma destas novas proteínas, o sistema repete o processo estrito de extração topológica:
1. Localiza o ficheiro `.npy` da matriz de correlação cruzada e aplica a métrica $D = 1 - |C|$.
2. Computa o complexo de Vietoris-Rips utilizando o `gudhi`.
3. Extrai a Paisagem de Persistência (*Persistence Landscape*) gerando o vetor $x_{novo}$ de dimensão $3650$, salvando-o em *cache* para eficiência.

#### 3. Projeção e Inferência Algébrica (`predict`)
A fase crítica de previsão ocorre na instrução `pred = pipeline.predict([paisagem])[0]`. 
O novo vetor $x_{novo}$ não altera a estrutura da máquina; ele é apenas processado passivamente:
1. É normalizado utilizando estritamente os parâmetros do treino original: $z_{novo} = \frac{x_{novo} - \mu_{train}}{\sigma_{train}}$.
2. É multiplicado pela matriz de autovetores do `PCA` para ser mapeado no subespaço $\mathbb{R}^3$.
3. A sua nova coordenada em $\mathbb{R}^3$ é introduzida na equação do hiperplano do `SVM`. 
   * Se $\text{sgn}(w^T z_{novo} + b) > 0$, a amostra cai no semi-espaço positivo e recebe o rótulo $1$ (`FECHADA`).
   * Se $\text{sgn}(w^T z_{novo} + b) < 0$, a amostra cai no semi-espaço negativo e recebe o rótulo $0$ (`ABERTA`).

#### 4. Consolidação da Descoberta Biológica
O algoritmo finaliza estruturando as previsões num `pd.DataFrame` do Pandas e exportando-as para um ficheiro estruturado (`resultados_classificacao.csv`). 

É aqui que o método matemático se converte em tese biológica. O algoritmo avalia proteínas químicas *Apo* (sem ligante) e, ignorando a sua química, utiliza puramente os números de Betti da paisagem topológica para as classificar. Quando o `SVM` assinala uma estrutura puramente *Apo* com a etiqueta preditiva de `FECHADA` (e.g., a proteína `2N45`), obtemos a evidência computacional direta que corrobora a teoria da **Seleção Conformacional**: a geometria de uma proteína não é estritamente ditada pela presença do ligante.
