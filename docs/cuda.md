# Treinamento CUDA

O objetivo desta etapa foi levar o treinamento do Barenn para a GPU sem mudar a maneira como a rede aprende. Continuamos com 51 épocas, as amostras na ordem do arquivo e uma atualização dos pesos depois de cada amostra. Não usamos embaralhamento nem mini-batches, porque isso mudaria o resultado que estamos tentando preservar.

A versão integrada executa **um kernel por época**. Dentro dele, um bloco CUDA percorre as amostras e divide os cálculos dos neurônios entre as threads. Os pesos ficam em memória compartilhada, e as barreiras evitam que uma etapa leia valores que ainda estão sendo alterados. No backward, o gradiente de entrada usa os pesos anteriores à atualização. Ao terminar a época, a GPU envia as perdas por amostra para a CPU, que faz a soma em `double` na mesma ordem da versão original.

**O que encontramos:** o programa acertou as mesmas **1.717 de 1.797** imagens e reproduziu as 51 perdas e a matriz de confusão. A comparação completa também encontrou **83.672 valores iguais** ao oráculo. A troca de um kernel por amostra para um por época diminuiu o tempo da própria versão CUDA, mas, neste problema pequeno, **a CPU continuou mais rápida**: mediana de 0,2687 s contra 1,670837 s na GPU. As medições e a diferença entre os nós usados estão documentadas abaixo.

Para comparar os resultados numéricos, usamos `--fmad=false` na compilação CUDA. Com FMA ativo, oito previsões diferiram da CPU no teste anterior; essa configuração continua fora do escopo validado. A captura do oráculo é opcional e não entra nos benchmarks.

## Compilação e execução no Franky

Solicite uma GPU conforme os limites atuais do Slurm e compile no nó alocado. Exemplo, a partir da raiz do projeto:

```bash
srun --partition=merry_gpu --gres=gpu:1 --cpus-per-task=1 --mem=2048M --time=00:04:30 --pty bash
module load gcc/14.2.0
module load cuda/12.8.1
scripts/compilar_cuda.sh ./barenn_cuda
./barenn_cuda
```

A execução padrão lê `optdigits.tra` e `optdigits.tes`, treina todas as amostras por 51 épocas e avalia o teste. Também é possível informar arquivos distintos e, opcionalmente, quantas amostras de treino e épocas executar:

```bash
./barenn_cuda optdigits.tra optdigits.tes 3823 51
```

A versão CUDA reporta perdas por época, acertos, perda de teste, matriz de confusão e tempos de treino, avaliação e total. O tempo total exclui a leitura dos arquivos, mas inclui preparação dos dados, alocações, transferências, treino, avaliação e liberações.

### Quando faltar `/usr/include` nos nós

Em nós do Franky onde os cabeçalhos do sistema não estão disponíveis, pode-se copiar os cabeçalhos do head-node para uma pasta pessoal temporária **antes** da alocação, desde que corresponda ao ambiente dos nós. Não versionar essa cópia:

```bash
mkdir -p "$HOME/cabecalhos-franky"
tar -C /usr -cf - include | tar -C "$HOME/cabecalhos-franky" -xf -
```

No shell do nó alocado, indique essa cópia ao compilar:

```bash
export BARENN_SYSTEM_INCLUDE="$HOME/cabecalhos-franky/include"
scripts/compilar_cuda.sh ./barenn_cuda
```

Esta é uma solução provisória; prefira uma instalação de cabeçalhos mantida pelo cluster quando disponível.

## Validação numérica

Gere uma captura CPU e uma CUDA em arquivos novos, no nó alocado, usando a mesma versão dos dados. No exemplo abaixo, `BARENN_SYSTEM_INCLUDE` é necessário apenas se os cabeçalhos estiverem ausentes no nó:

```bash
inc_flags=()
if [[ -n ${BARENN_SYSTEM_INCLUDE:-} ]]; then
  inc_flags=(-idirafter "$BARENN_SYSTEM_INCLUDE")
fi
g++ -std=c++17 -O3 -ffp-contract=off "${inc_flags[@]}" -DBARENN_ORACLE barenn.cpp -o barenn_cpu_oraculo
BARENN_CUDA_ORACLE=1 scripts/compilar_cuda.sh ./barenn_cuda_oraculo
BARENN_ORACLE_OUTPUT=cpu.tsv ./barenn_cpu_oraculo
BARENN_ORACLE_OUTPUT=cuda.tsv ./barenn_cuda_oraculo
python3 scripts/validar_oraculo.py cpu.tsv cuda.tsv --atol 0 --rtol 0
```

A comparação inclui perdas por época, parâmetros, valores intermediários selecionados e avaliação. Para outras configurações de compilação, registre as diferenças e justifique qualquer tolerância; previsões e contagens são comparadas exatamente.

O leitor, a perda e a captura do oráculo em `cuda/barenn_cuda_suporte.hpp` reproduzem os componentes da CPU. Essa duplicação localizada evita uma refatoração ampla da versão sequencial nesta entrega, mas alterações nesses componentes devem ser aplicadas às duas versões e verificadas pelo oráculo.

## Resultados experimentais preliminares

Nos experimentos no Franky, o treino completo usou 3.823 amostras em ordem fixa por 51 épocas (194.973 atualizações SGD). O código integrado reduz os lançamentos de treino de 194.973 (um kernel por amostra na variante anterior) para 51 (um por época). Isso reduz o custo de lançamento, mas não transforma o SGD sequencial em um treinamento totalmente paralelo.

As medianas abaixo são de **três execuções intercaladas por comparação**, com aquecimento em processos separados, sem profiler. Cada linha representa uma rodada própria; valores de linhas diferentes não formam, necessariamente, um único experimento controlado.

| Comparação | Versão A: treino | Versão B: treino |
| --- | ---: | ---: |
| CPU de referência × CUDA por amostra | CPU: 0,283 s | CUDA por amostra: 2,972 s |
| CUDA por amostra × CUDA por época (protótipo) | Por amostra: 2,980 s | Por época: 1,846 s |
| CUDA por época (protótipo) × CUDA integrada | Protótipo: 1,845 s | Integrada: 1,840 s |

Na segunda comparação, a versão por época foi aproximadamente **1,61× mais rápida** que a versão por amostra. A versão integrada preservou esse desempenho. A CPU ainda é mais rápida para esta rede pequena; não há ganho de desempenho da GPU em relação à CPU demonstrado por estes dados. A comparação CPU/GPU de 0,283 s contra 1,840 s deve ser tratada apenas como referência entre rodadas distintas, e não como medição pareada.

Em um perfil curto anterior com Nsight Systems (128 amostras), passar de sete kernels para um kernel por amostra reduziu os lançamentos de 896 para 128. O tempo acumulado de chamadas de lançamento na CPU caiu de 6,91 ms para 3,18 ms, enquanto o tempo acumulado de kernels na GPU caiu de 3,55 ms para 2,44 ms. Esses valores ajudam a localizar o gargalo, mas não medem isoladamente o efeito dos lançamentos porque existe sobreposição entre CPU e GPU e interferência do profiler. Os tempos de treino apresentados acima foram medidos **sem profiler**.

Com `--fmad=false`, a versão integrada reproduziu os **83.672 valores** do oráculo, as **51 perdas por época**, a matriz de confusão e **1.717 acertos em 1.797 testes**. O FMA ativo permanece fora da configuração validada. O teste curto com Compute Sanitizer não encontrou problemas reportados de memória, barreiras ou corridas; não foi testada uma falha real de alocação CUDA.

## Validação experimental final

Foi feita uma rodada de cinco execuções de treino CPU e cinco CUDA, sem profiler e sem captura do oráculo, com 3.823 amostras na ordem do arquivo, 51 épocas e 1.797 imagens de teste. A CPU usou `g++ 14.2.0 -O3 -ffp-contract=off`; a GPU usou CUDA 12.8.93 e `nvcc -O3 --fmad=false`. Cada versão teve um aquecimento completo em processo separado, descartado das medições. Os cinco pares foram intercalados em jobs Slurm separados, com uma CPU por job e uma GPU para cada job CUDA. Cada processo reinicializou os pesos; o aquecimento não preserva o contexto CUDA entre processos.

Os tempos de **treinamento** foram:

| Execução | CPU (s) | CUDA (s) |
| ---: | ---: | ---: |
| 1 | 0,2696 | 1,670263 |
| 2 | 0,2693 | 1,670837 |
| 3 | 0,2687 | 1,670818 |
| 4 | 0,2687 | 1,671449 |
| 5 | 0,2683 | 1,671034 |
| **Mediana** | **0,2687** | **1,670837** |
| **Mínimo** | **0,2683** | **1,670263** |
| **Máximo** | **0,2696** | **1,671449** |

Nesta rodada, a GPU levou cerca de **6,22 vezes** o tempo da CPU para treinar esta rede pequena. As máquinas não eram idênticas: CPU em `merry_cpu`, nó compute10 com Intel Core i7-13700T; CUDA em `merry_gpu`, nó compute20 com Intel Core i9-14900 e NVIDIA RTX A1000. Não se deve atribuir toda a diferença somente à GPU. O tempo CPU é impresso com quatro casas decimais. Não se comparam aqui avaliação ou tempo total, pois os escopos de medição das duas versões não são equivalentes.

Para conferir todas as perdas e a matriz, foi usada uma cópia temporária da CPU que apenas imprime as 51 perdas após o cronômetro de treino e a matriz durante a avaliação; a matemática e o laço medido não foram alterados. Em todas as dez execuções, as 51 perdas e as 100 células da matriz coincidiram com a referência capturada anteriormente, e houve **1.717 acertos em 1.797**. A validação completa anterior, com 83.672 valores idênticos e FMA desativado, permanece válida para o mesmo código CUDA. A captura de diagnóstico e o profiler não participaram desta rodada de tempos.

A aula de CUDA fundamenta as cópias explícitas e a manutenção dos dados na GPU; a de memória compartilhada e sincronização fundamenta a reutilização dos pesos e as barreiras dentro do bloco. O estudo de caso mostra a redução dos lançamentos por época e orienta usar profiling para investigar custos, medindo desempenho sem profiler. A rede e a ordem fixa do SGD desta APS foram preservadas. O FMA ativo continua experimental, e a duplicação do suporte CPU/CUDA permanece para avaliação separada.
