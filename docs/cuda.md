# Treinamento CUDA

A versão CUDA usa um bloco por época. O bloco percorre as amostras na ordem do arquivo e mantém pesos em memória compartilhada. Barreiras separam as camadas e as etapas do backward; o gradiente de entrada usa os pesos anteriores à atualização. Após cada época, as perdas por amostra são copiadas para o host e somadas em `double`, na mesma ordem da CPU. Essa organização segue os conceitos de memória compartilhada, sincronização e redução de lançamentos da aula de Estudo de Caso e Profiling em GPU.

O script compila com `--fmad=false`, configuração usada na comparação numérica com a CPU. **FMA ativo permanece experimental e suas diferenças ainda não foram aprovadas**: no experimento anterior, oito previsões diferiram da CPU. Não se deve assumir que os resultados com FMA ativo são equivalentes. A captura do oráculo é opcional e não deve ser usada para medir desempenho.

## Compilação e execução no Franky

Solicite uma GPU conforme os limites atuais do Slurm e compile no nó alocado. Exemplo, a partir da raiz do projeto:

```bash
srun --partition=merry_gpu --gres=gpu:1 --cpus-per-task=1 --mem=2048M --time=00:04:30 --pty bash
module swap gcc linux-rocky9-x86_64/Core/gcc/14.2.0
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
