# Oráculo numérico

O modo de diagnóstico registra o que o Barenn calcula para comparar versões futuras. Ele não altera a ordem das amostras nem a atualização SGD. A compilação normal não inclui o código de captura. Os tempos desse modo **não** são usados para benchmark.

## Gerar uma referência

Na raiz do repositório, com C++17 e Python 3:

```bash
g++ -std=c++17 -O3 -g -DBARENN_ORACLE barenn.cpp -o /tmp/barenn-oraculo
BARENN_ORACLE_OUTPUT=/tmp/referencia-barenn.tsv /tmp/barenn-oraculo
```

Escolha um arquivo de saída que ainda não exista. A variável ativa a captura apenas no binário compilado com `BARENN_ORACLE`; sem ela, o programa segue a interface original. Registre junto da captura o commit, compilador e versão, flags, CPU/GPU e hashes dos datasets. Por exemplo:

```bash
git rev-parse HEAD
g++ --version
sha256sum barenn.cpp optdigits.tra optdigits.tes
```

O arquivo começa com `BARENN_ORACLE`, versão `1`, e contém uma chave e um número por linha, separados por tabulação. Os valores de rede são serializados com precisão suficiente para recuperar um `float`; as perdas médias por época usam precisão de `double`. Não finitos e falhas de escrita encerram o diagnóstico com erro.

A captura contém todos os pesos e vieses iniciais e finais; perdas das 51 épocas; dez escores, perda, rótulo, previsão e margem de cada amostra de teste; matriz de confusão e total de acertos. Para as duas primeiras amostras das épocas 1, 2 e 51, registra também entradas, valores antes da ativação, saídas, gradientes, e parâmetros antes e depois de cada atualização. Os índices de amostras começam em zero. Capturas são arquivos de diagnóstico e não precisam ser versionadas.

## Comparar capturas

```bash
python3 scripts/validar_oraculo.py /tmp/referencia-barenn.tsv /tmp/candidato-barenn.tsv --atol 0 --rtol 0
```

O exemplo exige identidade numérica e é apropriado para testar repetibilidade no mesmo ambiente. Para comparar arquiteturas ou alterações futuras, defina explicitamente tolerâncias justificadas antes do teste de aceite. O comparador usa `abs(candidato-referencia) <= atol + rtol*abs(referencia)` e exige igualdade exata para rótulos, previsões, matriz de confusão e acertos. Ele verifica o cabeçalho, a presença das chaves esperadas, duplicatas e valores não finitos. Retorna 0 para comparação aprovada, 1 para divergências e 2 para arquivo ou argumento inválido. Por padrão espera 1.797 amostras de teste; para testes controlados use `--amostras-teste N`.

O arquivo e o comparador seguem a arquitetura e o protocolo atual (64→32→16→10, 51 épocas e amostras selecionadas). Uma implementação futura precisa produzir as mesmas chaves para ser comparada. O mesmo `atol` e `rtol` é aplicado às grandezas de ponto flutuante; examine os erros por chave e, quando necessário, compare categorias separadamente antes de fixar critérios específicos. Valores perto de zero precisam de tolerância absoluta, e qualquer previsão diferente é reportada como falha.

## Testes

```bash
python3 tests/testar_oraculo.py
```

O teste compila a versão original `27f88c44` e a atual, compara o pré-processamento sem macro e cinco saídas de cada build sem captura. Também compara duas capturas, verifica a continuidade dos parâmetros entre amostras, casos pequenos de camada, gradiente, ReLU e SGD, e erros no comparador. São usados diretórios temporários; os datasets permanecem intactos. É necessário que o commit original esteja disponível no clone.

A concordância da saída impressa com a versão original não prova que todos os estados internos são iguais. Uma conferência com GDB no código original comparou 18 valores da primeira amostra (pesos, vieses, escores e gradientes) com a captura, todos iguais a nove algarismos significativos. Essa verificação cobre apenas os pontos selecionados, não toda a trajetória. Tolerâncias entre CPU e CUDA também não foram calibradas: uma comparação deve começar com operações e uma atualização a partir dos mesmos parâmetros, avançar para trajetórias curtas e, por fim, para o treino completo, verificando mudanças de classe e margens.
