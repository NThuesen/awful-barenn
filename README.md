# Barenn — APS2 de SuperComputação

Este repositório reúne as otimizações de uma rede neural *feedforward* em C++ e sua implementação CUDA. O objetivo foi melhorar o desempenho sem mudar a matemática: mantivemos as 51 épocas, a ordem das amostras e as atualizações SGD uma a uma.

O resultado não foi simplesmente “GPU mais rápida”. A CPU melhorou em etapas, uma mudança de OpenMP ficou mais lenta e foi descartada, e a versão CUDA ganhou desempenho em relação à sua própria implementação anterior — embora ainda tenha perdido para a CPU nesta carga pequena. Abaixo estão os números e o caminho para reproduzir cada conclusão.

## Resultados em uma página

| Versão | Tag do código | Treino (mediana) | Onde foi medido |
| --- | --- | ---: | --- |
| Baseline legado | [v0.0-cpu-baseline](https://github.com/NThuesen/awful-barenn/tree/v0.0-cpu-baseline) | **0,3622 s** | CPU, `compute10`, [#2](https://github.com/NThuesen/awful-barenn/issues/2) |
| CPU otimizada (estado final da CPU) | [v0.5-cpu-openmp-datasets](https://github.com/NThuesen/awful-barenn/tree/v0.5-cpu-openmp-datasets) | **aprox. 0,255–0,256 s** | CPU, `compute10`; treino sem mudança material no experimento [#18](https://github.com/NThuesen/awful-barenn/issues/18) |
| CUDA inicial funcional | [v0.2-cuda-funcional](https://github.com/NThuesen/awful-barenn/tree/v0.2-cuda-funcional) | **1,676300 s** | RTX A1000, `compute20`, rodada [#10](https://github.com/NThuesen/awful-barenn/issues/10) |
| CUDA otimizada (64 threads, marco anterior) | [v0.3-cuda-backward-paralelo](https://github.com/NThuesen/awful-barenn/tree/v0.3-cuda-backward-paralelo) | **0,926385 s** | RTX A1000, `compute20`, mesma rodada [#10](https://github.com/NThuesen/awful-barenn/issues/10) |
| CUDA otimizada (256 threads, novo marco) | [v0.6-cuda-bloco-256](https://github.com/NThuesen/awful-barenn/tree/v0.6-cuda-bloco-256) | **0,579681 s / 0,579527 s** | RTX A1000, `compute20`, duas rodadas controladas da [#30](https://github.com/NThuesen/awful-barenn/issues/30) |

Esses tempos são do **treinamento**, medidos sem profiler. Os pares CUDA da #10 são da mesma rodada; a #30 também tem suas **próprias referências de 64 threads medidas em cada uma de duas rodadas**, não substituídas pela mediana histórica da #10. A versão CPU final otimiza também a **carga dos dados**, cujo ganho aparece no tempo do processo completo; misturar essa métrica com tempo de treino seria incorreto. Não calculamos um único speedup “CPU final versus baseline” a partir de rodadas diferentes.

| Mudança testada | Comparação feita na mesma rodada | Resultado |
| --- | --- | --- |
| Localidade no backward da CPU ([#5](https://github.com/NThuesen/awful-barenn/issues/5)) | 0,3618 → 0,2691 s de **treino** | **1,34×** |
| Evitar cópias no `Backward` ([#4](https://github.com/NThuesen/awful-barenn/issues/4)) | 0,2703 → 0,2561 s; segunda rodada 0,2698 → 0,2562 s de **treino** | **1,055× / 1,053×** |
| Conversão OpenMP de datasets, quatro threads ([#18](https://github.com/NThuesen/awful-barenn/issues/18)) | 0,313063 → 0,298183 s; segunda rodada 0,310348 → 0,294662 s de **processo completo** | **1,050× / 1,053×** |
| Paralelismo de pesos no backward CUDA ([#10](https://github.com/NThuesen/awful-barenn/issues/10)) | 1,676300 → 0,926385 s de **treino** | **1,81×** |
| Parâmetros CUDA global × shared ([#22](https://github.com/NThuesen/awful-barenn/issues/22)) | 1,235175 → 0,933711 s de **treino**, controle global experimental | **1,323×** |
| Bloco CUDA 64 → 256 threads ([#30](https://github.com/NThuesen/awful-barenn/issues/30)) | 0,925028 → 0,579681 s; confirmação 0,924796 → 0,579527 s de **treino** | **1,596× / 1,596×** |
| Bloco CUDA 64 → 256 threads ([#30](https://github.com/NThuesen/awful-barenn/issues/30)) | 3,126554 → 2,781905 s; confirmação 3,118194 → 2,780509 s de **processo completo** | **1,124× / 1,121×** |

O controle global da #22 não foi integrado: a comparação isola a escolha de localização junto com as cópias de parâmetros por época, com recursos compartilhados e ordem fixa dos pares. O compilador GCC 14.2.0 também confirmou vetorização de 16 bytes no laço interno da CPU otimizada ([#5](https://github.com/NThuesen/awful-barenn/issues/5)), sem quantificar sua contribuição ao ganho.

As comparações #5, #4, #18, #10 e #22 usaram cinco medições por configuração; a **#30 usou seis trios por rodada e duas rodadas independentes**, intercalando todas as ordens de 64/128/256 threads. Cada Issue registra dispersão, aquecimento, hardware e flags. Os ganhos **não devem ser multiplicados** como se tivessem sido obtidos em um único benchmark. A tentativa de OpenMP dentro do `Forward` ([#6](https://github.com/NThuesen/awful-barenn/issues/6)) preservou os resultados, mas piorou o desempenho; ficou fora da versão final. A ideia de reduzir cópias CUDA ([#9](https://github.com/NThuesen/awful-barenn/issues/9)) também não foi implementada, porque o profiling disponível não sustentou sua prioridade. A hipótese de pesos contíguos da CPU ([#29](https://github.com/NThuesen/awful-barenn/issues/29)) também regrediu, mesmo com oráculo idêntico, e não foi adotada.

## Corretude e limites da comparação

Usamos os arquivos originais `optdigits.tra` (3.823 amostras) e `optdigits.tes` (1.797 testes), com a rede **64 → 32 → 16 → 10**, 51 épocas e SGD na ordem do arquivo. Nas configurações validadas, **83.672 valores** coincidiram com o oráculo numérico, com `atol=0` e `rtol=0`; as execuções reproduziram **1.717 acertos em 1.797 testes**, as 51 perdas e a matriz de confusão.

Para preservar essa igualdade, as medições de referência CPU usaram `-ffp-contract=off` e a CUDA `--fmad=false`. Não estendemos a afirmação de igualdade exata a outras flags ou à CUDA com FMA ativo. As medidas CPU ocorreram no `compute10` (Intel i7-13700T), e as CUDA no `compute20` (Intel i9-14900 + RTX A1000): a diferença entre nós impede atribuir toda a comparação CPU/GPU apenas ao tipo de processador.

O [profiling de CPU](https://github.com/NThuesen/awful-barenn/issues/2) identificou o custo do gradiente de entrada no backward. No [profiling da CUDA inicial](https://github.com/NThuesen/awful-barenn/issues/8), o kernel de treino concentrou 99,37% do tempo acumulado dos kernels. Um [trace Nsight Systems da CUDA otimizada de 64 threads (v0.3)](https://github.com/NThuesen/awful-barenn/issues/11#issuecomment-7113595114), em outra execução **instrumentada**, registrou 51 lançamentos de `train_epoch` (951,128 ms acumulados) e 2,783 ms de eventos H2D+D2H na GPU. Esses valores de profiling **não** entram nos speedups, que vêm das medições pareadas sem profiler.

Na imagem carregada da RTX A1000, tanto a versão anterior **v0.3 (64 threads)** quanto a nova **v0.6 (256 threads)** usam **51 registradores/thread e 12.000 bytes de shared estática por bloco**. Para 64 threads, a API CUDA indicou limite de **7 blocos/SM**, ou **29,17% de ocupação teórica** com blocos suficientes. Para 256 threads, indicou **4 blocos/SM**, ou **66,67% de ocupação teórica** com blocos suficientes. A grid de treinamento continua tendo **apenas um bloco**: isso limita os warps residentes daquele lançamento a **2/48 (4,17%) com 64 threads** ou **8/48 (16,67%) com 256 threads** no único SM que recebe o bloco; os demais 17 SMs não recebem trabalho desse lançamento. Esses valores são **limites calculados, não ocupação real medida**. O Nsight Compute retornou `ERR_NVGPUCTRPERM` nas GPUs acessíveis testadas; ocupação atingida, throughput, tráfego interno da memória e custo isolado de barreiras permanecem sem medição. A melhora da #30 é compatível com menos atualizações de pesos por thread (2.048 pesos da primeira camada: 32 → 8), sem permitir atribuir uma fração causal a um único recurso.

## Como executar e verificar

Na raiz da versão final CPU (C++17 e OpenMP disponíveis):

```bash
git checkout v0.5-cpu-openmp-datasets
g++ -std=c++17 -O3 -g -ffp-contract=off -fopenmp barenn.cpp -o barenn_cpu
OMP_NUM_THREADS=4 OMP_DYNAMIC=FALSE ./barenn_cpu
python3 tests/testar_oraculo.py
python3 tests/testar_leitura_openmp.py
```

### Compilação opcional com AVX2

Em processadores com suporte a AVX2, é possível acrescentar `-mavx2` ao comando CPU acima. Essa opção pode produzir um executável incompatível com CPUs sem AVX2; mantenha o comando padrão para portabilidade. Não é necessário `-ffast-math`. AVX2 explora SIMD e não adiciona paralelismo entre núcleos.

Na comparação pareada da [#24](https://github.com/NThuesen/awful-barenn/issues/24), a mediana de treino passou de 0,2557 s para 0,2117 s (1,208×) e a do processo completo de 0,318 s para 0,260 s (1,223×), no i7-13700T do Franky. O oráculo comparou 83.672 valores sem divergências; são resultados específicos desse nó e dessas flags, não uma garantia para outros processadores.

A opção `-fopenmp` habilita a conversão paralela; sem ela, o mesmo código pode rodar sequencialmente. Quatro threads foram a configuração mais estável entre as testadas no Franky; não é uma recomendação universal. Nos benchmarks, a afinidade das threads também foi configurada (`OMP_PLACES=cores`, `OMP_PROC_BIND=close`, `OMP_WAIT_POLICY=PASSIVE`); em computadores diferentes, essas opções devem ser adequadas à topologia disponível. Os testes requerem Python 3 e um compilador compatível. A suíte do oráculo utiliza commits históricos, portanto execute em clone com o histórico disponível.

Para reproduzir o marco CUDA mais recente no Franky, faça o checkout da tag correspondente no nó de acesso e prepare os cabeçalhos quando necessário, conforme as instruções de [compilação e validação](docs/cuda.md). Em seguida, solicite uma GPU e execute os comandos no nó alocado:

```bash
git checkout v0.6-cuda-bloco-256
srun --partition=merry_gpu --gres=gpu:1 --cpus-per-task=1 --mem=2048M --time=00:04:30 --pty bash
module load gcc/14.2.0
module load cuda/12.8.1
scripts/compilar_cuda.sh ./barenn_cuda
./barenn_cuda
```

Os comandos completos de Slurm, o procedimento quando faltam cabeçalhos e o uso do [oráculo numérico](docs/oraculo-numerico.md) estão nesses documentos. Os tempos apresentados acima são resultados coletados **no Franky**, não benchmarks executados no computador pessoal.

## Índice dos experimentos

| Etapa | Evidências e discussão | Integração / marco |
| --- | --- | --- |
| Planejamento | [#1 — roadmap](https://github.com/NThuesen/awful-barenn/issues/1) | Acompanhamento; sem tag própria |
| Baseline e profiling CPU | [#2](https://github.com/NThuesen/awful-barenn/issues/2) | [v0.0-cpu-baseline](https://github.com/NThuesen/awful-barenn/tree/v0.0-cpu-baseline) |
| Oráculo de corretude | [#3](https://github.com/NThuesen/awful-barenn/issues/3) | [PR #13](https://github.com/NThuesen/awful-barenn/pull/13) · [v0.0-oraculo-numerico](https://github.com/NThuesen/awful-barenn/tree/v0.0-oraculo-numerico) |
| Localidade da CPU | [#5](https://github.com/NThuesen/awful-barenn/issues/5) | [PR #14](https://github.com/NThuesen/awful-barenn/pull/14) · [v0.1-cpu-localidade](https://github.com/NThuesen/awful-barenn/tree/v0.1-cpu-localidade) |
| Migração CUDA | [#7](https://github.com/NThuesen/awful-barenn/issues/7) | [PR #15](https://github.com/NThuesen/awful-barenn/pull/15) · [v0.2-cuda-funcional](https://github.com/NThuesen/awful-barenn/tree/v0.2-cuda-funcional) |
| Profiling GPU | [#8](https://github.com/NThuesen/awful-barenn/issues/8) | Diagnóstico sem mudança de código |
| Paralelismo CUDA | [#10](https://github.com/NThuesen/awful-barenn/issues/10) | [PR #16](https://github.com/NThuesen/awful-barenn/pull/16) · [v0.3-cuda-backward-paralelo](https://github.com/NThuesen/awful-barenn/tree/v0.3-cuda-backward-paralelo) |
| Redução de cópias CPU | [#4](https://github.com/NThuesen/awful-barenn/issues/4) | [PR #17](https://github.com/NThuesen/awful-barenn/pull/17) · [v0.4-cpu-backward-const-ref](https://github.com/NThuesen/awful-barenn/tree/v0.4-cpu-backward-const-ref) |
| OpenMP dentro do Forward | [#6](https://github.com/NThuesen/awful-barenn/issues/6) | Experimento descartado; sem PR/tag |
| Carga OpenMP dos datasets | [#18](https://github.com/NThuesen/awful-barenn/issues/18) | [PR #19](https://github.com/NThuesen/awful-barenn/pull/19) · [v0.5-cpu-openmp-datasets](https://github.com/NThuesen/awful-barenn/tree/v0.5-cpu-openmp-datasets) |
| Controle CUDA global × shared | [#22](https://github.com/NThuesen/awful-barenn/issues/22) | [PR #23 — documentação](https://github.com/NThuesen/awful-barenn/pull/23) · [controle `60e835c`](https://github.com/NThuesen/awful-barenn/commit/60e835c), sem tag própria |
| CPU com AVX2 opcional | [#24](https://github.com/NThuesen/awful-barenn/issues/24) | [PR #25 — documentação](https://github.com/NThuesen/awful-barenn/pull/25) · base [v0.5](https://github.com/NThuesen/awful-barenn/tree/v0.5-cpu-openmp-datasets) |
| Avaliação multicore experimental | [#26](https://github.com/NThuesen/awful-barenn/issues/26) | [Controle `38efba1`](https://github.com/NThuesen/awful-barenn/commit/38efba1), sem ganho consistente no processo nem nova tag |
| Pesos contíguos CPU (regressivo) | [#29](https://github.com/NThuesen/awful-barenn/issues/29) | [Controle `eb05cdd`](https://github.com/NThuesen/awful-barenn/commit/eb05cddbc94a2b2aeddd19fe56023a709f7f6bcc), não adotado, sem tag de ganho |
| Bloco de treino CUDA 64/128/256 | [#30](https://github.com/NThuesen/awful-barenn/issues/30) | [PR #31](https://github.com/NThuesen/awful-barenn/pull/31) · [v0.6-cuda-bloco-256](https://github.com/NThuesen/awful-barenn/tree/v0.6-cuda-bloco-256) |
| Transferências CUDA | [#9](https://github.com/NThuesen/awful-barenn/issues/9) | Hipótese não implementada; sem PR/tag |
| Comparação histórica | [#11](https://github.com/NThuesen/awful-barenn/issues/11) | Relatório; sem tag própria |
| Conferência de entrega | [#12](https://github.com/NThuesen/awful-barenn/issues/12) | Checklist e auditoria final |

As **sete tags anotadas históricas até v0.5** permanecem preservadas. O novo marco **v0.6-cuda-bloco-256** corresponde ao commit de integração [PR #31 (`296140af`)](https://github.com/NThuesen/awful-barenn/commit/296140af58a2a55b40c50bdbd423fe8efe17c920), com ganho medido na [#30](https://github.com/NThuesen/awful-barenn/issues/30). A branch padrão e de entrega é `main`, que contém os PRs de implementação aprovados e a ancestralidade dos controles experimentais da [documentação de tentativas descartadas](docs/experimentos-controle.md); os códigos regressivos não substituem o código final. A antiga `master` permanece como referência histórica. Tags por desempenho somente para mudanças positivas integradas e reproduzíveis.

---

## Enunciado da atividade (mantido abaixo para consulta)

# Projeto Barenn: O Legado, a Sobrevivência e o Salto para a GPU

Era uma segunda-feira aparentemente tranquila quando a demanda chegou à sua mesa. A liderança técnica se aproximou com um sorriso contido e soltou aquela frase clássica que faz qualquer engenheiro gelar por dentro:

> "Temos um componente crítico rodando em produção há anos. O desenvolvedor original já saiu da empresa, mas agora o projeto é oficialmente seu."
    
Você procurou documentação, manuais ou notas de arquitetura e não encontrou rigorosamente nada. Restou apenas um código hermético a correr em produção.

Depois de analisar o código entendeu que é a implementação de uma Feedforward Neural Network, mas por que razão reinventar a roda? No passado, exigências estritas de conformidade e restrições de licenciamento impediram o recurso a bibliotecas de mercado. Sem alternativa, a empresa optou por criar a sua própria implementação de raiz. O resultado funcionou durante anos, mas tornou-se um labirinto técnico esquecido.

Com o aumento exponencial do volume de dados, a máquina atual começou a falhar nos prazos operacionais. Antes de qualquer migração complexa, **o seu primeiro desafio é estabilizar o código e espremer todo o desempenho possível da CPU — atacando gargalos, repensando o acesso à memória e explorando o paralelismo nativo.**

Para resolver o problema a longo prazo, a empresa adquiriu servidores blade equipados com GPUs corporativas NVIDIA. O hardware de alto desempenho já está montado no datacenter: **a sua missão final é reescrever os pontos críticos do algoritmo em CUDA e levar o processamento ao nível do paralelismo massivo.**

---

## Diretrizes de Conduta e Governança no Repositório

Neste projeto, o repositório não é apenas um local de entrega de arquivos, mas a **comprovação do seu processo investigativo de engenharia**. Espera-se uma postura estritamente empírica: nenhuma otimização deve ser feita no escuro e nenhuma melhoria de velocidade tem valor se corromper o resultado final do programa.

---

### O Princípio do Bloco Lógico de Trabalho

Você tem autonomia para decidir **o que** otimizar e **em qual ordem**, mas cada entrega deve ser estruturada como um **bloco lógico fechado**.

> **O que é um bloco lógico?**
> É uma unidade completa de intervenção técnica que engloba: **um problema identificado por profiling $\rightarrow$ uma hipótese de melhoria $\rightarrow$ a alteração no código $\rightarrow$ a validação da corretude $\rightarrow$ a medição do ganho real.**

* **Não crie tags por tempo decorrido:** Evite tags como `versao-dia-2` ou commits de "fim de expediente".
* **Não crie tags para alterações inconclusivas:** Se a alteração piorou o desempenho ou quebrou os testes, documente na Issue, reverta ou ajuste antes de formalizar o marco.
* **Toda tag deve ser reprodutível:** O professor ou qualquer colega deve conseguir executar `git checkout <tag>`, compilar o código e obter exatamente o mesmo comportamento e ordem de grandeza de desempenho documentados.

---

### O Fluxo Esperado no GitHub

Para cada intervenção técnica planejada, seu ciclo de trabalho no repositório deve seguir três pilares:

#### 1. Abertura de Issue (A Hipótese e o Ponto de Partida)

Antes de alterar uma única linha de código, abra uma Issue explicando a intenção do bloco lógico:

* **Diagnóstico:** Qual gargalo você encontrou usando ferramentas de análise (profilers)?
* **Hipótese:** Qual conceito técnico você pretende aplicar para resolver esse gargalo?
* **Métrica Base:** Qual é o tempo ou custo de processamento atual dessa rotina?

#### 2. Desenvolvimento Isolado e Commits Semânticos

* Trabalhe em uma *branch* dedicada àquela hipótese.
* Faça commits descritivos focados na causa da mudança (ex: `perf: ...`, `refactor: ...`, `fix: ...`).
* Certifique-se de que a saída do modelo/algoritmo continua correta. Código rápido que calcula resultados errados é considerado código quebrado.

#### 3. Formalização com Tag Anotada e Fechamento da Issue

Quando o bloco estiver concluído e validado, crie uma **Git Tag anotada**:

* A tag marca o estado exato da base de código naquele degrau de maturidade.
* A mensagem da tag deve resumir sucintamente o resultado alcançado.
* Feche a Issue anexando as evidências (dados de tempo, prints do profiler ou tabelas comparando o "antes e depois").

---

### Exemplo Ilustrativo de um Bloco Lógico

*(Este é apenas um exemplo de como organizar a narrativa técnica e as ferramentas; não representa um passo obrigatório ou a solução do seu problema).*

---

#### 1. Cenário: O aluno percebeu lentidão na etapa de processamento sequencial

* **Abertura da Issue #04:**
> **Título:** *Redução de tempo de computação na rotina de cálculo vetorial*
> **Contexto:** Ao inspecionar o código com o profiler, identificou-se que 65% do tempo de execução é consumido em um laço interno que realiza operações matemáticas repetitivas.
> **Métricas atuais:** Tempo total de execução: `14.2s`.
> **Hipótese:** Reestruturar a ordem dos dados para favorecer o uso de registradores e paralelizar o laço com múltiplos núcleos deve reduzir o tempo dessa rotina.


* **Implementação na branch:** `feature/otimizacao-laco-vetorial`
* Commits atômicos ajustando o laço e aplicando a estratégia escolhida.
* Verificação de que o resultado final do programa bate com o original.


* **Criação da Tag:**
```bash
git tag -a v0.3-laco-paralelo -m "Otimizacao do calculo vetorial: tempo reduzido de 14.2s para 4.8s (Speedup 2.95x). Corretude validada."
git push origin v0.3-laco-paralelo

```


* **Fechamento da Issue:**
* O aluno responde na Issue #04 anexando o comparativo de tempo (antes: `14.2s` / depois: `4.8s`), faz o merge da branch e referencia a tag criada:


> *"Otimização implementada e validada na tag `v0.3-laco-paralelo`. Issue fechada pelo PR #05."*


---

# Rubrica de Avaliação (Total: 10,0 Pontos)

<div style="background:rgba(255, 0, 0, 0.65)"> 

> ### ⚠️ CLÁUSULA ELIMINATÓRIA GERAL (CONDIÇÃO DE NOTA ZERO)
> 
> 
> A avaliação adota uma política estrita de controle de qualidade de engenharia:
> **Se em QUALQUER um dos critérios ou subcritérios de avaliação for observado o indicativo de `0,0 ponto`, a ATIVIDADE SERÁ INTEGRALMENTE ZERADA (Nota Final = 0,0).**
> Não é tolerada a ausência completa de qualquer pilar fundamental da atividade (seja por código inoperante/incorreto, falta de governança no Git ou ausência de embasamento empírico nas Issues).

* **Implementação e Desempenho Técnico:** 4,0 pontos
* **Governança, Reprodutibilidade e Tags:** 2,0 pontos
* **Relatório Técnico nas Issues (Análise e Discussão):** 4,0 pontos

</div>

---

### Critério 1: Implementação e Desempenho Técnico (4,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — O código não compila, aborta durante a execução, não é funcional ou gera saídas matematicamente corrompidas.
* **+0,5 ponto** — Executou com sucesso o código legado, mediu o baseline de referência de forma reproduzível e identificou os gargalos iniciais.
* **+1,0 ponto** — Aplicou otimizações eficazes na CPU com ganhos consistentes de velocidade e paralelismo multicore preservando a integridade numérica.
* **+1,0 ponto** — **Migração Funcional para CUDA:** Porting inicial para GPU completo e correto (kernel funcional, alocação e liberação de memória no Device e transferências Host/Device íntegras).
* **+0,75 ponto** — Otimizou a execução em GPU.
* **+0,75 ponto** — Implementou técnicas avançadas em GPU, comprovando ganhos.

---

### Critério 2: Governança, Reprodutibilidade e Tags (2,0 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Ausência de tags, ou repositório com commit único ("tudo pronto") / commits lineares sem separação lógica, inviabilizando a reprodutibilidade dos marcos intermediários.
* **+0,5 ponto** — Utilizou tags, mas elas marcam datas/horários arbitrários em vez de blocos lógicos fechados de trabalho, ou as tags não compilam de forma independente.
* **+1,2 ponto** — Trabalho estruturado em **blocos lógicos**: cada intervenção possui uma Git Tag anotada associada a uma Issue, permitindo compilar e checar a métrica de cada versão via `git checkout <tag>`.
* **+2,0 pontos** — Reprodutibilidade e histórico impecáveis: commits atômicos com mensagens semânticas (`perf:`, `refactor:`, `test:`), branches limpas integradas via Pull Requests documentados e rastreabilidade total de cada degrau de performance.

---

### Critério 3: Relatório Técnico nas Issues (Análise e Discussão) (4,0 pontos)

#### 3.1 Profiling e Evidências Empíricas (1,5 ponto)

* **0,0 ponto [ZERA A ATIVIDADE]** — Não apresenta dados de medição nas Issues ou baseia-se unicamente em impressões subjetivas sem qualquer dado empírico.
* **+0,5 ponto** — Apresenta apenas tempos brutos de relógio (*wall-clock*) colados em texto corrido, sem dados de profilers ou repetições estatísticas.
* **+1,0 ponto** — Apresenta dados e capturas das ferramentas oficiais (**perf/callgrind** e **Nsight Systems/Compute**), mas faltam informações cruciais de ambiente (flags, parâmetros de teste ou hardware).
* **+1,5 ponto** — Issues contêm relatórios completos de profiling (CPU e GPU), métricas de hardware detalhadas (tempo de kernel vs transferência PCIe, cache misses, ocupação teórica vs real de SMs) e metodologia experimental rigorosa.

#### 3.2 Qualidade da Discussão Técnica e Argumentação de Engenharia (2,5 pontos)

* **0,0 ponto [ZERA A ATIVIDADE]** — Discussão inexistente, Issues fechadas apenas com "ok/concluído" ou explicações inconsistentes/conflitantes com os dados coletados.
* **+1,0 ponto** — Discussão puramente descritiva (lista o que foi alterado nas linhas de código, mas não analisa *por que* houve ou não ganho com base em hardware).
* **+1,8 ponto** — Discute tecnicamente a relação entre o código e o comportamento de hardware (ex: impacto do layout de memória nos caches da CPU, balanceamento de carga entre threads).
* **+2,5 pontos** — Análise técnica aprofundada: correlaciona os resultados com conceitos avançados de arquitetura (roofline model, limites da Lei de Amdahl, latência do PCIe vs throughput do kernel CUDA) e debate criticamente as decisões de engenharia tomadas.

---

## Tabela de Cenários e Balizadores

| Cenário | Situação Técnica no GitHub | Resultado / Nota |
| --- | --- | --- |
| **Desclassificado** | Tirou `0,0` em qualquer critério (ex: código não compila, sem tags, sem dados de profiling nas issues ou issues vazias). | **0,0 (Zerada)** |
| **Abaixo do Esperado** | Código compila e funciona, mas há otimizações superficiais, profiling fraco e tags incompletas. | **0,1 a 5,9** |
| **Linha de Base (Alvo: 6,0)** | CPU razoavelmente otimizada + **Migração simples para CUDA funcional** + Issues estruturadas com diagnóstico e tags atreladas. | **6,0 a 6,5** |
| **Avançado / Destaque** | CPU altamente eficiente + **CUDA com otimizações avançadas (Shared Memory/Streams)** + discussões ricas nas Issues com Nsight/Perf + governança impecável. | **8,5 a 10,0** |

---

## Formato de Entrega

A entrega consiste exclusivamente no **link do repositório no GitHub**.

1. O `README.md` principal deve atuar como o **sumário executivo do relatório**, sendo adiciocionado **NO TOPO DO README**:
* A tabela consolidada final comparando todas as versões (Baseline Legado, CPU Otimizada, CUDA Inicial, CUDA Otimizado);
* Um **índice navegável com links diretos** para cada uma das Issues/Tags que contêm a discussão detalhada;

2. Todas as branches de trabalho devem estar integradas à `main`;
3. Todas as Issues de experimentos devem estar fechadas e associadas à sua respectiva Git Tag anotada.
