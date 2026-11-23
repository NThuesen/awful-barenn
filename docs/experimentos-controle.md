# Controles experimentais não integrados como otimizações

Esta nota registra duas hipóteses investigadas na APS2 que **não** devem substituir o código de produção. Para preservar a rastreabilidade da governança, seus commits experimentais foram incorporados **à ancestralidade Git** por merges documentados que conservam a árvore de código aprovada. Essa integração histórica **não aplica** as versões alternativas ao `barenn.cpp` ou ao `cuda/barenn_cuda.cu` da `main`, e não cria tags de desempenho para hipóteses descartadas.

## #22 — CUDA: parâmetros em memória global como controle

- Referência aprovada: tag anotada `v0.3-cuda-backward-paralelo`, com parâmetros mantidos em shared.
- Controle global: commit [`60e835c2bb58462718e5a392545820aae785d0ad`](https://github.com/NThuesen/awful-barenn/commit/60e835c2bb58462718e5a392545820aae785d0ad), branch `experiment/issue-22-cuda-global`.
- Cinco pares sem profiler na RTX A1000: mediana shared **0,933711 s** e global **1,235175 s** de treino (shared **1,323×** mais rápida). Shared venceu 5/5 pares.
- Oráculo: **83.672 valores iguais**, tolerância zero. Sanitizer curto sem problemas reportados.
- A variante global também elimina cópias de parâmetros por época; o teste não mede isoladamente uma instrução nem demonstra porcentagem causal de cache/DRAM.
- Decisão: manter shared e encerrar a [Issue #22](https://github.com/NThuesen/awful-barenn/issues/22). Os dados detalhados estão [neste comentário](https://github.com/NThuesen/awful-barenn/issues/22#issuecomment-7113129341), e o sumário entrou no [PR #23](https://github.com/NThuesen/awful-barenn/pull/23).

## #26 — Avaliação CPU paralela com OpenMP

- Referência aprovada: tag anotada `v0.5-cpu-openmp-datasets`; SGD permanece sequencial entre amostras.
- Controle de inferência sem estado mutável compartilhado: commit [`38efba18598e49639824521d038dd812abcfd400`](https://github.com/NThuesen/awful-barenn/commit/38efba18598e49639824521d038dd812abcfd400), branch `experiment/issue-26-avaliacao`.
- Com quatro threads, avaliação **1,138 ms → 0,467 ms** de mediana (**2,44×** nessa etapa), mas o processo completo não demonstrou ganho consistente frente à dispersão.
- Oráculo: **83.672 valores iguais**, tolerância zero, nas configurações testadas de 1, 2, 4 e 8 threads.
- Decisão: não alterar a avaliação da versão final; Issue #26 encerrada como hipótese não adotada. Tempos e limitações estão [neste comentário](https://github.com/NThuesen/awful-barenn/issues/26#issuecomment-7113229667).

## Como reproduzir os controles a partir do histórico

Não é necessário usar branches de desenvolvimento nem alterar a `main`. O histórico Git mantém ambos os commits:

```bash
git worktree add --detach /tmp/aps2-global 60e835c2bb58462718e5a392545820aae785d0ad
git worktree add --detach /tmp/aps2-avaliacao 38efba18598e49639824521d038dd812abcfd400
```

Em cada worktree, consulte as instruções e flags da Issue correspondente. Execute compilações CUDA somente em ambiente com toolkit e GPU, usando Slurm no Franky. Os resultados históricos são específicos de hardware, dados e metodologias descritos nas Issues.

Os merges de ancestralidade foram produzidos com **estratégia de conteúdo equivalente a `git merge -s ours`**: registram os commits experimentais como ancestrais, mas conservam a árvore da `main`, exceto pela adição desta nota. Isso evita que o controle global lento ou a avaliação sem ganho de processo substituam o código aprovado. **Nenhuma tag anotada nova é criada**, pois não há novo degrau de desempenho integrado.
