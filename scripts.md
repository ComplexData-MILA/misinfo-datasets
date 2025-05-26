## 20250519- LangFuse Example

```bash
REPO=hf://ComplexDataLab/Misinfo_Datasets@ce06269

source .env && \
IGNORE_CACHE=1 uv run -m misinfo_data_eval.entrypoint_langfuse \
--langfuse_dataset_name cdl-misinfo-ce06269-dry-run-2a \
--langfuse_populate_dataset \
--evaluator_model_name gpt-4o-mini-2024-07-18 \
--web_search_variant oai \
--source_dataset_paths \
"$REPO:IFND:test" \
"$REPO:antivax:test" \
"$REPO:checkcovid:test" \
"$REPO:claimskg:test" \
"$REPO:climate_fever:test" \
"$REPO:cmu_miscov19:test" \
"$REPO:coaid:test" \
"$REPO:covid-19-disinformation:test" \
"$REPO:covid_19_rumor:test" \
"$REPO:covid_vaccine_misinfo_mic:test" \
"$REPO:covidfact:test" \
"$REPO:defakts:test" \
"$REPO:esoc:test" \
"$REPO:fakecovid:test" \
"$REPO:faviq:test" \
"$REPO:fever:test" \
"$REPO:feverous:test" \
"$REPO:fibvid:test" \
"$REPO:hover:test" \
"$REPO:liar:test" \
"$REPO:liar_new:test" \
"$REPO:mediaeval:test" \
"$REPO:mm-covid:test" \
"$REPO:multiclaim:test" \
"$REPO:nlp4if:test" \
"$REPO:pheme:test" \
"$REPO:pubhealthtab:test" \
"$REPO:rumors:test" \
"$REPO:snopes:test" \
"$REPO:truthseeker2023:test" \
"$REPO:twitter15:test" \
"$REPO:twitter16:test" \
"$REPO:verite:test" \
"$REPO:wico:test" \
"$REPO:x_fact:test" \
--max_concurrency 32 \
--limit 9
```

## 20250527- Feasibility Evaluation

```bash
REPO=hf://ComplexDataLab/Misinfo_Datasets@ce06269

source .env && \
IGNORE_CACHE=0 uv run -m misinfo_data_eval.entrypoint_langfuse \
--langfuse_dataset_name cdl-misinfo-ce06269-1a \
--evaluator_model_name gpt-4o-mini-2024-07-18 \
--web_search_variant oai \
--source_dataset_paths \
"$REPO:IFND:test" \
"$REPO:antivax:test" \
"$REPO:checkcovid:test" \
"$REPO:claimskg:test" \
"$REPO:climate_fever:test" \
"$REPO:cmu_miscov19:test" \
"$REPO:coaid:test" \
"$REPO:covid-19-disinformation:test" \
"$REPO:covid_19_rumor:test" \
"$REPO:covid_vaccine_misinfo_mic:test" \
"$REPO:covidfact:test" \
"$REPO:defakts:test" \
"$REPO:esoc:test" \
"$REPO:fakecovid:test" \
"$REPO:faviq:test" \
"$REPO:fever:test" \
"$REPO:feverous:test" \
"$REPO:fibvid:test" \
"$REPO:hover:test" \
"$REPO:liar:test" \
"$REPO:liar_new:test" \
"$REPO:mediaeval:test" \
"$REPO:mm-covid:test" \
"$REPO:multiclaim:test" \
"$REPO:nlp4if:test" \
"$REPO:pheme:test" \
"$REPO:pubhealthtab:test" \
"$REPO:rumors:test" \
"$REPO:snopes:test" \
"$REPO:truthseeker2023:test" \
"$REPO:twitter15:test" \
"$REPO:twitter16:test" \
"$REPO:verite:test" \
"$REPO:wico:test" \
"$REPO:x_fact:test" \
--max_concurrency 32 \
--limit 324
```

```bash
# wget -O data/export.jsonl \
# "https://langfuse-mtrl5-s3.j7n.me/langfuse/exports/1748265146124-lf-traces-export-cmavfjk0b0006p3064utgd9ax.jsonl"
# wget -O data/export.jsonl \
# "https://langfuse-mtrl5-s3.j7n.me/langfuse/exports/1748267827463-lf-traces-export-cmavfjk0b0006p3064utgd9ax.jsonl"
wget -O data/export.jsonl \
"https://langfuse-mtrl5-s3.j7n.me/langfuse/exports/1748287226769-lf-traces-export-cmavfjk0b0006p3064utgd9ax.jsonl"

uv run -m misinfo_data_eval.metrics.feasibility_accuracy_correlation \
data/export.jsonl
```

```javascript
                 accuracy  precision    recall        f1  mutual_info  phi_coefficient   tn    fp    fn    tp  total
data_source                                                                                                         
IFND             0.592476   0.709402  0.728070  0.718615     0.000193        -0.019592   23    68    62   166    319
checkcovid       0.754545   0.825641  0.889503  0.856383     0.000221         0.021308    5    34    20   161    220
claimskg         0.602410   0.771242  0.648352  0.704478     0.006497         0.114772   32    35    64   118    249
climate_fever    0.622378   0.700000  0.785714  0.740385     0.001638         0.057735   12    33    21    77    143
coaid            0.947368   1.000000  0.944444  0.971429     0.133229         0.687184    1     0     1    17     19
covid_19_rumor   0.698565   0.815029  0.819767  0.817391     0.001092        -0.045587    5    32    31   141    209
covidfact        0.582555   0.657025  0.757143  0.703540     0.000054         0.010374   28    83    51   159    321
defakts          0.619497   0.704698  0.576923  0.634441     0.032041         0.251226   92    44    77   105    318
esoc             0.511111   0.570681  0.602210  0.586022     0.000049        -0.009847   52    82    72   109    315
fakecovid        0.715385   0.936842  0.741667  0.827907     0.003362         0.085106    4     6    31    89    130
faviq            0.799383   0.817891  0.969697  0.887348     0.000808         0.042248    3    57     8   256    324
fever            0.845833   0.889868  0.943925  0.916100     0.000320        -0.024185    1    25    12   202    240
feverous         0.738983   0.758007  0.959459  0.846918     0.001489         0.056734    5    68     9   213    295
fibvid           0.618056   0.797872  0.675676  0.731707     0.003813         0.088209   14    19    36    75    144
hover            0.642405   0.710744  0.800000  0.752735     0.006726         0.117749   31    70    43   172    316
liar             0.576324   0.610169  0.765957  0.679245     0.003408         0.082872   41    92    44   144    321
liar_new         0.638365   0.858407  0.700361  0.771372     0.001838        -0.059219    9    32    83   194    318
mm-covid         0.661392   0.870536  0.714286  0.784708     0.000444         0.030089   14    29    78   195    316
multiclaim       0.761364   0.904412  0.809211  0.854167     0.021007         0.219090   11    13    29   123    176
nlp4if           0.548223   0.520833  0.789474  0.627615     0.008180         0.127331   33    69    20    75    197
pubhealthtab     0.636000   0.669065  0.673913  0.671480     0.034996         0.263429   66    46    45    93    250
rumors           0.817143   0.952055  0.847561  0.896774     0.007742         0.137858    4     7    25   139    175
snopes           0.779167   0.978610  0.788793  0.873508     0.006489         0.124973    4     4    49   183    240
truthseeker2023  0.800000   1.000000  0.800000  0.888889     0.000000         0.000000    0     0     1     4      5
twitter15        0.580882   0.738095  0.639175  0.685083     0.002419         0.069865   17    22    35    62    136
twitter16        0.500000   0.510638  0.571429  0.539326     0.000007        -0.003609   17    23    18    24     82
verite           0.545946   0.661765  0.703125  0.681818     0.006147        -0.108707   11    46    38    90    185
x_fact           0.540000   0.619355  0.548571  0.581818     0.002855         0.075542   66    59    79    96    300
ALL              0.651924   0.760262  0.762927  0.761592     0.006617         0.117013  601  1098  1082  3482   6263
```