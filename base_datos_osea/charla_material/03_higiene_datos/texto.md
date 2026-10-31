# Eje 3 — Higiene de datos: antes de culpar al modelo, limpiar los datos

## Qué es este eje

Tres problemas de datos, no de arquitectura, que cambiaron métricas más que cualquier ajuste de
modelo: fuga de espécimen entre train/val/test, un pipeline de fotorrealismo con IA evaluado y
descartado, y el efecto medible de sumar más fotos reales curadas con el criterio correcto.

## Línea temporal

- **2026-07-23 — Primer split por fila, fuga detectada de raíz.** Con el pipeline de features
  geométricas entrenado sobre un split aleatorio por fila, la accuracy parecía razonable (76-84%),
  pero casi todos los huesos tenían 1 solo espécimen físico real (24 vistas del mismo objeto) — el
  split repartía vistas del MISMO hueso entre train y test. Medía "reconozco este objeto concreto
  desde otro ángulo", no generalización. Con `GroupKFold` (espécimen fuera de train) la accuracy
  real cayó de 83.8% a **65.5%**, y grupos con solo 2 especímenes (costilla, sacro, vértebra) se
  desplomaron a **0% de recall** en algunos folds.

- **2026-08-03 — Auditoría explícita de fuga train/val/test, pedida por el usuario tras ver caer
  el híbrido.** Al sumar 37 fotos reales curadas (commit `c29bb88`), el híbrido empeoró de 0.772 a
  0.702 en el banco real de 57 fotos — se auditó si había fuga. **Conclusión: no la había.**
  `train_cnn.py` reparte por espécimen (`GroupShuffleSplit`, con excepción explícita si hay
  solape), `train_geometric.py` usa `GroupKFold`. El único sujeto compartido entre train y banco
  sintético (`nmnhvz_7049315`) aporta `bone_group` distinto en cada lado (mandíbula vs cráneo del
  mismo animal) — no es fuga real. La caída del híbrido resultó ser un problema de calibración de
  checkpoint (cabeza compartida descalibrada por fine-tuning corto), no de datos contaminados.

- **2026-08-13 — Fix de agrupación por `species::specimen`, no solo `specimen`.** La clave de
  agrupación por espécimen sola podía colisionar entre especies o colecciones distintas por
  casualidad de numeración. Repetida la auditoría de leakage con la clave nueva sobre
  `combined.csv` (4343 filas / 418 sujetos): 0 sujetos compartidos entre train y val, 0 compartidos
  con el examen honesto de 77 fotos reales. Confirmado otra vez: sin fuga real.

- **2026-07-26 — Pipeline de fotorrealismo con IA, rechazado.** 27 candidatos (Stable Diffusion
  img2img sobre renders) revisados a mano uno por uno. Veredicto del usuario: "todos horribles",
  incluido el cráneo que en un primer vistazo parecía convincente. Decisión: se abandona por
  completo la vía de fotorrealizar renders con difusión — no se sigue con más lotes ni con otro
  modelo. Limpieza de código (`generate_ai_variants.py`, `review_ai_generated.py`) y directorios
  locales eliminados.

- **2026-08-13 — Corrección del criterio de curación: especie no descalifica.** Primera revisión
  de 21 fotos de mandíbula buscadas para cerrar el hueco de `mandibula_maxilar` (0 fotos reales):
  **0 aceptadas**, 17 rechazadas por "especie incorrecta" (perro, lobo, ciervo...). El usuario frenó
  la revisión: el filtro correcto es por grupo anatómico (¿se reconoce la forma de mandíbula?), la
  especie se resuelve después, no es bloqueante aquí. Re-revisadas las mismas 21 fotos con el
  criterio corregido: **5 aceptadas** (3 de especie no-primate, a propósito). `combined.csv`
  3892→3897 filas.

- **2026-10-xx (sesión de taxón multi-clase) — Fotos reales curadas de taxón cambian el recall de
  `ungulado` en cráneo.** Con el taxón multi-clase (humano/carnívoro/ungulado/roedor/otro_primate),
  `ungulado` salía con **recall 0.0 en validación sintética** (alarma real, sin ninguna foto real de
  ungulado en el split de test para confirmarlo). Tras revisar y aceptar 20 fotos nuevas de
  Openverse (3 ungulado, 6 roedor, 11 carnívoro) con split train/test explícito por categoría,
  cráneo real pasó de 77 a 91 filas. Resultado: **`ungulado` en validación sintética pasó de recall
  0.0 a 0.600.** Resto de clases también razonables por primera vez: carnívoro 0.688, humano 0.700,
  otro_primate 0.784, roedor 0.750. En el examen honesto (n=12, el doble que antes), ungulado quedó
  en 0/1 — con n=1 no se puede concluir nada del examen real todavía, aunque la mejora en validación
  sintética sí es clara.

## Métricas: resumen de los tres golpes

| Qué cambió | Antes | Después |
|---|---|---|
| Split por fila → `GroupKFold` por espécimen (2026-07-23) | 83.8% accuracy (split por fila, optimista) | 65.5% accuracy (honesto, espécimen fuera) |
| Fotorrealismo IA | 27 candidatos generados | 0 adoptados — vía cerrada por completo |
| Criterio de curación mandíbula (2026-08-13) | 0/21 aceptadas (filtro por especie) | 5/21 aceptadas (filtro por grupo anatómico) |
| Fotos reales de taxón en cráneo | `ungulado` recall 0.0 (sintético) | `ungulado` recall 0.600 (sintético); 0/1 en examen real (n insuficiente) |

## Imágenes incluidas

Ninguna — este eje es tabular/numérico, no visual. Si se quiere una captura de MLflow o del app de
revisión (`scripts/review_app.py`) para la charla, no existe guardada en disco; habría que
generarla a mano desde las herramientas en marcha.
