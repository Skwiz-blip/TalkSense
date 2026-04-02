# LiveYDream — Module ML (NestJS + Python BERT)

Système complet d'analyse de conversations avec apprentissage continu.

---

## Architecture

```
lyd-ml/
├── migrations/
│   └── 001_ml_training_schema.sql        # 7 tables PostgreSQL
│
├── src/modules/ml-training/
│   ├── entities/                         # TypeORM entities
│   │   ├── conversation.entity.ts
│   │   ├── conversation-annotation.entity.ts
│   │   ├── model-feedback.entity.ts
│   │   ├── training-run.entity.ts
│   │   ├── model-deployment.entity.ts
│   │   ├── extracted-action.entity.ts
│   │   └── ml-inference-log.entity.ts
│   ├── dto/index.ts                      # Validated DTOs
│   ├── services/
│   │   ├── model-inference.service.ts    # ONNX inference + Redis cache
│   │   ├── ml-training.service.ts        # Training pipeline
│   │   └── feedback.service.ts           # Feedback + Conversation services
│   ├── controllers/
│   │   └── ml-training.controller.ts     # REST endpoints
│   ├── processors/
│   │   └── training.processor.ts         # Bull queue job handler
│   ├── ml-training.module.ts
│   └── ml-training.spec.ts               # 30+ unit tests
│
├── ml-models/
│   ├── src/
│   │   ├── conversation_analyzer.py      # Multi-task BERT model
│   │   ├── dataset.py                    # PyTorch Dataset + DataLoader
│   │   ├── trainer.py                    # Training loop + ONNX export
│   │   └── train_entry.py                # CLI entry (spawned by NestJS)
│   ├── tests/
│   │   └── test_ml_models.py             # pytest suite
│   └── requirements.txt
│
├── __mocks__/                            # Jest mocks for ONNX/transformers
├── Dockerfile.backend
├── Dockerfile.ml
├── docker-compose.yml
└── .github/workflows/ci-cd.yml
```

---

## Quick start

### 1. Prérequis

- Node.js 20+
- Python 3.11+
- Docker & Docker Compose
- (Optionnel) GPU NVIDIA avec CUDA 12.1+

### 2. Installation

```bash
# Cloner le repo
git clone https://github.com/your-org/lyd-ml.git
cd lyd-ml

# Variables d'environnement
cp .env.example .env
# → Éditer .env avec vos credentials Firebase, PostgreSQL, etc.

# Dépendances Node
npm install

# Dépendances Python
pip install -r ml-models/requirements.txt
```

### 3. Infrastructure locale

```bash
# Démarrer PostgreSQL + Redis + MinIO + MLflow + Bull Board
docker compose up postgres redis minio mlflow bull-board -d

# Vérifier santé
docker compose ps
```

### 4. Base de données

```bash
# Appliquer le schéma
psql -h localhost -U lyd -d lyd_dev -f migrations/001_ml_training_schema.sql
```

### 5. Lancer le backend

```bash
npm run start:dev
# → http://localhost:3000
# → Swagger: http://localhost:3000/api
```

### 6. Dashboard des queues

```
http://localhost:3001  (Bull Board)
```

---

## API Reference

### `POST /api/ml-training/analyze`
Analyser une conversation.

```json
{
  "text": "Jean doit finaliser le module paiement avant le 30 avril.",
  "type": "meeting",
  "language": "fr"
}
```

Réponse :
```json
{
  "conversationId": "uuid",
  "entities": [
    { "type": "person", "value": "Jean", "confidence": 0.94 },
    { "type": "date",   "value": "30 avril", "confidence": 0.91 }
  ],
  "intent": "planning",
  "intentConfidence": 0.89,
  "actions": [
    {
      "action": "create_task",
      "subject": "module paiement",
      "assignee": "Jean",
      "deadline": "30 avril",
      "priority": "high",
      "confidence": 0.87
    }
  ],
  "sentiment": 0.1,
  "priority": "high",
  "confidence": 0.90,
  "processingTimeMs": 87,
  "modelVersion": "v1.2",
  "cacheHit": false
}
```

### `POST /api/ml-training/feedback`
Corriger une prédiction.

```json
{
  "conversationId": "uuid",
  "actionId":       "uuid",
  "feedbackType":   "partially_correct",
  "userCorrection": { "intent": "decision_making" },
  "approved":       false
}
```

### `POST /api/ml-training/training/trigger`
Déclencher un réentraînement manuellement.

```json
{ "force": true }
```

### `GET /api/ml-training/status`
État du modèle en production.

### `GET /api/ml-training/training/history`
Historique des runs d'entraînement.

### `GET /api/ml-training/metrics/inference?hours=24`
Métriques de performance (p50, p95, cache hit rate).

### `DELETE /api/ml-training/conversations/:id`
Suppression RGPD (anonymisation du texte).

---

## Pipeline d'apprentissage continu

```
Feedback utilisateur
       │
       ▼
 usedForTraining = false
       │
       ▼ (count >= 500)
  Bull Queue job
       │
       ├─ 1. prepareTrainingData()  → split 80/10/10 + augmentation
       ├─ 2. trainModelPython()     → Python spawn → metrics JSON
       ├─ 3. validateModel()        → F1 > prev × 1.02, no regression
       ├─ 4. runABTest()            → agreement >= 85%
       ├─ 5. deployNewModel()       → 5% shadow → 50% canary → 100%
       └─ 6. markFeedbacksUsed()   → usedForTraining = true
```

**Seuil de déclenchement** : configurable via `TRAINING_THRESHOLD` (défaut: 500)

**Rollback automatique** : si error_rate > 5% en shadow ou canary

---

## Tests

```bash
# NestJS unit tests
npm run test
npm run test:cov

# Python tests
pytest ml-models/tests/ -v --cov=ml-models/src

# Tout
npm test && pytest ml-models/tests/
```

---

## Déploiement Docker

```bash
# Build images
docker compose build

# Lancer tout (sans GPU)
docker compose up

# Lancer avec GPU (ML worker)
docker compose --profile gpu up
```

---

## Variables d'environnement importantes

| Variable             | Défaut       | Description                                    |
|----------------------|--------------|------------------------------------------------|
| `TRAINING_THRESHOLD` | `500`        | Feedbacks minimum pour déclencher réentraînement |
| `PYTHON_WORKER_PATH` | `./ml-models`| Chemin vers les scripts Python                 |
| `MODEL_DIR`          | `./models`   | Répertoire racine des modèles ONNX             |
| `REDIS_HOST`         | `localhost`  | Hôte Redis (cache + queue)                     |
| `MLFLOW_TRACKING_URI`| `http://mlflow:5000` | Serveur MLflow                        |

---

## Monitoring

- **Bull Board** → `http://localhost:3001` — file des jobs de training
- **MLflow** → `http://localhost:5000` — métriques et artefacts des modèles
- **MinIO** → `http://localhost:9001` — stockage des modèles `.onnx`
- **`GET /api/ml-training/metrics/inference`** — latence p50/p95, cache hit rate

---

## RGPD

- `DELETE /api/ml-training/conversations/:id` anonymise le texte (remplacé par `[DELETED]`)
- Les feedbacks liés restent en base (valeur d'entraînement) mais sans le texte source
- Les inference logs ne stockent pas le contenu des conversations




# Guide complet : LiveYDream ML — De A à Z

## 1. Vue d'ensemble du système

**LiveYDream (LYD)** est un système d'analyse de conversations qui:
1. **Analyse** les conversations (réunions, emails, chats, appels)
2. **Extrait** des entités (personnes, dates, projets, etc.)
3. **Détecte** les intentions et actions à effectuer
4. **Apprend continuellement** à partir des corrections utilisateur

---

## 2. Architecture globale

```
┌─────────────────────────────────────────────────────────────────────┐
│                        UTILISATEUR                                   │
│                          │                                          │
│                          ▼                                          │
│  ┌─────────────────────────────────────────────────────────────┐   │
│  │                    NESTJS BACKEND (Port 3000)                │   │
│  │  ┌─────────────┐  ┌─────────────┐  ┌─────────────────────┐  │   │
│  │  │ Controllers │  │  Services   │  │   Bull Queue Jobs   │  │   │
│  │  │  (REST API) │  │ (Logique)   │  │  (Training async)   │  │   │
│  │  └─────────────┘  └─────────────┘  └─────────────────────┘  │   │
│  │         │                │                    │              │   │
│  │         ▼                ▼                    ▼              │   │
│  │  ┌──────────────────────────────────────────────────────┐   │   │
│  │  │              MODEL INFERENCE SERVICE                  │   │   │
│  │  │   • Charge modèle ONNX                                │   │   │
│  │  │   • Tokenize texte                                    │   │   │
│  │  │   • Prédit NER, Intent, Actions, Sentiment           │   │   │
│  │  │   • Cache Redis                                       │   │   │
│  │  └──────────────────────────────────────────────────────┘   │   │
│  └─────────────────────────────────────────────────────────────┘   │
│                          │                                          │
│         ┌────────────────┼────────────────┐                        │
│         ▼                ▼                ▼                        │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐                   │
│  │  SUPABASE  │  │   REDIS    │  │   MINIO    │                   │
│  │ (Postgres) │  │  (Cache)   │  │  (S3/ML)   │                   │
│  │            │  │  (Queue)   │  │            │                   │
│  └────────────┘  └────────────┘  └────────────┘                   │
│         │                                                       │
│         ▼                                                       │
│  ┌─────────────────────────────────────────────────────────┐    │
│  │              PYTHON ML WORKER (GPU optionnel)             │    │
│  │   • BERT multi-tâches (CamemBERT français)               │    │
│  │   • Training loop avec MLflow tracking                   │    │
│  │   • Export ONNX pour inférence                          │    │
│  └─────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────┘
```

---

## 3. Composants détaillés

### 3.1 Base de données Supabase (PostgreSQL)

**7 tables principales**:

| Table | Rôle | Exemple de données |
|-------|------|-------------------|
| `organizations` | Organisations clientes | `{id, name, slug}` |
| `users` | Utilisateurs | `{id, email, organization_id}` |
| `conversations` | Conversations analysées | `{text, type, language}` |
| `conversation_annotations` | Ground truth annoté | `{entities, intent, actions}` |
| `model_feedback` | Corrections utilisateur | `{model_prediction, user_correction}` |
| `training_runs` | Historique entraînements | `{version, metrics, status}` |
| `model_deployments` | Versions déployées | `{version, rollout_percentage, status}` |

**Flux des données**:
```
Conversation → Analyse → Prédiction → Feedback → Réentraînement
```

---

### 3.2 Services Docker locaux

| Service | Port | Rôle |
|---------|------|------|
| **Redis** | 6379 | Cache prédictions + Queue Bull |
| **MinIO** | 9000/9001 | Stockage modèles ONNX |
| **MLflow** | 5000 | Tracking expériences ML |
| **Bull Board** | 3001 | UI monitoring queues |

---

### 3.3 Backend NestJS

**Fichiers principaux**:

```
src/modules/ml-training/
├── services/
│   ├── ml-training.service.ts      # Orchestration training
│   ├── model-inference.service.ts  # Inférence ONNX
│   └── feedback.service.ts         # Gestion feedback
├── processors/
│   └── training.processor.ts       # Worker Bull queue
├── entities/                       # TypeORM models
└── dto/                            # Validation DTOs
```

---

### 3.4 Python ML

**Fichiers**:

| Fichier | Rôle |
|---------|------|
| [conversation_analyzer.py](cci:7://file:///c:/Users/HP/Downloads/files/conversation_analyzer.py:0:0-0:0) | Modèle BERT multi-tâches |
| [trainer.py](cci:7://file:///c:/Users/HP/Downloads/files/trainer.py:0:0-0:0) | Boucle d'entraînement + export ONNX |
| `dataset.py` | PyTorch DataLoader |

---

## 4. Flux de données complet (A à Z)

### Étape 1: Analyse d'une conversation

```
┌──────────────────────────────────────────────────────────────────┐
│  POST /api/ml-training/analyze                                   │
│  Body: { "text": "Jean doit finaliser le module paiement...",    │
│          "type": "meeting", "language": "fr" }                   │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. Vérifier cache Redis (hash SHA256 du texte)                 │
│     • Si hit → retour immédiat                                  │
│     • Si miss → continuer                                       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. Tokenization                                                 │
│     • Charger tokenizer CamemBERT                               │
│     • Convertir texte → input_ids [101, 1234, 5678, ..., 102]   │
│     • Padding à 512 tokens                                      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. Inférence ONNX                                               │
│     session.run({input_ids, attention_mask})                     │
│     → ner_logits [512×11], intent_logits [5], action_logits [5]  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. Décodage des sorties                                         │
│     • NER: argmax → BIO tags → entités {PERSON, DATE, PROJECT}  │
│     • Intent: argmax → "planning"                               │
│     • Actions: threshold 0.4 → "create_task"                    │
│     • Sentiment: tanh → 0.1 (légèrement positif)                │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  5. Sauvegarder en base                                          │
│     • INSERT INTO conversations (...)                           │
│     • INSERT INTO extracted_actions (...)                       │
│     • INSERT INTO ml_inference_logs (...)                       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  6. Retour à l'utilisateur                                       │
│  {                                                               │
│    "entities": [{"type":"person","value":"Jean","confidence":0.94}],│
│    "intent": "planning", "intentConfidence": 0.89,              │
│    "actions": [{"action":"create_task","assignee":"Jean"...}],  │
│    "sentiment": 0.1, "priority": "high"                         │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
```

---

### Étape 2: Feedback utilisateur

```
┌──────────────────────────────────────────────────────────────────┐
│  POST /api/ml-training/feedback                                  │
│  Body: {                                                         │
│    "conversationId": "uuid",                                     │
│    "feedbackType": "partially_correct",                          │
│    "userCorrection": { "intent": "decision_making" }             │
│  }                                                               │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. Sauvegarder le feedback                                      │
│     INSERT INTO model_feedback (                                 │
│       model_prediction = {...},  // ce que le modèle a prédit   │
│       user_correction = {...},   // correction utilisateur      │
│       used_for_training = false  // pas encore utilisé          │
│     )                                                             │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. Vérifier seuil de réentraînement                            │
│     SELECT COUNT(*) FROM model_feedback                          │
│       WHERE used_for_training = false                            │
│     → Si count >= 500 → déclencher training                     │
└──────────────────────────────────────────────────────────────────┘
```

---

### Étape 3: Réentraînement automatique

```
┌──────────────────────────────────────────────────────────────────┐
│  Bull Queue: Job "train-new-model"                               │
│  { organizationId, trainingRunId, version: "v1.1" }              │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  1. Préparer les données (prepareTrainingData)                  │
│     • Charger annotations + feedbacks                           │
│     • Augmentation: casse, ponctuation                          │
│     • Split stratifié: 80% train, 10% val, 10% test            │
│     • Écrire JSON: train.json, validation.json, test.json       │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  2. Lancer Python (trainModelPython)                            │
│     spawn('python3', [                                           │
│       'train_entry.py',                                          │
│       '--run-id', 'v1.1',                                        │
│       '--data-dir', './tmp/...',                                 │
│       '--epochs', '3'                                            │
│     ])                                                           │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  3. Training Python (trainer.py)                                │
│     • Charger modèle précédent ou CamemBERT                     │
│     • 3 epochs avec early stopping                              │
│     • Optimizer AdamW, lr=2e-5                                   │
│     • MLflow: log metrics chaque epoch                          │
│     • Sauvegarder meilleur checkpoint                           │
│     • Exporter ONNX                                             │
│     • Print JSON metrics (récupéré par NestJS)                  │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  4. Validation (validateModel)                                   │
│     • Vérifier: NER F1 >= 0.70                                   │
│     • Vérifier: Intent accuracy >= 0.75                         │
│     • Vérifier: Pas de régression > 2% vs modèle précédent      │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  5. A/B Test (runABTest)                                         │
│     • Comparer prédictions nouveau vs ancien modèle             │
│     • Agreement rate >= 85% pour valider                        │
└──────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌──────────────────────────────────────────────────────────────────┐
│  6. Déploiement progressif (deployNewModel)                     │
│     Phase 1: Shadow (5%)   → monitor 30s → error < 5% ?         │
│     Phase 2: Canary (50%)  → monitor 30s → error < 5% ?         │
│     Phase 3: Production (100%) → hot-reload ONNX               │
│     • Invalider cache Redis                                     │
│     • Marquer feedbacks used_for_training = true                │
└──────────────────────────────────────────────────────────────────┘
```

---

## 5. Modèle BERT Multi-tâches

### Architecture

```
                    ┌─────────────────────────┐
                    │     Texte en français    │
                    │  "Jean doit finir..."    │
                    └───────────┬─────────────┘
                                │
                                ▼
                    ┌─────────────────────────┐
                    │   Tokenizer CamemBERT   │
                    │  [CLS] Jean doit finir [SEP] [PAD]...
                    │  [101]  4521  2345  8765 [102]  [0]...
                    └───────────┬─────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────┐
│                      BERT ENCODER                                │
│           12 layers Transformer, 768 hidden size                │
│                                                                  │
│   Input: [batch_size, seq_len]                                  │
│   Output: [batch_size, seq_len, 768] + [batch_size, 768]        │
│              ↑ sequence_output        ↑ pooler_output (CLS)     │
└─────────────────────────────────────────────────────────────────┘
                    │                           │
                    │                           │
        ┌───────────┴───────────┐   ┌───────────┴───────────┐
        │                       │   │                       │
        ▼                       ▼   ▼                       ▼
┌─────────────┐  ┌─────────────────────────────────────────────────┐
│ NER HEAD    │  │              CLS-BASED HEADS                     │
│ (token-level)│  │                                                  │
│             │  │  ┌──────────┐ ┌──────────┐ ┌──────────┐         │
│ Linear 768  │  │  │ INTENT   │ │ ACTION   │ │SENTIMENT │         │
│   → 512     │  │  │ Linear   │ │ Linear   │ │ Linear   │         │
│   → GELU    │  │  │ 768→5    │ │ 768→256  │ │ 768→1    │         │
│   → 11      │  │  │          │ │ →GELU→5  │ │ →Tanh    │         │
│             │  │  └──────────┘ └──────────┘ └──────────┘         │
└─────────────┘  └─────────────────────────────────────────────────┘
        │
        ▼
┌─────────────────────────────────────────────────────────────────┐
│ SORTIES:                                                         │
│  • ner_logits:    [batch, 512, 11] → BIO tags                   │
│  • intent_logits: [batch, 5] → intent class                     │
│  • action_logits: [batch, 5] → action class                     │
│  • sentiment:     [batch, 1] → score [-1, 1]                    │
└─────────────────────────────────────────────────────────────────┘
```

### Labels

**NER (11 tags BIO)**:
```
O, B-PERSON, I-PERSON, B-DATE, I-DATE, 
B-PROJECT, I-PROJECT, B-AMOUNT, I-AMOUNT, B-ORG, I-ORG
```

**Intent (5 classes)**:
```
information_request, decision_making, 
problem_solving, status_update, planning
```

**Actions (5 classes)**:
```
none, create_task, create_reminder, follow_up, create_alert
```

---

## 6. Export ONNX et Inférence

### Pourquoi ONNX?

| Format | Usage | Vitesse |
|--------|-------|---------|
| PyTorch `.pt` | Training | Lent |
| ONNX `.onnx` | Inférence | Rapide (2-5×) |

### Export (Python)

```python
torch.onnx.export(
    model,
    args=(dummy_ids, dummy_mask),
    f="model.onnx",
    input_names=["input_ids", "attention_mask"],
    output_names=["ner_logits", "intent_logits", "action_logits", "sentiment_logits"],
    dynamic_axes={"input_ids": {0: "batch", 1: "seq"}}
)
```

### Inférence (NestJS)

```typescript
const session = await InferenceSession.create("model.onnx", {
    executionProviders: ["cuda", "cpu"]
});

const feeds = {
    input_ids: new Tensor("int64", inputIds, [1, 512]),
    attention_mask: new Tensor("int64", mask, [1, 512])
};

const results = await session.run(feeds);
const nerLogits = results["ner_logits"].data; // Float32Array
```

---

## 7. Démarrage en local (résumé)

### Commandes

```powershell
# 1. Démarrer les services Docker
docker compose -f docker-compose.local.yml up -d

# 2. Vérifier les services
docker compose -f docker-compose.local.yml ps

# 3. Démarrer le backend
npm run start:dev
```

### URLs

| Service | URL |
|---------|-----|
| API | http://localhost:3000 |
| Swagger | http://localhost:3000/api |
| Bull Board | http://localhost:3001 |
| MLflow | http://localhost:5000 |
| MinIO | http://localhost:9001 |

---

## 8. Schéma résumé du cycle complet

```
┌─────────────────────────────────────────────────────────────────────────┐
│                                                                         │
│    ┌──────────────┐      ┌──────────────┐      ┌──────────────┐        │
│    │ Conversation │ ──▶  │   Analyse    │ ──▶  │  Prédiction  │        │
│    │   (texte)    │      │   ONNX       │      │  (entités)   │        │
│    └──────────────┘      └──────────────┘      └──────────────┘        │
│                                  │                         │           │
│                                  ▼                         ▼           │
│                          ┌──────────────┐           ┌──────────────┐  │
│                          │    Cache     │           │   Feedback   │  │
│                          │   Redis      │           │  utilisateur │  │
│                          └──────────────┘           └──────────────┘  │
│                                                            │           │
│                                                            ▼           │
│                                                    ┌──────────────┐    │
│                                                    │  Count >= 500│    │
│                                                    │   feedbacks? │    │
│                                                    └──────┬───────┘    │
│                                                           │           │
│                                          ┌────────────────┘           │
│                                          ▼                            │
│                              ┌──────────────────────┐                 │
│                              │  Bull Queue: Training│                 │
│                              └──────────┬───────────┘                 │
│                                         │                             │
│                    ┌────────────────────┼────────────────────┐        │
│                    ▼                    ▼                    ▼        │
│           ┌─────────────┐      ┌─────────────┐      ┌─────────────┐  │
│           │ Préparation │      │  Training   │      │  Validation │  │
│           │   données   │      │   Python    │      │   Metrics   │  │
│           └─────────────┘      └─────────────┘      └─────────────┘  │
│                                                             │         │
│                                                             ▼         │
│                                              ┌──────────────────────┐ │
│                                              │  Déploiement         │ │
│                                              │  Shadow→Canary→Prod  │ │
│                                              └──────────────────────┘ │
│                                                           │           │
│                                                           ▼           │
│                                              ┌──────────────────────┐ │
│                                              │  Nouveau modèle ONNX │ │
│                                              │  en production       │ │
│                                              └──────────────────────┘ │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## 9. Points clés à retenir

1. **Supabase** = PostgreSQL hébergé avec RLS (Row Level Security)
2. **Redis** = Cache + Queue asynchrone
3. **ONNX** = Modèle optimisé pour l'inférence
4. **Bull Queue** = Jobs asynchrones pour le training
5. **MLflow** = Tracking des expériences ML
6. **Apprentissage continu** = Feedbacks → Réentraînement automatique