import { Injectable, Logger, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Inject } from '@nestjs/common';
import { Redis } from 'ioredis';
import * as crypto from 'crypto';
import { InferenceSession, Tensor } from 'onnxruntime-node';

import { ConversationEntity } from '../entities/conversation.entity';
import { MLInferenceLogEntity } from '../entities/ml-inference-log.entity';
import { REDIS_CLIENT } from '../../redis/redis.constants';

// ── Types ──────────────────────────────────────────────────
export interface EntityResult {
  type: string;
  start: number;
  end: number;
  value: string;
  confidence: number;
}

export interface ActionResult {
  action: string;
  subject: string;
  assignee?: string;
  deadline?: string;
  priority: string;
  confidence: number;
}

export interface PredictionResult {
  entities: EntityResult[];
  intent: string;
  intentConfidence: number;
  actions: ActionResult[];
  sentiment: number;
  priority: string;
  confidence: number;
  latencyMs: number;
  cacheHit: boolean;
}

// ── BIO tag scheme for NER ─────────────────────────────────
const NER_LABELS = [
  'O',
  'B-PERSON', 'I-PERSON',
  'B-DATE',   'I-DATE',
  'B-PROJECT','I-PROJECT',
  'B-AMOUNT', 'I-AMOUNT',
  'B-ORG',    'I-ORG',
];

const INTENT_LABELS = [
  'information_request',
  'decision_making',
  'problem_solving',
  'status_update',
  'planning',
];

const ACTION_LABELS = [
  'none',
  'create_task',
  'create_reminder',
  'follow_up',
  'create_alert',
];

const PRIORITY_THRESHOLDS = { critical: 0.85, high: 0.65, medium: 0.40, low: 0 };

// ── Service ────────────────────────────────────────────────
@Injectable()
export class ModelInferenceService implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(ModelInferenceService.name);
  private session: InferenceSession | null = null;
  private tokenizer: any = null;           // HuggingFace tokenizer (loaded via transformers)
  private currentVersion = 'current';
  private readonly CACHE_TTL_SECONDS = 3600;
  private readonly MAX_SEQ_LENGTH    = 512;

  constructor(
    @InjectRepository(ConversationEntity)
    private readonly conversationRepo: Repository<ConversationEntity>,
    @InjectRepository(MLInferenceLogEntity)
    private readonly inferenceLogRepo: Repository<MLInferenceLogEntity>,
    @Inject(REDIS_CLIENT)
    private readonly redis: Redis,
  ) {}

  // ── Lifecycle ──────────────────────────────────────────
  async onModuleInit(): Promise<void> {
    await this.loadModel();
    await this.loadTokenizer();
  }

  async onModuleDestroy(): Promise<void> {
    if (this.session) {
      await this.session.release?.();
    }
  }

  // ── Model loading ──────────────────────────────────────
  private async loadModel(modelPath?: string): Promise<void> {
    const path = modelPath ?? `./models/${this.currentVersion}/model.onnx`;
    try {
      this.session = await InferenceSession.create(path, {
        executionProviders: ['cuda', 'cpu'],
        graphOptimizationLevel: 'all',
        enableCpuMemArena: true,
      });
      this.logger.log(`✅ ONNX model loaded: ${path}`);
      this.logger.log(`   Inputs:  ${this.session.inputNames.join(', ')}`);
      this.logger.log(`   Outputs: ${this.session.outputNames.join(', ')}`);
    } catch (error) {
      this.logger.error(`❌ Failed to load ONNX model at ${path}:`, error);
      throw error;
    }
  }

  private async loadTokenizer(): Promise<void> {
    try {
      // Using @xenova/transformers for Node.js tokenization
      const { AutoTokenizer } = await import('@xenova/transformers');
      this.tokenizer = await AutoTokenizer.from_pretrained(
        `./models/${this.currentVersion}/tokenizer`,
      );
      this.logger.log('✅ Tokenizer loaded');
    } catch (error) {
      this.logger.warn('⚠️  Tokenizer load failed, falling back to simple tokenizer:', error);
      this.tokenizer = null;
    }
  }

  // ── Hot-reload model after deployment ──────────────────
  async reloadModel(version: string): Promise<void> {
    this.logger.log(`🔄 Reloading model to version ${version}`);
    const oldSession = this.session;
    await this.loadModel(`./models/${version}/model.onnx`);
    this.currentVersion = version;
    // Invalidate all cached predictions
    await this.invalidateCache();
    if (oldSession) {
      await oldSession.release?.();
    }
    this.logger.log(`✅ Model reloaded to ${version}`);
  }

  // ── Main predict ───────────────────────────────────────
  async predict(
    text: string,
    organizationId: string,
    userId?: string,
    conversationId?: string,
  ): Promise<PredictionResult> {
    if (!this.session) {
      throw new Error('Model not initialized. Call onModuleInit first.');
    }

    const wallStart = Date.now();
    const cacheKey  = this.buildCacheKey(text, this.currentVersion);

    // 1. Cache lookup
    const cached = await this.getCached(cacheKey);
    if (cached) {
      await this.logInference({
        organizationId, userId, conversationId,
        inputTokens: 0, inferenceTimeMs: Date.now() - wallStart,
        predictionsCount: cached.entities.length + cached.actions.length,
        confidenceAvg: cached.confidence,
        cacheHit: true,
      });
      return { ...cached, cacheHit: true };
    }

    // 2. Tokenize
    const { inputIds, attentionMask, tokenCount } = await this.tokenize(text);

    // 3. Run ONNX inference
    const inferenceStart = Date.now();
    const feeds = {
      input_ids:      new Tensor('int64', inputIds,      [1, tokenCount]),
      attention_mask: new Tensor('int64', attentionMask, [1, tokenCount]),
    };

    const results = await this.session.run(feeds);
    const inferenceMs = Date.now() - inferenceStart;

    // 4. Decode outputs
    const nerLogits     = results['ner_logits']?.data     as Float32Array;
    const intentLogits  = results['intent_logits']?.data  as Float32Array;
    const actionLogits  = results['action_logits']?.data  as Float32Array;
    const sentimentData = results['sentiment_logits']?.data as Float32Array;

    const entities            = this.decodeNER(nerLogits, tokenCount, text, inputIds);
    const { intent, conf: ic} = this.decodeIntent(intentLogits);
    const actions             = this.decodeActions(actionLogits, entities);
    const sentiment           = this.decodeSentiment(sentimentData);
    const confidence          = this.computeOverallConfidence(nerLogits, intentLogits, actionLogits);
    const priority            = this.computePriority(sentiment, confidence, actions);

    const prediction: PredictionResult = {
      entities,
      intent,
      intentConfidence: ic,
      actions,
      sentiment,
      priority,
      confidence,
      latencyMs: Date.now() - wallStart,
      cacheHit: false,
    };

    // 5. Store in cache
    await this.setCached(cacheKey, prediction);

    // 6. Log
    await this.logInference({
      organizationId, userId, conversationId,
      inputTokens: tokenCount,
      inferenceTimeMs: inferenceMs,
      predictionsCount: entities.length + actions.length,
      confidenceAvg: confidence,
      cacheHit: false,
    });

    return prediction;
  }

  // ── Batch predict ──────────────────────────────────────
  async predictBatch(
    texts: string[],
    organizationId: string,
    userId?: string,
  ): Promise<PredictionResult[]> {
    return Promise.all(
      texts.map((text) => this.predict(text, organizationId, userId)),
    );
  }

  // ── Tokenization ───────────────────────────────────────
  private async tokenize(text: string): Promise<{
    inputIds: BigInt64Array;
    attentionMask: BigInt64Array;
    tokenCount: number;
  }> {
    if (this.tokenizer) {
      const encoded = await this.tokenizer(text, {
        max_length: this.MAX_SEQ_LENGTH,
        truncation: true,
        padding: 'max_length',
        return_tensors: 'np',
      });
      const ids  = Array.from(encoded.input_ids.data as number[]);
      const mask = Array.from(encoded.attention_mask.data as number[]);
      return {
        inputIds:      BigInt64Array.from(ids.map(BigInt)),
        attentionMask: BigInt64Array.from(mask.map(BigInt)),
        tokenCount:    ids.length,
      };
    }

    // Fallback: whitespace tokenizer with [CLS]=101, [SEP]=102, [PAD]=0
    const tokens  = text.toLowerCase().split(/\s+/).slice(0, this.MAX_SEQ_LENGTH - 2);
    const ids     = [101n, ...tokens.map((_, i) => BigInt(i + 1000)), 102n];
    const padded  = [
      ...ids,
      ...Array(this.MAX_SEQ_LENGTH - ids.length).fill(0n),
    ].slice(0, this.MAX_SEQ_LENGTH);
    const mask = padded.map((v) => (v !== 0n ? 1n : 0n));
    return {
      inputIds:      BigInt64Array.from(padded),
      attentionMask: BigInt64Array.from(mask),
      tokenCount:    this.MAX_SEQ_LENGTH,
    };
  }

  // ── NER decoding (BIO scheme) ──────────────────────────
  private decodeNER(
    logits: Float32Array,
    seqLen: number,
    text: string,
    _inputIds: BigInt64Array,
  ): EntityResult[] {
    if (!logits || logits.length === 0) return [];

    const numLabels = NER_LABELS.length;
    const entities: EntityResult[] = [];
    let currentEntity: Partial<EntityResult> | null = null;
    let charOffset = 0;

    for (let i = 0; i < seqLen; i++) {
      // Find argmax for token i
      let maxVal = -Infinity;
      let maxIdx = 0;
      for (let j = 0; j < numLabels; j++) {
        const val = logits[i * numLabels + j];
        if (val > maxVal) { maxVal = val; maxIdx = j; }
      }

      const label      = NER_LABELS[maxIdx] ?? 'O';
      const confidence = this.softmaxMax(logits, i * numLabels, numLabels);

      if (label.startsWith('B-')) {
        if (currentEntity) entities.push(currentEntity as EntityResult);
        const entityType = label.slice(2).toLowerCase();
        currentEntity = {
          type: entityType, start: charOffset,
          end: charOffset, value: '', confidence,
        };
      } else if (label.startsWith('I-') && currentEntity) {
        currentEntity.end = charOffset;
        currentEntity.confidence = Math.min(currentEntity.confidence!, confidence);
      } else {
        if (currentEntity) {
          entities.push(currentEntity as EntityResult);
          currentEntity = null;
        }
      }
      // Approximate char offset (real impl uses offset_mapping from tokenizer)
      charOffset += 4;
    }
    if (currentEntity) entities.push(currentEntity as EntityResult);

    // Re-extract values from original text using offsets
    return entities.map((e) => ({
      ...e,
      value: text.slice(Math.max(0, e.start), Math.min(text.length, e.end)) || e.value,
    })).filter((e) => e.confidence > 0.5);
  }

  // ── Intent decoding ────────────────────────────────────
  private decodeIntent(logits: Float32Array): { intent: string; conf: number } {
    if (!logits || logits.length === 0) {
      return { intent: 'information_request', conf: 0.5 };
    }
    const probs = this.softmax(Array.from(logits));
    let maxIdx = 0;
    probs.forEach((p, i) => { if (p > probs[maxIdx]) maxIdx = i; });
    return {
      intent: INTENT_LABELS[maxIdx] ?? 'information_request',
      conf: probs[maxIdx],
    };
  }

  // ── Action decoding ────────────────────────────────────
  private decodeActions(
    logits: Float32Array,
    entities: EntityResult[],
  ): ActionResult[] {
    if (!logits || logits.length === 0) return [];

    const probs   = this.softmax(Array.from(logits));
    const actions: ActionResult[] = [];

    probs.forEach((prob, idx) => {
      if (idx === 0 || prob < 0.4) return; // skip 'none'
      const actionType = ACTION_LABELS[idx];
      if (!actionType) return;

      const person  = entities.find((e) => e.type === 'person');
      const date    = entities.find((e) => e.type === 'date');
      const project = entities.find((e) => e.type === 'project');

      actions.push({
        action:    actionType,
        subject:   project?.value ?? 'Action à définir',
        assignee:  person?.value,
        deadline:  date?.value,
        priority:  prob > 0.85 ? 'critical' : prob > 0.65 ? 'high' : prob > 0.45 ? 'medium' : 'low',
        confidence: prob,
      });
    });

    return actions;
  }

  // ── Sentiment decoding ─────────────────────────────────
  private decodeSentiment(logits: Float32Array): number {
    if (!logits || logits.length === 0) return 0;
    // Clamp tanh output to [-1, 1]
    const raw = logits[0];
    return Math.max(-1, Math.min(1, Math.tanh(raw)));
  }

  // ── Priority ───────────────────────────────────────────
  private computePriority(
    sentiment: number,
    confidence: number,
    actions: ActionResult[],
  ): string {
    const urgencyScore = (1 - sentiment) * 0.3 + confidence * 0.3
      + (actions.length > 0 ? 0.4 : 0);
    for (const [level, thresh] of Object.entries(PRIORITY_THRESHOLDS)) {
      if (urgencyScore >= thresh) return level;
    }
    return 'low';
  }

  // ── Overall confidence ─────────────────────────────────
  private computeOverallConfidence(
    nerLogits:    Float32Array,
    intentLogits: Float32Array,
    actionLogits: Float32Array,
  ): number {
    const scores: number[] = [];
    if (nerLogits?.length)    scores.push(this.softmaxMax(nerLogits, 0, NER_LABELS.length));
    if (intentLogits?.length) scores.push(this.softmaxMax(intentLogits, 0, INTENT_LABELS.length));
    if (actionLogits?.length) scores.push(this.softmaxMax(actionLogits, 0, ACTION_LABELS.length));
    return scores.length ? scores.reduce((a, b) => a + b) / scores.length : 0.5;
  }

  // ── Math helpers ───────────────────────────────────────
  private softmax(values: number[]): number[] {
    const max  = Math.max(...values);
    const exps = values.map((v) => Math.exp(v - max));
    const sum  = exps.reduce((a, b) => a + b, 0);
    return exps.map((e) => e / sum);
  }

  private softmaxMax(
    logits: Float32Array,
    offset: number,
    numLabels: number,
  ): number {
    const slice  = Array.from(logits.slice(offset, offset + numLabels));
    const probs  = this.softmax(slice);
    return Math.max(...probs);
  }

  // ── Cache ──────────────────────────────────────────────
  private buildCacheKey(text: string, version: string): string {
    const hash = crypto.createHash('sha256').update(`${version}:${text}`).digest('hex');
    return `lyd:pred:${version}:${hash}`;
  }

  private async getCached(key: string): Promise<PredictionResult | null> {
    try {
      const raw = await this.redis.get(key);
      return raw ? (JSON.parse(raw) as PredictionResult) : null;
    } catch {
      return null;
    }
  }

  private async setCached(key: string, value: PredictionResult): Promise<void> {
    try {
      await this.redis.setex(key, this.CACHE_TTL_SECONDS, JSON.stringify(value));
    } catch (err) {
      this.logger.warn('Cache write failed:', err);
    }
  }

  private async invalidateCache(): Promise<void> {
    try {
      const keys = await this.redis.keys('lyd:pred:*');
      if (keys.length) await this.redis.del(...keys);
      this.logger.log(`🗑️  Cache invalidated (${keys.length} entries)`);
    } catch (err) {
      this.logger.warn('Cache invalidation failed:', err);
    }
  }

  // ── Inference logging ──────────────────────────────────
  private async logInference(params: {
    organizationId: string;
    userId?: string;
    conversationId?: string;
    inputTokens: number;
    inferenceTimeMs: number;
    predictionsCount: number;
    confidenceAvg: number;
    cacheHit: boolean;
  }): Promise<void> {
    try {
      const log = this.inferenceLogRepo.create({
        organizationId:   params.organizationId,
        modelVersion:     this.currentVersion,
        userId:           params.userId,
        conversationId:   params.conversationId,
        inputTokens:      params.inputTokens,
        inferenceTimeMs:  params.inferenceTimeMs,
        predictionsCount: params.predictionsCount,
        confidenceAvg:    params.confidenceAvg,
        cacheHit:         params.cacheHit,
      });
      await this.inferenceLogRepo.save(log);
    } catch (err) {
      this.logger.warn('Failed to persist inference log:', err);
    }
  }

  // ── Monitoring ─────────────────────────────────────────
  async getInferenceStats(organizationId: string, hours = 24) {
    const since = new Date(Date.now() - hours * 3_600_000);
    const rows = await this.inferenceLogRepo
      .createQueryBuilder('l')
      .select([
        'AVG(l.inferenceTimeMs) AS avg_ms',
        'PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY l.inferenceTimeMs) AS p50_ms',
        'PERCENTILE_CONT(0.95) WITHIN GROUP (ORDER BY l.inferenceTimeMs) AS p95_ms',
        'COUNT(*) AS total',
        'SUM(CASE WHEN l.cacheHit THEN 1 ELSE 0 END)::float / COUNT(*) AS cache_rate',
        'AVG(l.confidenceAvg) AS avg_confidence',
      ])
      .where('l.organizationId = :orgId', { orgId: organizationId })
      .andWhere('l.createdAt >= :since', { since })
      .getRawOne();
    return rows;
  }

  get version(): string { return this.currentVersion; }
}
