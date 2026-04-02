import { Injectable, Logger, OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository } from 'typeorm';
import { Inject } from '@nestjs/common';
import { Redis } from 'ioredis';
import * as crypto from 'crypto';
import { InferenceSession, Tensor } from 'onnxruntime-node';

import { ConversationEntity } from '../../../entities/conversation.entity';
import { MLInferenceLogEntity } from '../../../entities/ml-inference-log.entity';
import { REDIS_CLIENT } from '../../../redis/redis.constants';

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
  'B-DATE', 'I-DATE',
  'B-PROJECT', 'I-PROJECT',
  'B-AMOUNT', 'I-AMOUNT',
  'B-ORG', 'I-ORG',
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

const PRIORITY_THRESHOLDS = {
  critical: 0.85,
  high: 0.65,
  medium: 0.40,
  low: 0,
};

// ── Service ────────────────────────────────────────────────
@Injectable()
export class ModelInferenceService implements OnModuleInit, OnModuleDestroy {
  private readonly logger = new Logger(ModelInferenceService.name);
  private session: InferenceSession | null = null;
  private tokenizer: any = null;
  private currentVersion = 'v1.0.0';
  private modelDir = process.env.MODEL_DIR || './models';
  private inferenceCount = 0;
  private totalLatency = 0;

  constructor(
    @InjectRepository(ConversationEntity)
    private conversationRepo: Repository<ConversationEntity>,
    @InjectRepository(MLInferenceLogEntity)
    private inferenceLogRepo: Repository<MLInferenceLogEntity>,
    @Inject(REDIS_CLIENT)
    private redis: Redis,
  ) { }

  async onModuleInit() {
    await this.loadModel();
  }

  async onModuleDestroy() {
    if (this.session) {
      await this.session.release();
    }
  }

  // ── Model Loading ────────────────────────────────────────
  private async loadModel() {
    const modelPath = `${this.modelDir}/${this.currentVersion}/model.onnx`;
    try {
      this.session = await InferenceSession.create(modelPath, {
        executionProviders: ['cpu'],
      });
      this.logger.log(`Model loaded: ${modelPath}`);
      await this.loadTokenizer();
    } catch (err) {
      this.logger.error(`Failed to load model: ${err}`);
    }
  }

  private async loadTokenizer() {
    try {
      const { AutoTokenizer } = await import('@xenova/transformers');
      this.tokenizer = await AutoTokenizer.from_pretrained(
        `${this.modelDir}/${this.currentVersion}/tokenizer`
      );
    } catch {
      this.logger.warn('Tokenizer not found, using fallback');
    }
  }

  // ── Prediction ───────────────────────────────────────────
  async predict(
    text: string,
    organizationId: string,
    userId?: string,
    conversationId?: string,
  ): Promise<PredictionResult> {
    const startTime = Date.now();
    const cacheKey = this.buildCacheKey(text, this.currentVersion);

    // Check cache
    const cached = await this.getCached(cacheKey);
    if (cached) {
      return { ...cached, cacheHit: true, latencyMs: Date.now() - startTime };
    }

    // Tokenize
    const { inputIds, attentionMask, tokenCount } = await this.tokenize(text);

    // Run inference
    const feeds = {
      input_ids: new Tensor('int64', inputIds, [1, tokenCount]),
      attention_mask: new Tensor('int64', attentionMask, [1, tokenCount]),
    };

    const results = await this.session!.run(feeds);

    // Decode outputs
    const entities = this.decodeNER(
      results['ner_logits'].data as Float32Array,
      tokenCount,
      text,
      inputIds,
    );
    const { intent, conf } = this.decodeIntent(
      results['intent_logits'].data as Float32Array,
    );
    const actions = this.decodeActions(
      results['action_logits'].data as Float32Array,
      entities,
    );
    const sentiment = this.decodeSentiment(
      results['sentiment_logits'].data as Float32Array,
    );
    const confidence = this.computeOverallConfidence(
      results['ner_logits'].data as Float32Array,
      results['intent_logits'].data as Float32Array,
      results['action_logits'].data as Float32Array,
    );
    const priority = this.computePriority(sentiment, confidence, actions);

    const prediction: PredictionResult = {
      entities,
      intent,
      intentConfidence: conf,
      actions,
      sentiment,
      priority,
      confidence,
      latencyMs: Date.now() - startTime,
      cacheHit: false,
    };

    // Cache result
    await this.setCached(cacheKey, prediction);

    // Log inference
    await this.logInference({
      organizationId,
      modelVersion: this.currentVersion,
      inputTokens: tokenCount,
      inferenceTimeMs: prediction.latencyMs,
      confidenceAvg: confidence,
      cacheHit: false,
      userId,
      conversationId,
    });

    return prediction;
  }

  // ── Tokenization ─────────────────────────────────────────
  private async tokenize(text: string): Promise<{
    inputIds: bigint[];
    attentionMask: bigint[];
    tokenCount: number;
  }> {
    if (this.tokenizer) {
      const encoded = await this.tokenizer(text, {
        padding: 'max_length',
        max_length: 512,
        truncation: true,
      });
      return {
        inputIds: Array.from(encoded.input_ids.data).map((x: number) => BigInt(x)),
        attentionMask: Array.from(encoded.attention_mask.data).map((x: number) => BigInt(x)),
        tokenCount: encoded.input_ids.size,
      };
    }

    // Fallback: whitespace tokenization
    const words = text.split(/\s+/).slice(0, 510);
    const inputIds = [BigInt(101)]; // [CLS]
    for (const word of words) {
      inputIds.push(BigInt(word.charCodeAt(0) % 30000 + 1000));
    }
    inputIds.push(BigInt(102)); // [SEP]
    while (inputIds.length < 512) {
      inputIds.push(BigInt(0)); // [PAD]
    }
    const attentionMask = inputIds.map((x) => (x === BigInt(0) ? BigInt(0) : BigInt(1)));
    return { inputIds, attentionMask, tokenCount: 512 };
  }

  // ── Decoding ─────────────────────────────────────────────
  private decodeNER(
    logits: Float32Array,
    tokenCount: number,
    text: string,
    inputIds: bigint[],
  ): EntityResult[] {
    const entities: EntityResult[] = [];
    const labels: number[] = [];

    for (let i = 0; i < tokenCount; i++) {
      const start = i * NER_LABELS.length;
      let maxIdx = 0;
      let maxVal = -Infinity;
      for (let j = 0; j < NER_LABELS.length; j++) {
        if (logits[start + j] > maxVal) {
          maxVal = logits[start + j];
          maxIdx = j;
        }
      }
      labels.push(maxIdx);
    }

    // BIO decoding
    let currentEntity: { type: string; start: number } | null = null;
    let charOffset = 0;

    for (let i = 1; i < tokenCount - 1; i++) {
      const label = NER_LABELS[labels[i]];
      if (label.startsWith('B-')) {
        if (currentEntity) {
          entities.push(this.finalizeEntity(currentEntity, charOffset, text));
        }
        currentEntity = { type: label.slice(2), start: charOffset };
      } else if (label.startsWith('I-') && currentEntity) {
        // Continue entity
      } else {
        if (currentEntity) {
          entities.push(this.finalizeEntity(currentEntity, charOffset, text));
          currentEntity = null;
        }
      }
      charOffset += 4; // Approximate
    }

    if (currentEntity) {
      entities.push(this.finalizeEntity(currentEntity, charOffset, text));
    }

    return entities;
  }

  private finalizeEntity(
    entity: { type: string; start: number },
    end: number,
    text: string,
  ): EntityResult {
    return {
      type: entity.type.toLowerCase(),
      start: entity.start,
      end: Math.min(end, text.length),
      value: text.slice(entity.start, Math.min(end, text.length)).trim(),
      confidence: 0.85,
    };
  }

  private decodeIntent(logits: Float32Array): { intent: string; conf: number } {
    let maxIdx = 0;
    let maxVal = -Infinity;
    for (let i = 0; i < INTENT_LABELS.length; i++) {
      if (logits[i] > maxVal) {
        maxVal = logits[i];
        maxIdx = i;
      }
    }
    const conf = this.softmax([logits])[0][maxIdx];
    return { intent: INTENT_LABELS[maxIdx], conf };
  }

  private decodeActions(logits: Float32Array, entities: EntityResult[]): ActionResult[] {
    const actions: ActionResult[] = [];
    const probs = this.softmax([logits])[0];

    for (let i = 1; i < ACTION_LABELS.length; i++) {
      if (probs[i] > 0.4) {
        actions.push({
          action: ACTION_LABELS[i],
          subject: entities[0]?.value || 'Unknown',
          priority: 'medium',
          confidence: probs[i],
        });
      }
    }

    return actions;
  }

  private decodeSentiment(logits: Float32Array): number {
    const val = Math.tanh(logits[0]);
    return Math.max(-1, Math.min(1, val));
  }

  // ── Helpers ──────────────────────────────────────────────
  private softmax(arr: Float32Array[]): number[][] {
    return arr.map((row) => {
      const rowArray = Array.from(row);
      const max = Math.max(...rowArray);
      const exps = rowArray.map((x) => Math.exp(x - max));
      const sum = exps.reduce((a, b) => a + b, 0);
      return exps.map((x) => x / sum);
    });
  }

  private computeOverallConfidence(
    nerLogits: Float32Array,
    intentLogits: Float32Array,
    actionLogits: Float32Array,
  ): number {
    const nerConf = this.softmax([nerLogits])[0].reduce((a, b) => Math.max(a, b), 0);
    const intentConf = this.softmax([intentLogits])[0].reduce((a, b) => Math.max(a, b), 0);
    const actionConf = this.softmax([actionLogits])[0].reduce((a, b) => Math.max(a, b), 0);
    return (nerConf + intentConf + actionConf) / 3;
  }

  private computePriority(
    sentiment: number,
    confidence: number,
    actions: ActionResult[],
  ): string {
    if (confidence >= PRIORITY_THRESHOLDS.critical) return 'critical';
    if (confidence >= PRIORITY_THRESHOLDS.high) return 'high';
    if (confidence >= PRIORITY_THRESHOLDS.medium) return 'medium';
    return 'low';
  }

  // ── Cache ────────────────────────────────────────────────
  private buildCacheKey(text: string, version: string): string {
    const hash = crypto.createHash('sha256').update(text).digest('hex');
    return `lyd:pred:${version}:${hash}`;
  }

  private async getCached(key: string): Promise<PredictionResult | null> {
    try {
      const cached = await this.redis.get(key);
      return cached ? JSON.parse(cached) : null;
    } catch {
      return null;
    }
  }

  private async setCached(key: string, value: PredictionResult): Promise<void> {
    try {
      await this.redis.setex(key, 3600, JSON.stringify(value));
    } catch (err) {
      this.logger.warn(`Cache set failed: ${err}`);
    }
  }

  // ── Logging ──────────────────────────────────────────────
  private async logInference(data: {
    organizationId: string;
    modelVersion: string;
    inputTokens?: number;
    inferenceTimeMs: number;
    confidenceAvg?: number;
    cacheHit: boolean;
    userId?: string;
    conversationId?: string;
  }): Promise<void> {
    try {
      const log = this.inferenceLogRepo.create(data);
      await this.inferenceLogRepo.save(log);
      this.inferenceCount++;
      this.totalLatency += data.inferenceTimeMs;
    } catch (err) {
      this.logger.warn(`Failed to log inference: ${err}`);
    }
  }

  // ── Stats ────────────────────────────────────────────────
  getStats() {
    return {
      inferenceCount: this.inferenceCount,
      avgLatencyMs: this.inferenceCount > 0 ? this.totalLatency / this.inferenceCount : 0,
      currentVersion: this.currentVersion,
    };
  }

  // ── Reload ───────────────────────────────────────────────
  async reloadModel(version: string) {
    this.currentVersion = version;
    await this.loadModel();
    this.logger.log(`Model reloaded: ${version}`);
  }
}
