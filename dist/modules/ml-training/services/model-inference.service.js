"use strict";
var __createBinding = (this && this.__createBinding) || (Object.create ? (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    var desc = Object.getOwnPropertyDescriptor(m, k);
    if (!desc || ("get" in desc ? !m.__esModule : desc.writable || desc.configurable)) {
      desc = { enumerable: true, get: function() { return m[k]; } };
    }
    Object.defineProperty(o, k2, desc);
}) : (function(o, m, k, k2) {
    if (k2 === undefined) k2 = k;
    o[k2] = m[k];
}));
var __setModuleDefault = (this && this.__setModuleDefault) || (Object.create ? (function(o, v) {
    Object.defineProperty(o, "default", { enumerable: true, value: v });
}) : function(o, v) {
    o["default"] = v;
});
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
var __importStar = (this && this.__importStar) || (function () {
    var ownKeys = function(o) {
        ownKeys = Object.getOwnPropertyNames || function (o) {
            var ar = [];
            for (var k in o) if (Object.prototype.hasOwnProperty.call(o, k)) ar[ar.length] = k;
            return ar;
        };
        return ownKeys(o);
    };
    return function (mod) {
        if (mod && mod.__esModule) return mod;
        var result = {};
        if (mod != null) for (var k = ownKeys(mod), i = 0; i < k.length; i++) if (k[i] !== "default") __createBinding(result, mod, k[i]);
        __setModuleDefault(result, mod);
        return result;
    };
})();
var __metadata = (this && this.__metadata) || function (k, v) {
    if (typeof Reflect === "object" && typeof Reflect.metadata === "function") return Reflect.metadata(k, v);
};
var __param = (this && this.__param) || function (paramIndex, decorator) {
    return function (target, key) { decorator(target, key, paramIndex); }
};
var ModelInferenceService_1;
Object.defineProperty(exports, "__esModule", { value: true });
exports.ModelInferenceService = void 0;
const common_1 = require("@nestjs/common");
const typeorm_1 = require("@nestjs/typeorm");
const typeorm_2 = require("typeorm");
const common_2 = require("@nestjs/common");
const ioredis_1 = require("ioredis");
const crypto = __importStar(require("crypto"));
const onnxruntime_node_1 = require("onnxruntime-node");
const conversation_entity_1 = require("../../../entities/conversation.entity");
const ml_inference_log_entity_1 = require("../../../entities/ml-inference-log.entity");
const redis_constants_1 = require("../../../redis/redis.constants");
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
let ModelInferenceService = ModelInferenceService_1 = class ModelInferenceService {
    constructor(conversationRepo, inferenceLogRepo, redis) {
        this.conversationRepo = conversationRepo;
        this.inferenceLogRepo = inferenceLogRepo;
        this.redis = redis;
        this.logger = new common_1.Logger(ModelInferenceService_1.name);
        this.session = null;
        this.tokenizer = null;
        this.currentVersion = 'v1.0.0';
        this.modelDir = process.env.MODEL_DIR || './models';
        this.inferenceCount = 0;
        this.totalLatency = 0;
    }
    async onModuleInit() {
        await this.loadModel();
    }
    async onModuleDestroy() {
        if (this.session) {
            await this.session.release();
        }
    }
    async loadModel() {
        const modelPath = `${this.modelDir}/${this.currentVersion}/model.onnx`;
        try {
            this.session = await onnxruntime_node_1.InferenceSession.create(modelPath, {
                executionProviders: ['cpu'],
            });
            this.logger.log(`Model loaded: ${modelPath}`);
            await this.loadTokenizer();
        }
        catch (err) {
            this.logger.error(`Failed to load model: ${err}`);
        }
    }
    async loadTokenizer() {
        try {
            const { AutoTokenizer } = await Promise.resolve().then(() => __importStar(require('@xenova/transformers')));
            this.tokenizer = await AutoTokenizer.from_pretrained(`${this.modelDir}/${this.currentVersion}/tokenizer`);
        }
        catch {
            this.logger.warn('Tokenizer not found, using fallback');
        }
    }
    async predict(text, organizationId, userId, conversationId) {
        const startTime = Date.now();
        const cacheKey = this.buildCacheKey(text, this.currentVersion);
        const cached = await this.getCached(cacheKey);
        if (cached) {
            return { ...cached, cacheHit: true, latencyMs: Date.now() - startTime };
        }
        const { inputIds, attentionMask, tokenCount } = await this.tokenize(text);
        const feeds = {
            input_ids: new onnxruntime_node_1.Tensor('int64', inputIds, [1, tokenCount]),
            attention_mask: new onnxruntime_node_1.Tensor('int64', attentionMask, [1, tokenCount]),
        };
        const results = await this.session.run(feeds);
        const entities = this.decodeNER(results['ner_logits'].data, tokenCount, text, inputIds);
        const { intent, conf } = this.decodeIntent(results['intent_logits'].data);
        const actions = this.decodeActions(results['action_logits'].data, entities);
        const sentiment = this.decodeSentiment(results['sentiment_logits'].data);
        const confidence = this.computeOverallConfidence(results['ner_logits'].data, results['intent_logits'].data, results['action_logits'].data);
        const priority = this.computePriority(sentiment, confidence, actions);
        const prediction = {
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
        await this.setCached(cacheKey, prediction);
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
    async tokenize(text) {
        if (this.tokenizer) {
            const encoded = await this.tokenizer(text, {
                padding: 'max_length',
                max_length: 512,
                truncation: true,
            });
            return {
                inputIds: Array.from(encoded.input_ids.data).map((x) => BigInt(x)),
                attentionMask: Array.from(encoded.attention_mask.data).map((x) => BigInt(x)),
                tokenCount: encoded.input_ids.size,
            };
        }
        const words = text.split(/\s+/).slice(0, 510);
        const inputIds = [BigInt(101)];
        for (const word of words) {
            inputIds.push(BigInt(word.charCodeAt(0) % 30000 + 1000));
        }
        inputIds.push(BigInt(102));
        while (inputIds.length < 512) {
            inputIds.push(BigInt(0));
        }
        const attentionMask = inputIds.map((x) => (x === BigInt(0) ? BigInt(0) : BigInt(1)));
        return { inputIds, attentionMask, tokenCount: 512 };
    }
    decodeNER(logits, tokenCount, text, inputIds) {
        const entities = [];
        const labels = [];
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
        let currentEntity = null;
        let charOffset = 0;
        for (let i = 1; i < tokenCount - 1; i++) {
            const label = NER_LABELS[labels[i]];
            if (label.startsWith('B-')) {
                if (currentEntity) {
                    entities.push(this.finalizeEntity(currentEntity, charOffset, text));
                }
                currentEntity = { type: label.slice(2), start: charOffset };
            }
            else if (label.startsWith('I-') && currentEntity) {
            }
            else {
                if (currentEntity) {
                    entities.push(this.finalizeEntity(currentEntity, charOffset, text));
                    currentEntity = null;
                }
            }
            charOffset += 4;
        }
        if (currentEntity) {
            entities.push(this.finalizeEntity(currentEntity, charOffset, text));
        }
        return entities;
    }
    finalizeEntity(entity, end, text) {
        return {
            type: entity.type.toLowerCase(),
            start: entity.start,
            end: Math.min(end, text.length),
            value: text.slice(entity.start, Math.min(end, text.length)).trim(),
            confidence: 0.85,
        };
    }
    decodeIntent(logits) {
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
    decodeActions(logits, entities) {
        const actions = [];
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
    decodeSentiment(logits) {
        const val = Math.tanh(logits[0]);
        return Math.max(-1, Math.min(1, val));
    }
    softmax(arr) {
        return arr.map((row) => {
            const rowArray = Array.from(row);
            const max = Math.max(...rowArray);
            const exps = rowArray.map((x) => Math.exp(x - max));
            const sum = exps.reduce((a, b) => a + b, 0);
            return exps.map((x) => x / sum);
        });
    }
    computeOverallConfidence(nerLogits, intentLogits, actionLogits) {
        const nerConf = this.softmax([nerLogits])[0].reduce((a, b) => Math.max(a, b), 0);
        const intentConf = this.softmax([intentLogits])[0].reduce((a, b) => Math.max(a, b), 0);
        const actionConf = this.softmax([actionLogits])[0].reduce((a, b) => Math.max(a, b), 0);
        return (nerConf + intentConf + actionConf) / 3;
    }
    computePriority(sentiment, confidence, actions) {
        if (confidence >= PRIORITY_THRESHOLDS.critical)
            return 'critical';
        if (confidence >= PRIORITY_THRESHOLDS.high)
            return 'high';
        if (confidence >= PRIORITY_THRESHOLDS.medium)
            return 'medium';
        return 'low';
    }
    buildCacheKey(text, version) {
        const hash = crypto.createHash('sha256').update(text).digest('hex');
        return `lyd:pred:${version}:${hash}`;
    }
    async getCached(key) {
        try {
            const cached = await this.redis.get(key);
            return cached ? JSON.parse(cached) : null;
        }
        catch {
            return null;
        }
    }
    async setCached(key, value) {
        try {
            await this.redis.setex(key, 3600, JSON.stringify(value));
        }
        catch (err) {
            this.logger.warn(`Cache set failed: ${err}`);
        }
    }
    async logInference(data) {
        try {
            const log = this.inferenceLogRepo.create(data);
            await this.inferenceLogRepo.save(log);
            this.inferenceCount++;
            this.totalLatency += data.inferenceTimeMs;
        }
        catch (err) {
            this.logger.warn(`Failed to log inference: ${err}`);
        }
    }
    getStats() {
        return {
            inferenceCount: this.inferenceCount,
            avgLatencyMs: this.inferenceCount > 0 ? this.totalLatency / this.inferenceCount : 0,
            currentVersion: this.currentVersion,
        };
    }
    async reloadModel(version) {
        this.currentVersion = version;
        await this.loadModel();
        this.logger.log(`Model reloaded: ${version}`);
    }
};
exports.ModelInferenceService = ModelInferenceService;
exports.ModelInferenceService = ModelInferenceService = ModelInferenceService_1 = __decorate([
    (0, common_1.Injectable)(),
    __param(0, (0, typeorm_1.InjectRepository)(conversation_entity_1.ConversationEntity)),
    __param(1, (0, typeorm_1.InjectRepository)(ml_inference_log_entity_1.MLInferenceLogEntity)),
    __param(2, (0, common_2.Inject)(redis_constants_1.REDIS_CLIENT)),
    __metadata("design:paramtypes", [typeorm_2.Repository,
        typeorm_2.Repository,
        ioredis_1.Redis])
], ModelInferenceService);
//# sourceMappingURL=model-inference.service.js.map