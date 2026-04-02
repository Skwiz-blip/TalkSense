import { OnModuleInit, OnModuleDestroy } from '@nestjs/common';
import { Repository } from 'typeorm';
import { Redis } from 'ioredis';
import { ConversationEntity } from '../../../entities/conversation.entity';
import { MLInferenceLogEntity } from '../../../entities/ml-inference-log.entity';
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
export declare class ModelInferenceService implements OnModuleInit, OnModuleDestroy {
    private conversationRepo;
    private inferenceLogRepo;
    private redis;
    private readonly logger;
    private session;
    private tokenizer;
    private currentVersion;
    private modelDir;
    private inferenceCount;
    private totalLatency;
    constructor(conversationRepo: Repository<ConversationEntity>, inferenceLogRepo: Repository<MLInferenceLogEntity>, redis: Redis);
    onModuleInit(): Promise<void>;
    onModuleDestroy(): Promise<void>;
    private loadModel;
    private loadTokenizer;
    predict(text: string, organizationId: string, userId?: string, conversationId?: string): Promise<PredictionResult>;
    private tokenize;
    private decodeNER;
    private finalizeEntity;
    private decodeIntent;
    private decodeActions;
    private decodeSentiment;
    private softmax;
    private computeOverallConfidence;
    private computePriority;
    private buildCacheKey;
    private getCached;
    private setCached;
    private logInference;
    getStats(): {
        inferenceCount: number;
        avgLatencyMs: number;
        currentVersion: string;
    };
    reloadModel(version: string): Promise<void>;
}
