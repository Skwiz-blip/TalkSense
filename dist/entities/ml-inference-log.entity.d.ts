export declare class MLInferenceLogEntity {
    id: string;
    organizationId: string;
    modelVersion: string;
    inputTokens: number;
    outputTokens: number;
    inferenceTimeMs: number;
    memoryUsedMb: number;
    predictionsCount: number;
    confidenceAvg: number;
    cacheHit: boolean;
    userId: string;
    conversationId: string;
    createdAt: Date;
}
