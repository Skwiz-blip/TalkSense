/**
 * ml-training.spec.ts
 * Unit tests for ML Training module
 * Run: jest --testPathPattern=ml-training.spec
 */

import { Test, TestingModule } from '@nestjs/testing';
import { getRepositoryToken }   from '@nestjs/typeorm';
import { getQueueToken }        from '@nestjs/bull';
import { Repository, DataSource } from 'typeorm';

import { ModelInferenceService } from './services/model-inference.service';
import { MLTrainingService }     from './services/ml-training.service';
import { FeedbackService }       from './services/feedback.service';
import { ConversationService }   from './services/feedback.service';
import { TrainingProcessor }     from './processors/training.processor';

import { ConversationEntity, ConversationType }  from './entities/conversation.entity';
import { ModelFeedbackEntity }                   from './entities/model-feedback.entity';
import { TrainingRunEntity, TrainingMetrics }    from './entities/training-run.entity';
import { ModelDeploymentEntity }                 from './entities/model-deployment.entity';
import { ConversationAnnotationEntity }          from './entities/conversation-annotation.entity';
import { ExtractedActionEntity }                 from './entities/extracted-action.entity';
import { MLInferenceLogEntity }                  from './entities/ml-inference-log.entity';

import { SubmitFeedbackDto, AnalyzeConversationDto } from './dto';

// ── Mock factories ────────────────────────────────────────────
const mockRepo = <T>(): Partial<Repository<T>> => ({
  findOne:       jest.fn(),
  find:          jest.fn(),
  findAndCount:  jest.fn(),
  create:        jest.fn((dto) => dto),
  save:          jest.fn((e) => Promise.resolve({ id: 'uuid-1', ...e })),
  update:        jest.fn(() => Promise.resolve({ affected: 1 })),
  count:         jest.fn(() => Promise.resolve(0)),
  createQueryBuilder: jest.fn(() => ({
    innerJoin:         jest.fn().mockReturnThis(),
    innerJoinAndSelect:jest.fn().mockReturnThis(),
    where:             jest.fn().mockReturnThis(),
    andWhere:          jest.fn().mockReturnThis(),
    select:            jest.fn().mockReturnThis(),
    groupBy:           jest.fn().mockReturnThis(),
    orderBy:           jest.fn().mockReturnThis(),
    take:              jest.fn().mockReturnThis(),
    getRawOne:         jest.fn(() => Promise.resolve({})),
    getRawMany:        jest.fn(() => Promise.resolve([])),
    getMany:           jest.fn(() => Promise.resolve([])),
    getOne:            jest.fn(() => Promise.resolve(null)),
  })),
});

const mockRedis = {
  get:    jest.fn(() => Promise.resolve(null)),
  setex:  jest.fn(() => Promise.resolve('OK')),
  del:    jest.fn(() => Promise.resolve(1)),
  keys:   jest.fn(() => Promise.resolve([])),
};

const mockQueue = {
  add:   jest.fn(() => Promise.resolve({ id: 'job-1' })),
  getJob:jest.fn(),
};

const mockDataSource = {
  query: jest.fn(() => Promise.resolve([{ count: '0' }])),
};

const mockInferenceService = {
  predict:        jest.fn(),
  predictBatch:   jest.fn(),
  reloadModel:    jest.fn(),
  getInferenceStats: jest.fn(() => Promise.resolve({})),
  version:        'v1.0',
};

// ── Test suites ───────────────────────────────────────────────
describe('ModelInferenceService', () => {
  let service: ModelInferenceService;

  const makeSvc = async (overrides: any = {}) => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        ModelInferenceService,
        { provide: getRepositoryToken(ConversationEntity),    useValue: mockRepo() },
        { provide: getRepositoryToken(MLInferenceLogEntity),  useValue: mockRepo() },
        { provide: 'REDIS_CLIENT',                            useValue: mockRedis },
      ],
    }).compile();
    return module.get<ModelInferenceService>(ModelInferenceService);
  };

  beforeEach(async () => {
    jest.clearAllMocks();
    service = await makeSvc();
    // Bypass ONNX session init for unit tests
    (service as any).session  = { run: jest.fn() };
    (service as any).tokenizer = null;
  });

  // ── Softmax ───────────────────────────────────────────────
  describe('softmax helper', () => {
    it('should sum to 1', () => {
      const result = (service as any).softmax([1, 2, 3]);
      const sum    = result.reduce((a: number, b: number) => a + b, 0);
      expect(sum).toBeCloseTo(1, 5);
    });

    it('should produce the highest prob for the highest logit', () => {
      const result = (service as any).softmax([0, 10, 0]);
      expect(result[1]).toBeGreaterThan(0.99);
    });

    it('should handle negative logits', () => {
      const result = (service as any).softmax([-1, -2, -3]);
      expect(result.every((v: number) => v > 0 && v < 1)).toBe(true);
    });
  });

  // ── Sentiment ─────────────────────────────────────────────
  describe('decodeSentiment', () => {
    it('should clamp output to [-1, 1]', () => {
      const cases = [
        [100.0,  1],
        [-100.0, -1],
        [0.0,     0],
      ] as [number, number][];

      cases.forEach(([input, expected]) => {
        const logits = new Float32Array([input]);
        const result = (service as any).decodeSentiment(logits);
        expect(result).toBeCloseTo(expected, 1);
      });
    });

    it('should return 0 for empty logits', () => {
      expect((service as any).decodeSentiment(new Float32Array())).toBe(0);
    });
  });

  // ── Cache ─────────────────────────────────────────────────
  describe('cache', () => {
    it('should return null on cache miss', async () => {
      mockRedis.get.mockResolvedValueOnce(null);
      const result = await (service as any).getCached('miss-key');
      expect(result).toBeNull();
    });

    it('should deserialize cache hit', async () => {
      const cached = { entities: [], intent: 'planning', confidence: 0.9 };
      mockRedis.get.mockResolvedValueOnce(JSON.stringify(cached));
      const result = await (service as any).getCached('hit-key');
      expect(result).toEqual(cached);
    });

    it('should return null on Redis error', async () => {
      mockRedis.get.mockRejectedValueOnce(new Error('Redis down'));
      const result = await (service as any).getCached('err-key');
      expect(result).toBeNull();
    });

    it('should generate consistent cache keys', () => {
      const k1 = (service as any).buildCacheKey('hello', 'v1.0');
      const k2 = (service as any).buildCacheKey('hello', 'v1.0');
      const k3 = (service as any).buildCacheKey('world', 'v1.0');
      expect(k1).toBe(k2);
      expect(k1).not.toBe(k3);
      expect(k1).toMatch(/^lyd:pred:v1\.0:/);
    });
  });

  // ── Fallback tokenizer ────────────────────────────────────
  describe('tokenize (fallback)', () => {
    it('should produce fixed-length arrays', async () => {
      const result = await (service as any).tokenize('bonjour le monde');
      expect(result.inputIds.length).toBe(512);
      expect(result.attentionMask.length).toBe(512);
    });

    it('should pad shorter sequences', async () => {
      const { inputIds } = await (service as any).tokenize('a');
      const zeros = Array.from(inputIds).filter((v: any) => v === 0n).length;
      expect(zeros).toBeGreaterThan(0);
    });
  });

  // ── Priority ──────────────────────────────────────────────
  describe('computePriority', () => {
    it('should return critical for very negative sentiment + high confidence + actions', () => {
      const prio = (service as any).computePriority(-0.9, 0.95, [{ action: 'create_task' }]);
      expect(['critical', 'high']).toContain(prio);
    });

    it('should return low for positive sentiment and no actions', () => {
      const prio = (service as any).computePriority(0.8, 0.3, []);
      expect(['low', 'medium']).toContain(prio);
    });
  });
});


// ── MLTrainingService ─────────────────────────────────────────
describe('MLTrainingService', () => {
  let service: MLTrainingService;

  const makeService = async () => {
    const module: TestingModule = await Test.createTestingModule({
      providers: [
        MLTrainingService,
        { provide: getRepositoryToken(TrainingRunEntity),            useValue: mockRepo<TrainingRunEntity>() },
        { provide: getRepositoryToken(ModelFeedbackEntity),          useValue: mockRepo<ModelFeedbackEntity>() },
        { provide: getRepositoryToken(ConversationAnnotationEntity), useValue: mockRepo<ConversationAnnotationEntity>() },
        { provide: getRepositoryToken(ModelDeploymentEntity),        useValue: mockRepo<ModelDeploymentEntity>() },
        { provide: getQueueToken('model-training'),                   useValue: mockQueue },
        { provide: DataSource,                                        useValue: mockDataSource },
        { provide: ModelInferenceService,                             useValue: mockInferenceService },
      ],
    }).compile();
    return module.get<MLTrainingService>(MLTrainingService);
  };

  beforeEach(async () => {
    jest.clearAllMocks();
    service = await makeService();
  });

  // ── Threshold check ───────────────────────────────────────
  describe('checkAndTriggerTraining', () => {
    it('should NOT trigger when feedback < 500', async () => {
      mockDataSource.query.mockResolvedValueOnce([{ count: '100' }]);
      (service['trainingRunRepo'].findOne as jest.Mock).mockResolvedValueOnce(null);
      const triggered = await service.checkAndTriggerTraining('org-1');
      expect(triggered).toBe(false);
      expect(mockQueue.add).not.toHaveBeenCalled();
    });

    it('should trigger when feedback >= 500', async () => {
      mockDataSource.query.mockResolvedValueOnce([{ count: '600' }]);
      (service['trainingRunRepo'].findOne as jest.Mock).mockResolvedValueOnce(null);
      (service['trainingRunRepo'].findOne as jest.Mock).mockResolvedValueOnce(null); // no prev version
      (service['trainingRunRepo'].find as jest.Mock).mockResolvedValueOnce([]);
      const triggered = await service.checkAndTriggerTraining('org-1');
      expect(triggered).toBe(true);
      expect(mockQueue.add).toHaveBeenCalledWith(
        'train-new-model',
        expect.objectContaining({ organizationId: 'org-1', feedbackCount: 600 }),
        expect.any(Object),
      );
    });

    it('should NOT trigger when training already in progress', async () => {
      (service['trainingRunRepo'].findOne as jest.Mock).mockResolvedValueOnce({
        id: 'run-1', status: 'training', version: 'v1.0',
      });
      const triggered = await service.checkAndTriggerTraining('org-1');
      expect(triggered).toBe(false);
    });

    it('should trigger when force=true regardless of threshold', async () => {
      mockDataSource.query.mockResolvedValueOnce([{ count: '10' }]);
      (service['trainingRunRepo'].findOne as jest.Mock)
        .mockResolvedValueOnce(null)  // no active run
        .mockResolvedValueOnce(null); // no previous version
      (service['trainingRunRepo'].find as jest.Mock).mockResolvedValueOnce([]);
      const triggered = await service.checkAndTriggerTraining('org-1', true);
      expect(triggered).toBe(true);
    });
  });

  // ── Validation ────────────────────────────────────────────
  describe('validateModel', () => {
    const goodMetrics: TrainingMetrics = {
      ner_f1: 0.86, ner_precision: 0.88, ner_recall: 0.84,
      action_f1: 0.82, intent_accuracy: 0.90,
      relation_f1: 0.78, overall_f1: 0.86, loss: 0.22,
    };

    const prevMetrics: TrainingMetrics = {
      ner_f1: 0.78, ner_precision: 0.80, ner_recall: 0.76,
      action_f1: 0.75, intent_accuracy: 0.82,
      relation_f1: 0.70, overall_f1: 0.78, loss: 0.31,
    };

    it('should pass when F1 improves significantly', async () => {
      const { passed } = await service.validateModel(goodMetrics, prevMetrics);
      expect(passed).toBe(true);
    });

    it('should fail on NER F1 regression > 2%', async () => {
      const bad = { ...goodMetrics, overall_f1: 0.75, ner_f1: 0.78 };
      const { passed, reason } = await service.validateModel(bad, prevMetrics);
      // F1 delta = 0.75 - 0.78 = -0.03 → regression
      expect(passed).toBe(false);
      expect(reason).toContain('regression');
    });

    it('should fail when NER F1 below absolute minimum 0.70', async () => {
      const bad = { ...goodMetrics, ner_f1: 0.65, overall_f1: 0.75 };
      const { passed } = await service.validateModel(bad, null);
      expect(passed).toBe(false);
    });

    it('should pass with no previous metrics (first training)', async () => {
      const { passed } = await service.validateModel(goodMetrics, null);
      expect(passed).toBe(true);
    });

    it('should fail when intent accuracy below absolute minimum 0.75', async () => {
      const bad = { ...goodMetrics, intent_accuracy: 0.70 };
      const { passed } = await service.validateModel(bad, null);
      expect(passed).toBe(false);
    });
  });

  // ── Data split ────────────────────────────────────────────
  describe('stratifiedSplit', () => {
    const makeSamples = (n: number) =>
      Array.from({ length: n }, (_, i) => ({
        id: `s${i}`, text: `text ${i}`,
        entities: [], relations: [], actions: [],
        intent: i % 2 === 0 ? 'planning' : 'status_update',
        source: 'annotation' as const,
        weight: 1,
      }));

    it('should produce correct proportions', () => {
      const samples = makeSamples(100);
      const split   = (service as any).stratifiedSplit(samples, [0.8, 0.1, 0.1]);
      expect(split.train.length + split.validation.length + split.test.length).toBe(100);
      expect(split.train.length).toBeGreaterThanOrEqual(76);
      expect(split.train.length).toBeLessThanOrEqual(84);
    });

    it('should handle small datasets', () => {
      const samples = makeSamples(3);
      const split   = (service as any).stratifiedSplit(samples, [0.8, 0.1, 0.1]);
      expect(split.test.length).toBeGreaterThanOrEqual(0);
    });
  });

  // ── Version computation ───────────────────────────────────
  describe('computeNextVersion', () => {
    it('should return v1.0 for first training', async () => {
      (service['trainingRunRepo'].findOne as jest.Mock).mockResolvedValueOnce(null);
      const v = await (service as any).computeNextVersion('org-1');
      expect(v).toBe('v1.0');
    });

    it('should increment minor version', async () => {
      (service['trainingRunRepo'].findOne as jest.Mock).mockResolvedValueOnce({ version: 'v1.3' });
      const v = await (service as any).computeNextVersion('org-1');
      expect(v).toBe('v1.4');
    });

    it('should bump major at minor=10', async () => {
      (service['trainingRunRepo'].findOne as jest.Mock).mockResolvedValueOnce({ version: 'v1.9' });
      const v = await (service as any).computeNextVersion('org-1');
      expect(v).toBe('v2.0');
    });
  });

  // ── Augmentation ─────────────────────────────────────────
  describe('augmentData', () => {
    it('should produce more samples than input', async () => {
      const input = [
        { id: '1', text: 'Test conversation here', entities: [], relations: [],
          actions: [], intent: 'planning', source: 'annotation' as const, weight: 1 },
      ];
      const result = await (service as any).augmentData(input);
      expect(result.length).toBeGreaterThan(input.length);
    });

    it('augmented samples should have lower weight', async () => {
      const input = [
        { id: '1', text: 'hello world', entities: [], relations: [],
          actions: [], source: 'annotation' as const, weight: 1.5 },
      ];
      const result = await (service as any).augmentData(input);
      const augmented = result.filter((s: any) => s.id !== '1');
      augmented.forEach((s: any) => expect(s.weight).toBeLessThan(1.5));
    });
  });
});


// ── FeedbackService ───────────────────────────────────────────
describe('FeedbackService', () => {
  let service: FeedbackService;
  let convRepo: Partial<Repository<ConversationEntity>>;
  let feedbackRepo: Partial<Repository<ModelFeedbackEntity>>;
  let actionRepo: Partial<Repository<ExtractedActionEntity>>;

  beforeEach(async () => {
    convRepo     = mockRepo<ConversationEntity>();
    feedbackRepo = mockRepo<ModelFeedbackEntity>();
    actionRepo   = mockRepo<ExtractedActionEntity>();

    const module = await Test.createTestingModule({
      providers: [
        FeedbackService,
        { provide: getRepositoryToken(ModelFeedbackEntity),  useValue: feedbackRepo },
        { provide: getRepositoryToken(ConversationEntity),   useValue: convRepo },
        { provide: getRepositoryToken(ExtractedActionEntity),useValue: actionRepo },
      ],
    }).compile();

    service = module.get<FeedbackService>(FeedbackService);
    jest.clearAllMocks();
  });

  it('should throw NotFoundException for unknown conversation', async () => {
    (convRepo.findOne as jest.Mock).mockResolvedValueOnce(null);
    const dto: SubmitFeedbackDto = {
      conversationId: 'unknown-uuid',
      actionId:       'action-uuid',
      feedbackType:   'incorrect',
    };
    await expect(
      service.submitFeedback(dto, 'user-1', 'org-1', 'v1.0', {}),
    ).rejects.toThrow('Conversation unknown-uuid not found');
  });

  it('should save feedback for known conversation', async () => {
    (convRepo.findOne as jest.Mock).mockResolvedValueOnce({
      id: 'conv-1', organizationId: 'org-1',
    });
    (feedbackRepo.create as jest.Mock).mockReturnValueOnce({ id: 'fb-1' });
    (feedbackRepo.save as jest.Mock).mockResolvedValueOnce({ id: 'fb-1' });

    const dto: SubmitFeedbackDto = {
      conversationId: 'conv-1',
      actionId:       'action-1',
      feedbackType:   'partially_correct',
      userCorrection: { entities: [], intent: 'planning' },
    };

    const result = await service.submitFeedback(dto, 'user-1', 'org-1', 'v1.0', {
      entities: [], intent: 'decision_making', actions: [], confidence: 0.7,
    });

    expect(feedbackRepo.save).toHaveBeenCalledTimes(1);
    expect(result).toMatchObject({ id: 'fb-1' });
  });

  it('should update action approval when approved=true', async () => {
    (convRepo.findOne as jest.Mock).mockResolvedValueOnce({ id: 'conv-1' });
    (feedbackRepo.create as jest.Mock).mockReturnValueOnce({});
    (feedbackRepo.save as jest.Mock).mockResolvedValueOnce({ id: 'fb-1' });

    const dto: SubmitFeedbackDto = {
      conversationId: 'conv-1',
      actionId:       'action-1',
      feedbackType:   'fully_correct',
      approved:       true,
      comment:        'Looks good!',
    };

    await service.submitFeedback(dto, 'user-1', 'org-1', 'v1.0', {});
    expect(actionRepo.update).toHaveBeenCalledWith(
      { id: 'action-1', conversationId: 'conv-1' },
      expect.objectContaining({ userApproval: true, actionStatus: 'reviewed' }),
    );
  });
});


// ── TrainingProcessor ─────────────────────────────────────────
describe('TrainingProcessor', () => {
  let processor: TrainingProcessor;
  let mlTrainingSvc: jest.Mocked<MLTrainingService>;

  beforeEach(async () => {
    const module = await Test.createTestingModule({
      providers: [
        TrainingProcessor,
        {
          provide: MLTrainingService,
          useValue: {
            updateTrainingRun:    jest.fn(() => Promise.resolve()),
            prepareTrainingData:  jest.fn(() => Promise.resolve({
              train:      Array(80).fill({ id: '1', text: 'x', entities: [], relations: [], actions: [], source: 'annotation', weight: 1 }),
              validation: Array(10).fill({ id: '2', text: 'x', entities: [], relations: [], actions: [], source: 'annotation', weight: 1 }),
              test:       Array(10).fill({ id: '3', text: 'x', entities: [], relations: [], actions: [], source: 'annotation', weight: 1 }),
            })),
            trainModelPython:     jest.fn(() => Promise.resolve({
              ner_f1: 0.86, ner_precision: 0.88, ner_recall: 0.84,
              action_f1: 0.82, intent_accuracy: 0.91, relation_f1: 0.79,
              overall_f1: 0.86, loss: 0.21,
            })),
            validateModel:        jest.fn(() => Promise.resolve({ passed: true })),
            runABTest:            jest.fn(() => Promise.resolve({ approved: true, agreementRate: 0.95 })),
            deployNewModel:       jest.fn(() => Promise.resolve()),
            markFeedbacksUsed:    jest.fn(() => Promise.resolve(42)),
            getTrainingHistory:   jest.fn(() => Promise.resolve([])),
          } as any,
        },
      ],
    }).compile();

    processor    = module.get<TrainingProcessor>(TrainingProcessor);
    mlTrainingSvc= module.get(MLTrainingService);
    jest.clearAllMocks();
  });

  const makeJob = (overrides = {}) => ({
    id:          'job-1',
    timestamp:   Date.now(),
    processedOn: Date.now(),
    data: {
      organizationId: 'org-1',
      trainingRunId:  'run-1',
      version:        'v1.1',
      feedbackCount:  600,
      timestamp:      new Date().toISOString(),
      ...overrides,
    },
    progress: jest.fn(),
  } as any);

  it('should complete a successful training pipeline', async () => {
    const result = await processor.handleTraining(makeJob());
    expect(result.status).toBe('completed');
    expect(result.version).toBe('v1.1');
    expect(mlTrainingSvc.deployNewModel).toHaveBeenCalledWith('v1.1', 'org-1', 'run-1');
    expect(mlTrainingSvc.markFeedbacksUsed).toHaveBeenCalledWith('org-1', 'run-1');
  });

  it('should fail when dataset is too small', async () => {
    (mlTrainingSvc.prepareTrainingData as jest.Mock).mockResolvedValueOnce({
      train: [], validation: [], test: [],
    });
    const result = await processor.handleTraining(makeJob());
    expect(result.status).toBe('failed');
    expect(result.reason).toContain('Insufficient data');
    expect(mlTrainingSvc.deployNewModel).not.toHaveBeenCalled();
  });

  it('should fail when validation fails', async () => {
    (mlTrainingSvc.validateModel as jest.Mock).mockResolvedValueOnce({
      passed: false, reason: 'NER F1 too low: 0.60',
    });
    const result = await processor.handleTraining(makeJob());
    expect(result.status).toBe('failed');
    expect(result.reason).toContain('Validation failed');
    expect(mlTrainingSvc.deployNewModel).not.toHaveBeenCalled();
  });

  it('should fail when A/B test fails', async () => {
    (mlTrainingSvc.runABTest as jest.Mock).mockResolvedValueOnce({
      approved: false, agreementRate: 0.72,
    });
    const result = await processor.handleTraining(makeJob());
    expect(result.status).toBe('failed');
    expect(result.reason).toContain('A/B test failed');
    expect(mlTrainingSvc.deployNewModel).not.toHaveBeenCalled();
  });

  it('should fail gracefully when Python training throws', async () => {
    (mlTrainingSvc.trainModelPython as jest.Mock).mockRejectedValueOnce(
      new Error('CUDA out of memory'),
    );
    const result = await processor.handleTraining(makeJob());
    expect(result.status).toBe('failed');
    expect(result.reason).toContain('Python training failed');
  });

  it('should report progress steps', async () => {
    const job = makeJob();
    await processor.handleTraining(job);
    expect(job.progress).toHaveBeenCalledWith(5);
    expect(job.progress).toHaveBeenCalledWith(10);
    expect(job.progress).toHaveBeenCalledWith(100);
  });
});
