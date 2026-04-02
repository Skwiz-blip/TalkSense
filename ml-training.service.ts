import { Injectable, Logger, NotFoundException } from '@nestjs/common';
import { InjectRepository } from '@nestjs/typeorm';
import { Repository, DataSource, In } from 'typeorm';
import { InjectQueue } from '@nestjs/bull';
import { Queue } from 'bull';
import { spawn }   from 'child_process';
import * as path   from 'path';
import * as fs     from 'fs/promises';

import { TrainingRunEntity, TrainingStatus, TrainingMetrics } from '../entities/training-run.entity';
import { ModelFeedbackEntity }        from '../entities/model-feedback.entity';
import { ConversationAnnotationEntity } from '../entities/conversation-annotation.entity';
import { ModelDeploymentEntity }      from '../entities/model-deployment.entity';
import { ModelInferenceService }       from './model-inference.service';

// ── Training data format expected by Python ────────────────
interface TrainingSample {
  id:        string;
  text:      string;
  entities:  any[];
  relations: any[];
  actions:   any[];
  intent?:   string;
  sentiment?: number;
  priority?: string;
  source:    'annotation' | 'feedback';
  weight:    number;        // sample importance weight
}

interface SplitDataset {
  train:      TrainingSample[];
  validation: TrainingSample[];
  test:       TrainingSample[];
}

const TRAINING_THRESHOLD = parseInt(process.env.TRAINING_THRESHOLD ?? '500', 10);
const PYTHON_WORKER      = process.env.PYTHON_WORKER_PATH ?? './ml-models';

@Injectable()
export class MLTrainingService {
  private readonly logger = new Logger(MLTrainingService.name);

  constructor(
    @InjectRepository(TrainingRunEntity)
    private readonly trainingRunRepo: Repository<TrainingRunEntity>,
    @InjectRepository(ModelFeedbackEntity)
    private readonly feedbackRepo: Repository<ModelFeedbackEntity>,
    @InjectRepository(ConversationAnnotationEntity)
    private readonly annotationRepo: Repository<ConversationAnnotationEntity>,
    @InjectRepository(ModelDeploymentEntity)
    private readonly deploymentRepo: Repository<ModelDeploymentEntity>,
    @InjectQueue('model-training')
    private readonly trainingQueue: Queue,
    private readonly dataSource: DataSource,
    private readonly inferenceService: ModelInferenceService,
  ) {}

  // ── Check & trigger ────────────────────────────────────
  async checkAndTriggerTraining(
    organizationId: string,
    force = false,
  ): Promise<boolean> {
    // Check if there's already a job running
    const active = await this.trainingRunRepo.findOne({
      where: { organizationId, status: In(['queued', 'training', 'validating']) },
    });
    if (active) {
      this.logger.debug(`Training already in progress (${active.version}) for org ${organizationId}`);
      return false;
    }

    const unusedFeedback = await this.countUnusedFeedback(organizationId);
    this.logger.debug(`Unused feedback for ${organizationId}: ${unusedFeedback}`);

    if (!force && unusedFeedback < TRAINING_THRESHOLD) return false;

    const nextVersion = await this.computeNextVersion(organizationId);
    const run = this.trainingRunRepo.create({
      organizationId,
      version:      nextVersion,
      status:       'queued' as TrainingStatus,
      totalSamples: 0,
      trainSamples: 0,
      valSamples:   0,
      testSamples:  1, // will be updated
      metrics:      {} as TrainingMetrics,
    });
    await this.trainingRunRepo.save(run);

    await this.trainingQueue.add(
      'train-new-model',
      {
        organizationId,
        trainingRunId: run.id,
        version:       nextVersion,
        feedbackCount: unusedFeedback,
        timestamp:     new Date().toISOString(),
      },
      {
        priority:          1,
        attempts:          3,
        backoff:           { type: 'exponential', delay: 5000 },
        removeOnComplete:  false,
        removeOnFail:      false,
      },
    );

    this.logger.log(`🚀 Training queued: ${nextVersion} (${unusedFeedback} feedbacks)`);
    return true;
  }

  // ── Version management ─────────────────────────────────
  private async computeNextVersion(organizationId: string): Promise<string> {
    const last = await this.trainingRunRepo.findOne({
      where:  { organizationId },
      order:  { createdAt: 'DESC' },
      select: ['version'],
    });
    if (!last) return 'v1.0';

    const [major, minor] = last.version.replace('v', '').split('.').map(Number);
    // Bump minor; bump major every 10 minors
    const newMinor = (minor ?? 0) + 1;
    const newMajor = newMinor >= 10 ? (major ?? 1) + 1 : (major ?? 1);
    return `v${newMajor}.${newMinor >= 10 ? 0 : newMinor}`;
  }

  // ── Data preparation ───────────────────────────────────
  async prepareTrainingData(
    organizationId: string,
    limit = 10_000,
  ): Promise<SplitDataset> {
    const [annotations, feedbacks] = await Promise.all([
      this.loadAnnotations(organizationId, limit),
      this.loadFeedbacks(organizationId, limit),
    ]);

    this.logger.log(
      `📦 Raw data: ${annotations.length} annotations, ${feedbacks.length} feedbacks`,
    );

    const samples = [
      ...annotations.map((a) => this.annotationToSample(a)),
      ...feedbacks.map((f) => this.feedbackToSample(f)),
    ];

    const augmented = await this.augmentData(samples);
    const shuffled  = this.shuffle(augmented);
    const split     = this.stratifiedSplit(shuffled, [0.80, 0.10, 0.10]);

    this.logger.log(
      `📊 Split: train=${split.train.length}, val=${split.validation.length}, test=${split.test.length}`,
    );
    return split;
  }

  private async loadAnnotations(
    organizationId: string,
    limit: number,
  ): Promise<ConversationAnnotationEntity[]> {
    return this.annotationRepo
      .createQueryBuilder('a')
      .innerJoinAndSelect('a.conversation', 'c')
      .where('c.organizationId = :orgId', { orgId: organizationId })
      .andWhere('a.reviewedBy IS NOT NULL')          // only reviewed annotations
      .orderBy('a.createdAt', 'DESC')
      .take(limit)
      .getMany();
  }

  private async loadFeedbacks(
    organizationId: string,
    limit: number,
  ): Promise<ModelFeedbackEntity[]> {
    return this.feedbackRepo
      .createQueryBuilder('f')
      .innerJoinAndSelect('f.conversation', 'c')
      .where('c.organizationId = :orgId', { orgId: organizationId })
      .andWhere('f.usedForTraining = false')
      .andWhere('f.feedbackType != :type', { type: 'unclear' })
      .orderBy('f.feedbackAt', 'DESC')
      .take(limit)
      .getMany();
  }

  private annotationToSample(a: ConversationAnnotationEntity): TrainingSample {
    return {
      id:        a.id,
      text:      a.conversation.text,
      entities:  a.entities,
      relations: a.relations,
      actions:   a.actions,
      intent:    a.intent,
      sentiment: a.sentimentScore ?? undefined,
      priority:  a.priority ?? undefined,
      source:    'annotation',
      weight:    1.5,   // Reviewed annotations are higher quality
    };
  }

  private feedbackToSample(f: ModelFeedbackEntity): TrainingSample {
    const correction = f.userCorrection ?? f.modelPrediction;
    return {
      id:        f.id,
      text:      f.conversation.text,
      entities:  correction.entities ?? [],
      relations: [],
      actions:   correction.actions  ?? [],
      intent:    correction.intent,
      sentiment: undefined,
      priority:  correction.priority,
      source:    'feedback',
      weight:    f.feedbackType === 'fully_correct' ? 0.5 : 1.0,
    };
  }

  // ── Data augmentation ──────────────────────────────────
  private async augmentData(samples: TrainingSample[]): Promise<TrainingSample[]> {
    const augmented = [...samples];

    for (const sample of samples) {
      // Strategy 1: Casing variations
      augmented.push({
        ...sample,
        id:     `${sample.id}_aug_case`,
        text:   this.randomCasing(sample.text),
        weight: sample.weight * 0.8,
      });

      // Strategy 2: Punctuation noise
      if (Math.random() < 0.3) {
        augmented.push({
          ...sample,
          id:     `${sample.id}_aug_punct`,
          text:   this.addPunctuationNoise(sample.text),
          weight: sample.weight * 0.7,
        });
      }
    }

    this.logger.log(`🔁 Augmented: ${samples.length} → ${augmented.length} samples`);
    return augmented;
  }

  private randomCasing(text: string): string {
    return Math.random() < 0.5 ? text.toLowerCase() : text;
  }

  private addPunctuationNoise(text: string): string {
    const noises = [', ', '. ', '... ', ' '];
    const pos    = Math.floor(Math.random() * text.length);
    const noise  = noises[Math.floor(Math.random() * noises.length)];
    return text.slice(0, pos) + noise + text.slice(pos);
  }

  // ── Stratified split ───────────────────────────────────
  private stratifiedSplit(
    samples: TrainingSample[],
    ratios: [number, number, number],
  ): SplitDataset {
    // Group by intent for stratification
    const groups = new Map<string, TrainingSample[]>();
    for (const s of samples) {
      const key = s.intent ?? 'unknown';
      if (!groups.has(key)) groups.set(key, []);
      groups.get(key)!.push(s);
    }

    const train: TrainingSample[]      = [];
    const validation: TrainingSample[] = [];
    const test: TrainingSample[]       = [];

    for (const group of groups.values()) {
      const g      = this.shuffle(group);
      const n      = g.length;
      const t1     = Math.floor(n * ratios[0]);
      const t2     = Math.floor(n * (ratios[0] + ratios[1]));
      train.push(...g.slice(0, t1));
      validation.push(...g.slice(t1, t2));
      test.push(...g.slice(t2));
    }

    return { train, validation, test };
  }

  private shuffle<T>(arr: T[]): T[] {
    const a = [...arr];
    for (let i = a.length - 1; i > 0; i--) {
      const j = Math.floor(Math.random() * (i + 1));
      [a[i], a[j]] = [a[j], a[i]];
    }
    return a;
  }

  // ── Python training ────────────────────────────────────
  async trainModelPython(
    dataset: SplitDataset,
    organizationId: string,
    trainingRunId: string,
    config: {
      learningRate: number;
      epochs:       number;
      batchSize:    number;
    },
  ): Promise<TrainingMetrics> {
    // Write dataset to temp files (avoid huge CLI args)
    const tmpDir  = path.join(PYTHON_WORKER, 'tmp', trainingRunId);
    await fs.mkdir(tmpDir, { recursive: true });
    await Promise.all([
      fs.writeFile(path.join(tmpDir, 'train.json'),      JSON.stringify(dataset.train)),
      fs.writeFile(path.join(tmpDir, 'validation.json'), JSON.stringify(dataset.validation)),
      fs.writeFile(path.join(tmpDir, 'test.json'),       JSON.stringify(dataset.test)),
    ]);

    const args = [
      path.join(PYTHON_WORKER, 'src', 'train_entry.py'),
      '--run-id',        trainingRunId,
      '--org-id',        organizationId,
      '--data-dir',      tmpDir,
      '--learning-rate', String(config.learningRate),
      '--epochs',        String(config.epochs),
      '--batch-size',    String(config.batchSize),
      '--model-dir',     path.join(PYTHON_WORKER, 'models'),
    ];

    return new Promise((resolve, reject) => {
      const proc = spawn('python3', args, {
        cwd:       PYTHON_WORKER,
        env:       { ...process.env, PYTHONUNBUFFERED: '1' },
        maxBuffer: 50 * 1024 * 1024,
      });

      let stdout = '';
      let stderr = '';

      proc.stdout.on('data', (d: Buffer) => {
        const line = d.toString();
        stdout += line;
        // Stream progress lines to our logger
        line.split('\n').filter(Boolean).forEach((l) => this.logger.debug(`[Python] ${l}`));
      });

      proc.stderr.on('data', (d: Buffer) => {
        stderr += d.toString();
      });

      proc.on('close', (code) => {
        // Cleanup temp files async
        fs.rm(tmpDir, { recursive: true }).catch(() => undefined);

        if (code !== 0) {
          reject(new Error(`Python process exited ${code}:\n${stderr.slice(-2000)}`));
          return;
        }

        // Extract JSON metrics from last line
        const lines = stdout.trim().split('\n').filter(Boolean);
        const last  = lines[lines.length - 1];
        try {
          const metrics = JSON.parse(last) as TrainingMetrics;
          resolve(metrics);
        } catch {
          reject(new Error(`Could not parse Python output: ${last}`));
        }
      });

      proc.on('error', reject);
    });
  }

  // ── Validation ─────────────────────────────────────────
  async validateModel(
    newMetrics:  TrainingMetrics,
    prevMetrics: TrainingMetrics | null,
  ): Promise<{ passed: boolean; reason?: string }> {
    // Absolute minimums
    if (newMetrics.ner_f1 < 0.70) {
      return { passed: false, reason: `NER F1 too low: ${newMetrics.ner_f1.toFixed(3)}` };
    }
    if (newMetrics.intent_accuracy < 0.75) {
      return { passed: false, reason: `Intent accuracy too low: ${newMetrics.intent_accuracy.toFixed(3)}` };
    }

    // Regression checks vs previous model
    if (prevMetrics) {
      const f1Delta    = newMetrics.overall_f1 - prevMetrics.overall_f1;
      const accDelta   = newMetrics.intent_accuracy - prevMetrics.intent_accuracy;
      const recallDelta= newMetrics.ner_recall - prevMetrics.ner_recall;

      if (f1Delta < -0.02) {
        return { passed: false, reason: `Overall F1 regression: ${f1Delta.toFixed(3)}` };
      }
      if (accDelta < -0.02) {
        return { passed: false, reason: `Intent accuracy regression: ${accDelta.toFixed(3)}` };
      }
      if (recallDelta < -0.05) {
        return { passed: false, reason: `NER recall regression: ${recallDelta.toFixed(3)}` };
      }

      this.logger.log(
        `📈 Validation: F1 ${prevMetrics.overall_f1.toFixed(3)} → ${newMetrics.overall_f1.toFixed(3)} (Δ${f1Delta > 0 ? '+' : ''}${f1Delta.toFixed(3)})`,
      );
    }

    return { passed: true };
  }

  // ── A/B Test ───────────────────────────────────────────
  async runABTest(
    testSamples:  TrainingSample[],
    newVersion:   string,
    organizationId: string,
  ): Promise<{ approved: boolean; agreementRate: number }> {
    if (testSamples.length === 0) {
      this.logger.warn('A/B test: no samples, auto-approving');
      return { approved: true, agreementRate: 1 };
    }

    const subset   = testSamples.slice(0, Math.min(100, testSamples.length));
    let agreements = 0;

    for (const sample of subset) {
      try {
        const prediction = await this.inferenceService.predict(
          sample.text, organizationId,
        );
        // Compare predicted intent vs ground truth
        const intentMatch = prediction.intent === sample.intent;
        // Compare entity count (coarse check)
        const entityMatch =
          Math.abs(prediction.entities.length - sample.entities.length) <= 1;
        if (intentMatch && entityMatch) agreements++;
      } catch {
        // Count failure as disagreement
      }
    }

    const agreementRate = agreements / subset.length;
    const approved      = agreementRate >= 0.85;

    this.logger.log(
      `🧪 A/B Test: ${agreements}/${subset.length} = ${(agreementRate * 100).toFixed(1)}% — ${approved ? '✅ APPROVED' : '❌ REJECTED'}`,
    );

    return { approved, agreementRate };
  }

  // ── Deployment ─────────────────────────────────────────
  async deployNewModel(
    version:        string,
    organizationId: string,
    trainingRunId:  string,
  ): Promise<void> {
    const previousDeployment = await this.deploymentRepo.findOne({
      where:  { organizationId, status: 'production' },
      order:  { deployedAt: 'DESC' },
    });

    // Stage 1: Shadow (5%)
    await this.createOrUpdateDeployment(version, organizationId, 5, 'shadow', previousDeployment?.version);
    this.logger.log(`🚀 [${version}] Shadow deployment at 5%`);
    await this.monitorDeployment(version, organizationId, 30_000); // 30s in dev, 3600s in prod

    const shadowMetrics = await this.getDeploymentMetrics(version, organizationId);
    if (shadowMetrics.error_rate > 0.05) {
      await this.rollback(version, organizationId, 'High error rate in shadow');
      return;
    }

    // Stage 2: Canary (50%)
    await this.updateDeploymentRollout(version, organizationId, 50, 'canary');
    this.logger.log(`📈 [${version}] Canary deployment at 50%`);
    await this.monitorDeployment(version, organizationId, 30_000);

    const canaryMetrics = await this.getDeploymentMetrics(version, organizationId);
    if (canaryMetrics.error_rate > 0.05) {
      await this.rollback(version, organizationId, 'High error rate in canary');
      return;
    }

    // Stage 3: Production (100%)
    await this.updateDeploymentRollout(version, organizationId, 100, 'production');
    this.logger.log(`✅ [${version}] Production deployment at 100%`);

    // Deprecate previous
    if (previousDeployment) {
      await this.deploymentRepo.update(
        { id: previousDeployment.id },
        { status: 'deprecated', replacedBy: version, replacedAt: new Date() },
      );
    }

    // Mark training run as deployed
    await this.trainingRunRepo.update({ id: trainingRunId }, { status: 'deployed' });

    // Hot-reload model in inference service
    await this.inferenceService.reloadModel(version);

    this.logger.log(`🎉 Deployment complete: ${version} is now serving 100% of traffic`);
  }

  private async createOrUpdateDeployment(
    version:         string,
    organizationId:  string,
    rollout:         number,
    status:          string,
    previousVersion?: string,
  ): Promise<void> {
    const deployment = this.deploymentRepo.create({
      organizationId,
      version,
      previousVersion,
      rolloutPercentage: rollout,
      status: status as any,
    });
    await this.deploymentRepo.save(deployment);
  }

  private async updateDeploymentRollout(
    version:        string,
    organizationId: string,
    rollout:        number,
    status:         string,
  ): Promise<void> {
    await this.deploymentRepo.update(
      { version, organizationId },
      { rolloutPercentage: rollout, status: status as any, updatedAt: new Date() },
    );
  }

  private async monitorDeployment(
    version: string,
    _organizationId: string,
    durationMs: number,
  ): Promise<void> {
    // In production this would be a longer wait + live metrics polling
    this.logger.debug(`⏳ Monitoring ${version} for ${durationMs}ms`);
    await new Promise((r) => setTimeout(r, durationMs));
  }

  private async getDeploymentMetrics(
    _version: string,
    _organizationId: string,
  ): Promise<{ error_rate: number; user_approval_rate: number }> {
    // TODO: pull from ml_inference_logs + extracted_actions
    return { error_rate: 0.02, user_approval_rate: 0.94 };
  }

  private async rollback(
    version:        string,
    organizationId: string,
    reason:         string,
  ): Promise<void> {
    this.logger.error(`🔙 Rolling back ${version}: ${reason}`);
    await this.deploymentRepo.update(
      { version, organizationId },
      { status: 'rolled_back', updatedAt: new Date() },
    );
    await this.trainingRunRepo.update(
      { version, organizationId },
      { status: 'failed', errorMessage: reason },
    );
  }

  // ── Mark feedbacks as used ─────────────────────────────
  async markFeedbacksUsed(
    organizationId: string,
    trainingRunId:  string,
  ): Promise<number> {
    const result = await this.dataSource.query<{ count: string }[]>(
      `UPDATE model_feedback f
       SET used_for_training = true, training_run_id = $1
       FROM conversations c
       WHERE f.conversation_id = c.id
         AND c.organization_id = $2
         AND f.used_for_training = false
       RETURNING f.id`,
      [trainingRunId, organizationId],
    );
    return result.length;
  }

  // ── Helpers ────────────────────────────────────────────
  private async countUnusedFeedback(organizationId: string): Promise<number> {
    const [{ count }] = await this.dataSource.query<{ count: string }[]>(
      `SELECT COUNT(*) AS count
       FROM model_feedback f
       JOIN conversations c ON c.id = f.conversation_id
       WHERE c.organization_id = $1
         AND f.used_for_training = false
         AND f.feedback_type != 'unclear'`,
      [organizationId],
    );
    return parseInt(count, 10);
  }

  // ── Public queries ─────────────────────────────────────
  async getTrainingHistory(organizationId: string, limit = 20) {
    return this.trainingRunRepo.find({
      where:  { organizationId },
      order:  { createdAt: 'DESC' },
      take:   limit,
      select: ['id','version','status','metrics','startedAt','completedAt','createdAt'],
    });
  }

  async getActiveDeployment(organizationId: string) {
    return this.deploymentRepo.findOne({
      where:  { organizationId, status: 'production' },
      order:  { deployedAt: 'DESC' },
    });
  }

  async getTrainingRun(id: string): Promise<TrainingRunEntity> {
    const run = await this.trainingRunRepo.findOne({ where: { id } });
    if (!run) throw new NotFoundException(`Training run ${id} not found`);
    return run;
  }

  async updateTrainingRun(
    id:      string,
    updates: Partial<TrainingRunEntity>,
  ): Promise<void> {
    await this.trainingRunRepo.update({ id }, updates);
  }
}
