import { Entity, Column, PrimaryGeneratedColumn, CreateDateColumn, Index } from 'typeorm';

@Entity('ml_inference_logs')
export class MLInferenceLogEntity {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  @Index()
  organizationId: string;

  @Column({ type: 'varchar', length: 50 })
  @Index()
  modelVersion: string;

  @Column({ type: 'int', nullable: true })
  inputTokens: number;

  @Column({ type: 'int', nullable: true })
  outputTokens: number;

  @Column({ type: 'int' })
  inferenceTimeMs: number;

  @Column({ type: 'float', nullable: true })
  memoryUsedMb: number;

  @Column({ type: 'int', nullable: true })
  predictionsCount: number;

  @Column({ type: 'float', nullable: true })
  confidenceAvg: number;

  @Column({ type: 'boolean', default: false })
  cacheHit: boolean;

  @Column({ type: 'uuid', nullable: true })
  userId: string;

  @Column({ type: 'uuid', nullable: true })
  conversationId: string;

  @CreateDateColumn()
  @Index()
  createdAt: Date;
}
