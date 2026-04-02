import { Entity, Column, PrimaryGeneratedColumn, CreateDateColumn, UpdateDateColumn, Index } from 'typeorm';

@Entity('conversations')
export class ConversationEntity {
  @PrimaryGeneratedColumn('uuid')
  id: string;

  @Column({ type: 'uuid' })
  @Index()
  organizationId: string;

  @Column({ type: 'text' })
  text: string;

  @Column({ type: 'varchar', length: 10, default: 'fr' })
  language: string;

  @Column({ type: 'varchar', length: 50 })
  type: 'meeting' | 'email' | 'chat' | 'call';

  @Column({ type: 'varchar', length: 255, nullable: true })
  title: string;

  @Column({ type: 'uuid', array: true, nullable: true })
  participants: string[];

  @Column({ type: 'varchar', length: 255, array: true, nullable: true })
  participantEmails: string[];

  @Column({ type: 'varchar', length: 50, nullable: true })
  @Index()
  modelVersion: string;

  @Column({ type: 'timestamp with time zone', nullable: true })
  conversationDate: Date;

  @CreateDateColumn()
  createdAt: Date;

  @UpdateDateColumn()
  updatedAt: Date;
}
