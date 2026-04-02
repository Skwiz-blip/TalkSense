import { Module } from '@nestjs/common';
import { TypeOrmModule } from '@nestjs/typeorm';
import { ModelInferenceService } from './services/model-inference.service';
import { ConversationEntity } from '../../entities/conversation.entity';
import { MLInferenceLogEntity } from '../../entities/ml-inference-log.entity';

@Module({
  imports: [TypeOrmModule.forFeature([ConversationEntity, MLInferenceLogEntity])],
  providers: [ModelInferenceService],
  exports: [ModelInferenceService],
})
export class MlTrainingModule {}
