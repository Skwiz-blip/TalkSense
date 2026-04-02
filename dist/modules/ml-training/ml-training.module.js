"use strict";
var __decorate = (this && this.__decorate) || function (decorators, target, key, desc) {
    var c = arguments.length, r = c < 3 ? target : desc === null ? desc = Object.getOwnPropertyDescriptor(target, key) : desc, d;
    if (typeof Reflect === "object" && typeof Reflect.decorate === "function") r = Reflect.decorate(decorators, target, key, desc);
    else for (var i = decorators.length - 1; i >= 0; i--) if (d = decorators[i]) r = (c < 3 ? d(r) : c > 3 ? d(target, key, r) : d(target, key)) || r;
    return c > 3 && r && Object.defineProperty(target, key, r), r;
};
Object.defineProperty(exports, "__esModule", { value: true });
exports.MlTrainingModule = void 0;
const common_1 = require("@nestjs/common");
const typeorm_1 = require("@nestjs/typeorm");
const model_inference_service_1 = require("./services/model-inference.service");
const conversation_entity_1 = require("../../entities/conversation.entity");
const ml_inference_log_entity_1 = require("../../entities/ml-inference-log.entity");
let MlTrainingModule = class MlTrainingModule {
};
exports.MlTrainingModule = MlTrainingModule;
exports.MlTrainingModule = MlTrainingModule = __decorate([
    (0, common_1.Module)({
        imports: [typeorm_1.TypeOrmModule.forFeature([conversation_entity_1.ConversationEntity, ml_inference_log_entity_1.MLInferenceLogEntity])],
        providers: [model_inference_service_1.ModelInferenceService],
        exports: [model_inference_service_1.ModelInferenceService],
    })
], MlTrainingModule);
//# sourceMappingURL=ml-training.module.js.map