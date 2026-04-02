CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE TABLE IF NOT EXISTS conversations (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id     UUID        NOT NULL,
    text                TEXT        NOT NULL,
    language            VARCHAR(10) NOT NULL DEFAULT 'fr',
    type                VARCHAR(50) NOT NULL,
    title               VARCHAR(255),
    participants        UUID[],
    participant_emails  VARCHAR(255)[],
    model_version       VARCHAR(50),
    conversation_date   TIMESTAMP WITH TIME ZONE,
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_conversations_organization
        FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    CONSTRAINT chk_conversations_text_nonempty
        CHECK (LENGTH(TRIM(text)) > 0),
    CONSTRAINT chk_conversations_type
        CHECK (type IN ('meeting', 'email', 'chat', 'call'))
);

CREATE INDEX IF NOT EXISTS idx_conversations_org_date
    ON conversations(organization_id, conversation_date DESC);
CREATE INDEX IF NOT EXISTS idx_conversations_type
    ON conversations(type);
CREATE INDEX IF NOT EXISTS idx_conversations_model_version
    ON conversations(model_version);

CREATE TABLE IF NOT EXISTS conversation_annotations (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID        NOT NULL UNIQUE,
    entities        JSONB       NOT NULL DEFAULT '[]',
    relations       JSONB       NOT NULL DEFAULT '[]',
    actions         JSONB       NOT NULL DEFAULT '[]',

    intent          VARCHAR(100),
    sentiment_score FLOAT,
    priority        VARCHAR(20),
    manual_summary  TEXT,

    annotated_by    UUID        NOT NULL,
    annotated_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    reviewed_by     UUID,
    reviewed_at     TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_annotations_conversation
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    CONSTRAINT fk_annotations_annotated_by
        FOREIGN KEY (annotated_by) REFERENCES users(id) ON DELETE RESTRICT,
    CONSTRAINT fk_annotations_reviewed_by
        FOREIGN KEY (reviewed_by) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT chk_annotations_sentiment
        CHECK (sentiment_score IS NULL OR (sentiment_score >= -1 AND sentiment_score <= 1)),
    CONSTRAINT chk_annotations_intent
        CHECK (intent IS NULL OR intent IN (
            'information_request','decision_making','problem_solving',
            'status_update','planning'
        )),
    CONSTRAINT chk_annotations_priority
        CHECK (priority IS NULL OR priority IN ('low','medium','high','critical'))
);

CREATE INDEX IF NOT EXISTS idx_annotations_conversation
    ON conversation_annotations(conversation_id);
CREATE INDEX IF NOT EXISTS idx_annotations_annotated_by
    ON conversation_annotations(annotated_by);
CREATE INDEX IF NOT EXISTS idx_annotations_intent
    ON conversation_annotations(intent);

CREATE TABLE IF NOT EXISTS training_runs (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID        NOT NULL,
    version         VARCHAR(50) NOT NULL UNIQUE,
    base_version    VARCHAR(50),

    total_samples   INT         NOT NULL,
    train_samples   INT         NOT NULL,
    val_samples     INT         NOT NULL,
    test_samples    INT         NOT NULL,

    learning_rate   FLOAT       NOT NULL DEFAULT 2e-5,
    epochs          INT         NOT NULL DEFAULT 3,
    batch_size      INT         NOT NULL DEFAULT 32,
    optimizer       VARCHAR(50) NOT NULL DEFAULT 'adamw',
    metrics         JSONB       NOT NULL DEFAULT '{}',

    status          VARCHAR(50) NOT NULL DEFAULT 'queued',
    model_path      VARCHAR(500),
    onnx_path       VARCHAR(500),
    tokenizer_path  VARCHAR(500),
    config_path     VARCHAR(500),
    error_message   TEXT,

    started_at      TIMESTAMP WITH TIME ZONE,
    completed_at    TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_training_runs_organization
        FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    CONSTRAINT chk_training_runs_samples
        CHECK (train_samples + val_samples + test_samples = total_samples),
    CONSTRAINT chk_training_runs_test_positive
        CHECK (test_samples > 0),
    CONSTRAINT chk_training_runs_status
        CHECK (status IN ('queued','training','validating','completed','failed','deployed'))
);

CREATE INDEX IF NOT EXISTS idx_training_runs_version
    ON training_runs(version);
CREATE INDEX IF NOT EXISTS idx_training_runs_status
    ON training_runs(status);
CREATE INDEX IF NOT EXISTS idx_training_runs_organization
    ON training_runs(organization_id);

CREATE TABLE IF NOT EXISTS model_feedback (
    id              UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    conversation_id UUID        NOT NULL,
    model_version   VARCHAR(50) NOT NULL,
    model_prediction    JSONB   NOT NULL,
    user_correction     JSONB,

    feedback_type       VARCHAR(50) NOT NULL,
    confidence_score    FLOAT,
    model_confidence    FLOAT,

    feedback_by         UUID        NOT NULL,
    feedback_at         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    used_for_training   BOOLEAN     NOT NULL DEFAULT FALSE,
    training_run_id     UUID,

    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_feedback_conversation
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    CONSTRAINT fk_feedback_feedback_by
        FOREIGN KEY (feedback_by) REFERENCES users(id) ON DELETE RESTRICT,
    CONSTRAINT fk_feedback_training_run
        FOREIGN KEY (training_run_id) REFERENCES training_runs(id) ON DELETE SET NULL,
    CONSTRAINT chk_feedback_type
        CHECK (feedback_type IN ('fully_correct','partially_correct','incorrect','unclear')),
    CONSTRAINT chk_feedback_confidence
        CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)),
    CONSTRAINT chk_feedback_model_confidence
        CHECK (model_confidence IS NULL OR (model_confidence >= 0 AND model_confidence <= 1)),
    CONSTRAINT chk_feedback_correction_on_error
        CHECK (user_correction IS NULL OR feedback_type != 'fully_correct')
);

CREATE INDEX IF NOT EXISTS idx_feedback_conversation
    ON model_feedback(conversation_id);
CREATE INDEX IF NOT EXISTS idx_feedback_model_version
    ON model_feedback(model_version);
CREATE INDEX IF NOT EXISTS idx_feedback_used_for_training
    ON model_feedback(used_for_training) WHERE used_for_training = FALSE;
CREATE INDEX IF NOT EXISTS idx_feedback_training_run
    ON model_feedback(training_run_id);

CREATE TABLE IF NOT EXISTS model_deployments (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id     UUID        NOT NULL,
    version             VARCHAR(50) NOT NULL,
    previous_version    VARCHAR(50),
    rollout_percentage  INT         NOT NULL DEFAULT 100,
    status              VARCHAR(50) NOT NULL DEFAULT 'shadow',
    metrics             JSONB,

    deployed_at         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    replaced_by         VARCHAR(50),
    replaced_at         TIMESTAMP WITH TIME ZONE,
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_deployments_organization
        FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    CONSTRAINT chk_deployments_rollout
        CHECK (rollout_percentage >= 0 AND rollout_percentage <= 100),
    CONSTRAINT chk_deployments_status
        CHECK (status IN ('shadow','canary','beta','production','deprecated','rolled_back'))
);

CREATE INDEX IF NOT EXISTS idx_deployments_version
    ON model_deployments(version);
CREATE INDEX IF NOT EXISTS idx_deployments_status
    ON model_deployments(status);
CREATE INDEX IF NOT EXISTS idx_deployments_organization
    ON model_deployments(organization_id);

CREATE TABLE IF NOT EXISTS extracted_actions (
    id                      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id         UUID        NOT NULL,
    conversation_id         UUID        NOT NULL,

    action_type             VARCHAR(100) NOT NULL,
    title                   VARCHAR(255) NOT NULL,
    description             TEXT,

    assignee_id             UUID,
    assignee_email          VARCHAR(255),
    due_date                DATE,
    priority                VARCHAR(20),
    urgency                 FLOAT,

    source_conversation_id  UUID        NOT NULL,
    model_version           VARCHAR(50),
    confidence_score        FLOAT,

    related_project_id      UUID,
    related_task_id         UUID,
    tags                    VARCHAR(100)[],

    action_status           VARCHAR(50) NOT NULL DEFAULT 'generated',
    notification_config     JSONB,
    user_approval           BOOLEAN,
    approval_feedback       TEXT,

    created_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    assigned_at             TIMESTAMP WITH TIME ZONE,
    completed_at            TIMESTAMP WITH TIME ZONE,
    updated_at              TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_actions_organization
        FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    CONSTRAINT fk_actions_conversation
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE CASCADE,
    CONSTRAINT fk_actions_assignee
        FOREIGN KEY (assignee_id) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT chk_actions_type
        CHECK (action_type IN ('create_task','create_reminder','follow_up','create_alert')),
    CONSTRAINT chk_actions_priority
        CHECK (priority IS NULL OR priority IN ('low','medium','high','critical')),
    CONSTRAINT chk_actions_confidence
        CHECK (confidence_score IS NULL OR (confidence_score >= 0 AND confidence_score <= 1)),
    CONSTRAINT chk_actions_urgency
        CHECK (urgency IS NULL OR (urgency >= 0 AND urgency <= 1)),
    CONSTRAINT chk_actions_status
        CHECK (action_status IN ('generated','reviewed','assigned','in_progress','completed','skipped'))
);

CREATE INDEX IF NOT EXISTS idx_actions_organization
    ON extracted_actions(organization_id);
CREATE INDEX IF NOT EXISTS idx_actions_conversation
    ON extracted_actions(conversation_id);
CREATE INDEX IF NOT EXISTS idx_actions_assignee
    ON extracted_actions(assignee_id);
CREATE INDEX IF NOT EXISTS idx_actions_status
    ON extracted_actions(action_status);
CREATE INDEX IF NOT EXISTS idx_actions_model_version
    ON extracted_actions(model_version);
CREATE INDEX IF NOT EXISTS idx_actions_due_date
    ON extracted_actions(due_date) WHERE action_status NOT IN ('completed','skipped');

CREATE TABLE IF NOT EXISTS ml_inference_logs (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id     UUID        NOT NULL,
    model_version       VARCHAR(50) NOT NULL,
    input_tokens        INT,
    output_tokens       INT,
    inference_time_ms   INT         NOT NULL,
    memory_used_mb      FLOAT,
    predictions_count   INT,
    confidence_avg      FLOAT,
    cache_hit           BOOLEAN     NOT NULL DEFAULT FALSE,
    user_id             UUID,
    conversation_id     UUID,
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),

    CONSTRAINT fk_inference_logs_organization
        FOREIGN KEY (organization_id) REFERENCES organizations(id) ON DELETE CASCADE,
    CONSTRAINT fk_inference_logs_user
        FOREIGN KEY (user_id) REFERENCES users(id) ON DELETE SET NULL,
    CONSTRAINT fk_inference_logs_conversation
        FOREIGN KEY (conversation_id) REFERENCES conversations(id) ON DELETE SET NULL
);

CREATE INDEX IF NOT EXISTS idx_inference_logs_model_version
    ON ml_inference_logs(model_version);
CREATE INDEX IF NOT EXISTS idx_inference_logs_created_at
    ON ml_inference_logs(created_at DESC);
CREATE INDEX IF NOT EXISTS idx_inference_logs_org_date
    ON ml_inference_logs(organization_id, created_at DESC);

CREATE OR REPLACE FUNCTION update_updated_at()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

DO $$
DECLARE
    t TEXT;
BEGIN
    FOREACH t IN ARRAY ARRAY[
        'conversations',
        'conversation_annotations',
        'training_runs',
        'model_deployments',
        'extracted_actions'
    ] LOOP
        EXECUTE format(
            'CREATE TRIGGER trg_%s_updated_at
             BEFORE UPDATE ON %s
             FOR EACH ROW EXECUTE FUNCTION update_updated_at()',
            t, t
        );
    END LOOP;
END;
$$;
