
CREATE EXTENSION IF NOT EXISTS "pgcrypto";
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

CREATE TABLE IF NOT EXISTS organizations (
    id          UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    name        VARCHAR(255) NOT NULL,
    slug        VARCHAR(100) UNIQUE,
    created_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at  TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS users (
    id              UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id UUID REFERENCES organizations(id) ON DELETE CASCADE,
    email           VARCHAR(255) UNIQUE NOT NULL,
    name            VARCHAR(255),
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS conversations (
    id                  UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id     UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
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
    conversation_id UUID        NOT NULL UNIQUE REFERENCES conversations(id) ON DELETE CASCADE,
    entities        JSONB       NOT NULL DEFAULT '[]',
    relations       JSONB       NOT NULL DEFAULT '[]',
    actions         JSONB       NOT NULL DEFAULT '[]',
    intent          VARCHAR(100),
    sentiment_score FLOAT,
    priority        VARCHAR(20),
    manual_summary  TEXT,
    annotated_by    UUID        NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    annotated_at    TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    reviewed_by     UUID REFERENCES users(id) ON DELETE SET NULL,
    reviewed_at     TIMESTAMP WITH TIME ZONE,
    created_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    updated_at      TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
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
    organization_id UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
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
    conversation_id UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    model_version   VARCHAR(50) NOT NULL,
    model_prediction    JSONB   NOT NULL,
    user_correction     JSONB,
    feedback_type       VARCHAR(50) NOT NULL,
    confidence_score    FLOAT,
    model_confidence    FLOAT,
    feedback_by         UUID        NOT NULL REFERENCES users(id) ON DELETE RESTRICT,
    feedback_at         TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
    used_for_training   BOOLEAN     NOT NULL DEFAULT FALSE,
    training_run_id     UUID REFERENCES training_runs(id) ON DELETE SET NULL,
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW(),
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
    organization_id     UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
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

-- ── 8. EXTRACTED ACTIONS ─────────────────────────────────────
CREATE TABLE IF NOT EXISTS extracted_actions (
    id                      UUID        PRIMARY KEY DEFAULT gen_random_uuid(),
    organization_id         UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    conversation_id         UUID        NOT NULL REFERENCES conversations(id) ON DELETE CASCADE,
    action_type             VARCHAR(100) NOT NULL,
    title                   VARCHAR(255) NOT NULL,
    description             TEXT,
    assignee_id             UUID REFERENCES users(id) ON DELETE SET NULL,
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
    organization_id     UUID        NOT NULL REFERENCES organizations(id) ON DELETE CASCADE,
    model_version       VARCHAR(50) NOT NULL,
    input_tokens        INT,
    output_tokens       INT,
    inference_time_ms   INT         NOT NULL,
    memory_used_mb      FLOAT,
    predictions_count   INT,
    confidence_avg      FLOAT,
    cache_hit           BOOLEAN     NOT NULL DEFAULT FALSE,
    user_id             UUID REFERENCES users(id) ON DELETE SET NULL,
    conversation_id     UUID REFERENCES conversations(id) ON DELETE SET NULL,
    created_at          TIMESTAMP WITH TIME ZONE NOT NULL DEFAULT NOW()
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
            'DROP TRIGGER IF EXISTS trg_%s_updated_at ON %s;
             CREATE TRIGGER trg_%s_updated_at
             BEFORE UPDATE ON %s
             FOR EACH ROW EXECUTE FUNCTION update_updated_at()',
            t, t, t, t
        );
    END LOOP;
END;
$$;

INSERT INTO organizations (id, name, slug)
VALUES ('00000000-0000-0000-0000-000000000001', 'Test Organization', 'test-org')
ON CONFLICT (id) DO NOTHING;

ALTER TABLE conversations ENABLE ROW LEVEL SECURITY;
ALTER TABLE conversation_annotations ENABLE ROW LEVEL SECURITY;
ALTER TABLE training_runs ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_feedback ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_deployments ENABLE ROW LEVEL SECURITY;
ALTER TABLE extracted_actions ENABLE ROW LEVEL SECURITY;
ALTER TABLE ml_inference_logs ENABLE ROW LEVEL SECURITY;

-- Politique: les utilisateurs peuvent voir les données de leur organisation
CREATE POLICY "Users can view their org data" ON conversations
    FOR SELECT USING (organization_id IN (SELECT organization_id FROM users WHERE id = auth.uid()));

CREATE POLICY "Users can view their org annotations" ON conversation_annotations
    FOR SELECT USING (conversation_id IN (SELECT id FROM conversations WHERE organization_id IN (SELECT organization_id FROM users WHERE id = auth.uid())));

-- Politique service_role: accès total (pour le backend)
CREATE POLICY "Service role full access on conversations" ON conversations
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access on annotations" ON conversation_annotations
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access on training_runs" ON training_runs
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access on feedback" ON model_feedback
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access on deployments" ON model_deployments
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access on actions" ON extracted_actions
    FOR ALL TO service_role USING (true) WITH CHECK (true);

CREATE POLICY "Service role full access on logs" ON ml_inference_logs
    FOR ALL TO service_role USING (true) WITH CHECK (true);
SELECT 'Schema created successfully!' AS status;
