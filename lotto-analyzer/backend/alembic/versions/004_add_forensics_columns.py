"""Add forensics columns for v2.0 - emission order, jackpot, and new tables

Revision ID: 004
Revises: 003_add_bonus_numbers_array
Create Date: 2026-02-21

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '004'
down_revision = '003'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Add new columns to draws table
    op.add_column('draws', sa.Column('emission_order', sa.ARRAY(sa.Integer()), nullable=True))
    op.add_column('draws', sa.Column('bonus_emission_order', sa.ARRAY(sa.Integer()), nullable=True))
    op.add_column('draws', sa.Column('jackpot_amount', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('draws', sa.Column('jackpot_rollover', sa.Boolean(), server_default='false', nullable=True))
    op.add_column('draws', sa.Column('jackpot_consecutive_rollovers', sa.Integer(), server_default='0', nullable=True))
    op.add_column('draws', sa.Column('must_be_won', sa.Boolean(), server_default='false', nullable=True))
    op.add_column('draws', sa.Column('n_winners_div1', sa.Integer(), nullable=True))
    
    # Create index for jackpot queries
    op.create_index('idx_draws_jackpot', 'draws', ['game_id', 'jackpot_amount'], 
                    postgresql_where=sa.text('jackpot_amount IS NOT NULL'))
    op.create_index('idx_draws_emission', 'draws', ['game_id'], 
                    postgresql_where=sa.text('emission_order IS NOT NULL'))
    
    # Create generator_profiles table
    op.create_table(
        'generator_profiles',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('period_start', sa.Date(), nullable=True),
        sa.Column('period_end', sa.Date(), nullable=True),
        sa.Column('n_draws', sa.Integer(), nullable=False),
        sa.Column('conformity_score', sa.Float(), nullable=True),
        sa.Column('conformity_ci_low', sa.Float(), nullable=True),
        sa.Column('conformity_ci_high', sa.Float(), nullable=True),
        sa.Column('conformity_n_simulations', sa.Integer(), server_default='1000'),
        sa.Column('generator_type', sa.String(20), nullable=True),
        sa.Column('standard_tests', postgresql.JSONB(), nullable=True),
        sa.Column('nist_tests', postgresql.JSONB(), nullable=True),
        sa.Column('physical_tests', postgresql.JSONB(), nullable=True),
        sa.Column('rng_tests', postgresql.JSONB(), nullable=True),
        sa.Column('structural_tests', postgresql.JSONB(), nullable=True),
        sa.Column('dataset_hash', sa.String(64), nullable=False),
        sa.Column('app_version', sa.String(20), nullable=True),
        sa.Column('params', postgresql.JSONB(), nullable=True),
        sa.Column('seed', sa.Integer(), nullable=True),
        sa.CheckConstraint('conformity_score >= 0 AND conformity_score <= 1', name='ck_conformity_score_range'),
        sa.CheckConstraint("generator_type IN ('physical', 'rng', 'hybrid', 'unknown')", name='ck_generator_type'),
        sa.UniqueConstraint('game_id', 'dataset_hash', name='uq_generator_profile')
    )
    
    # Create fraud_alerts table
    op.create_table(
        'fraud_alerts',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('created_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('severity', sa.String(10), nullable=False),
        sa.Column('signal_type', sa.String(50), nullable=False),
        sa.Column('category', sa.String(30), nullable=True),
        sa.Column('title', sa.String(200), nullable=False),
        sa.Column('description', sa.Text(), nullable=True),
        sa.Column('statistical_evidence', postgresql.JSONB(), nullable=True),
        sa.Column('draw_ids', postgresql.ARRAY(postgresql.UUID(as_uuid=True)), nullable=True),
        sa.Column('period_start', sa.Date(), nullable=True),
        sa.Column('period_end', sa.Date(), nullable=True),
        sa.Column('status', sa.String(20), server_default='OPEN', nullable=False),
        sa.Column('assigned_to', sa.String(100), nullable=True),
        sa.Column('resolution_note', sa.Text(), nullable=True),
        sa.Column('resolved_at', sa.DateTime(timezone=True), nullable=True),
        sa.Column('analysis_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('analyses.id'), nullable=True),
        sa.CheckConstraint("severity IN ('INFO', 'WARNING', 'HIGH', 'CRITICAL')", name='ck_fraud_severity'),
        sa.CheckConstraint("category IN ('generator', 'data_quality', 'behavioral', 'structural')", name='ck_fraud_category'),
        sa.CheckConstraint("status IN ('OPEN', 'INVESTIGATING', 'CLOSED', 'FALSE_POSITIVE')", name='ck_fraud_status')
    )
    
    # Create indexes for fraud_alerts
    op.create_index('idx_fraud_alerts_open', 'fraud_alerts', ['game_id', 'severity'],
                    postgresql_where=sa.text("status = 'OPEN'"))
    op.create_index('idx_fraud_alerts_game_date', 'fraud_alerts', ['game_id', 'created_at'])
    
    # Create jackpot_analyses table
    op.create_table(
        'jackpot_analyses',
        sa.Column('id', postgresql.UUID(as_uuid=True), primary_key=True, server_default=sa.text('gen_random_uuid()')),
        sa.Column('game_id', postgresql.UUID(as_uuid=True), sa.ForeignKey('games.id', ondelete='CASCADE'), nullable=False),
        sa.Column('computed_at', sa.DateTime(timezone=True), server_default=sa.text('NOW()'), nullable=False),
        sa.Column('generator_independence_test', postgresql.JSONB(), nullable=True),
        sa.Column('player_bias_analysis', postgresql.JSONB(), nullable=True),
        sa.Column('threshold_effect', postgresql.JSONB(), nullable=True),
        sa.Column('must_be_won_analysis', postgresql.JSONB(), nullable=True),
        sa.Column('jackpot_vs_emission', postgresql.JSONB(), nullable=True),
        sa.Column('n_draws_analyzed', sa.Integer(), nullable=True),
        sa.Column('jackpot_range_min', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('jackpot_range_max', sa.Numeric(precision=15, scale=2), nullable=True),
        sa.Column('dataset_hash', sa.String(64), nullable=True)
    )
    
    # Add columns to existing changepoints/alerts if needed (extend alerts table)
    op.add_column('alerts', sa.Column('affected_numbers', sa.ARRAY(sa.Integer()), nullable=True))
    op.add_column('alerts', sa.Column('magnitude', sa.Float(), nullable=True))
    op.add_column('alerts', sa.Column('context_note', sa.Text(), nullable=True))
    op.add_column('alerts', sa.Column('is_validated', sa.Boolean(), server_default='false', nullable=True))
    op.add_column('alerts', sa.Column('validation_note', sa.Text(), nullable=True))
    op.add_column('alerts', sa.Column('jackpot_at_changepoint', sa.Numeric(precision=15, scale=2), nullable=True))
    op.add_column('alerts', sa.Column('emission_position_affected', sa.Integer(), nullable=True))


def downgrade() -> None:
    # Remove columns from alerts
    op.drop_column('alerts', 'emission_position_affected')
    op.drop_column('alerts', 'jackpot_at_changepoint')
    op.drop_column('alerts', 'validation_note')
    op.drop_column('alerts', 'is_validated')
    op.drop_column('alerts', 'context_note')
    op.drop_column('alerts', 'magnitude')
    op.drop_column('alerts', 'affected_numbers')
    
    # Drop new tables
    op.drop_table('jackpot_analyses')
    op.drop_index('idx_fraud_alerts_game_date', table_name='fraud_alerts')
    op.drop_index('idx_fraud_alerts_open', table_name='fraud_alerts')
    op.drop_table('fraud_alerts')
    op.drop_table('generator_profiles')
    
    # Remove indexes and columns from draws
    op.drop_index('idx_draws_emission', table_name='draws')
    op.drop_index('idx_draws_jackpot', table_name='draws')
    op.drop_column('draws', 'n_winners_div1')
    op.drop_column('draws', 'must_be_won')
    op.drop_column('draws', 'jackpot_consecutive_rollovers')
    op.drop_column('draws', 'jackpot_rollover')
    op.drop_column('draws', 'jackpot_amount')
    op.drop_column('draws', 'bonus_emission_order')
    op.drop_column('draws', 'emission_order')
