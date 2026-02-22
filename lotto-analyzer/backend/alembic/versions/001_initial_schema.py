"""initial schema

Revision ID: 001
Revises: 
Create Date: 2025-01-23 11:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

revision = '001'
down_revision = None
branch_labels = None
depends_on = None


def upgrade() -> None:
    op.create_table('games',
    sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('description', sa.Text(), nullable=True),
    sa.Column('rules_json', sa.JSON(), nullable=False),
    sa.Column('version', sa.Integer(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.PrimaryKeyConstraint('id'),
    sa.UniqueConstraint('name')
    )
    
    op.create_table('draws',
    sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('game_id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('draw_date', sa.DateTime(), nullable=False),
    sa.Column('numbers', postgresql.ARRAY(sa.Integer()), nullable=False),
    sa.Column('bonus', sa.Integer(), nullable=True),
    sa.Column('raw_payload', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['game_id'], ['games.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_draws_game_date', 'draws', ['game_id', 'draw_date'])
    
    op.create_table('imports',
    sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('game_id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('source', sa.String(), nullable=False),
    sa.Column('file_hash', sa.String(), nullable=True),
    sa.Column('status', sa.String(), nullable=False),
    sa.Column('stats_json', sa.JSON(), nullable=True),
    sa.Column('error_log', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['game_id'], ['games.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('analyses',
    sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('game_id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('name', sa.String(), nullable=False),
    sa.Column('params_json', sa.JSON(), nullable=True),
    sa.Column('results_json', sa.JSON(), nullable=True),
    sa.Column('dataset_hash', sa.String(), nullable=True),
    sa.Column('code_version', sa.String(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['game_id'], ['games.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    
    op.create_table('alerts',
    sa.Column('id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('game_id', postgresql.UUID(as_uuid=True), nullable=False),
    sa.Column('analysis_id', postgresql.UUID(as_uuid=True), nullable=True),
    sa.Column('severity', sa.String(), nullable=False),
    sa.Column('score', sa.Integer(), nullable=False),
    sa.Column('message', sa.Text(), nullable=False),
    sa.Column('evidence_json', sa.JSON(), nullable=True),
    sa.Column('created_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['analysis_id'], ['analyses.id'], ),
    sa.ForeignKeyConstraint(['game_id'], ['games.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    op.create_index('idx_alerts_game_severity', 'alerts', ['game_id', 'severity'])


def downgrade() -> None:
    op.drop_index('idx_alerts_game_severity', table_name='alerts')
    op.drop_table('alerts')
    op.drop_table('analyses')
    op.drop_table('imports')
    op.drop_index('idx_draws_game_date', table_name='draws')
    op.drop_table('draws')
    op.drop_table('games')
