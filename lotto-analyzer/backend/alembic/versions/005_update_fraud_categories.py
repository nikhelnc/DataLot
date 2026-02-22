"""Update fraud_alerts category constraint to include all fraud test categories

Revision ID: 005
Revises: 004
Create Date: 2026-02-22

"""
from alembic import op
import sqlalchemy as sa

# revision identifiers, used by Alembic.
revision = '005'
down_revision = '004'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # Drop old constraint
    op.drop_constraint('ck_fraud_category', 'fraud_alerts', type_='check')
    
    # Add new constraint with all fraud test categories
    op.create_check_constraint(
        'ck_fraud_category',
        'fraud_alerts',
        "category IN ('generator', 'data_quality', 'behavioral', 'structural', 'benford', 'dispersion', 'clustering', 'jackpot', 'temporal', 'statistical')"
    )


def downgrade() -> None:
    # Drop new constraint
    op.drop_constraint('ck_fraud_category', 'fraud_alerts', type_='check')
    
    # Restore old constraint
    op.create_check_constraint(
        'ck_fraud_category',
        'fraud_alerts',
        "category IN ('generator', 'data_quality', 'behavioral', 'structural')"
    )
