"""add bonus_numbers array

Revision ID: 003
Revises: 001
Create Date: 2026-01-23 15:00:00.000000

"""
from alembic import op
import sqlalchemy as sa
from sqlalchemy.dialects import postgresql

# revision identifiers, used by Alembic.
revision = '003'
down_revision = '001'
branch_labels = None
depends_on = None


def upgrade():
    # Add new bonus_numbers column as ARRAY
    op.add_column('draws', sa.Column('bonus_numbers', postgresql.ARRAY(sa.Integer()), nullable=True))
    
    # Migrate existing bonus data to bonus_numbers array
    op.execute("""
        UPDATE draws 
        SET bonus_numbers = ARRAY[bonus]::integer[]
        WHERE bonus IS NOT NULL
    """)
    
    # Drop old bonus column
    op.drop_column('draws', 'bonus')


def downgrade():
    # Add back bonus column
    op.add_column('draws', sa.Column('bonus', sa.Integer(), nullable=True))
    
    # Migrate first element of bonus_numbers back to bonus
    op.execute("""
        UPDATE draws 
        SET bonus = bonus_numbers[1]
        WHERE bonus_numbers IS NOT NULL AND array_length(bonus_numbers, 1) > 0
    """)
    
    # Drop bonus_numbers column
    op.drop_column('draws', 'bonus_numbers')
