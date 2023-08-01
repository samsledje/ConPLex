"""Create the model output table.

Revision ID: 7d36484eafc7
Revises: 49539f93c63d
Create Date: 2023-07-22 00:00:48.512433

"""
from alembic import op
import sqlalchemy as sa


# revision identifiers, used by Alembic.
revision = '7d36484eafc7'
down_revision = '49539f93c63d'
branch_labels = None
depends_on = None


def upgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.create_table('ModelOutput',
    sa.Column('pairing_id', sa.Integer(), nullable=False),
    sa.Column('predictions', sa.LargeBinary(), nullable=False),
    sa.Column('drug_projections', sa.LargeBinary(), nullable=False),
    sa.Column('target_projections', sa.LargeBinary(), nullable=False),
    sa.Column('id', sa.Integer(), nullable=False),
    sa.Column('created_at', sa.DateTime(), nullable=False),
    sa.Column('updated_at', sa.DateTime(), nullable=False),
    sa.Column('deleted_at', sa.DateTime(), nullable=True),
    sa.ForeignKeyConstraint(['pairing_id'], ['Pairing.id'], ),
    sa.PrimaryKeyConstraint('id')
    )
    # ### end Alembic commands ###


def downgrade() -> None:
    # ### commands auto generated by Alembic - please adjust! ###
    op.drop_table('ModelOutput')
    # ### end Alembic commands ###
